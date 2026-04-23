"""Top-level ASR model assembly: encoder + decoder → SpeechEncoderDecoderModel.

Architecture pieces live in their own packages:
  - ``conformer_asr/downsamplers/`` — spectrogram → transformer-input bridges
  - ``conformer_asr/encoders/`` — speech encoders (currently Conformer)
  - ``conformer_asr/decoders/`` — autoregressive decoders with cross-attn
    (currently BART)

``build_model`` wires those together via the registries in each package and
wraps the result in ``SpeechEncoderDecoderModel`` (or ``ConformerAEDWithCTC``
if hybrid CTC/AED loss is enabled).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedTokenizerFast,
    SpeechEncoderDecoderModel,
)
from transformers.modeling_outputs import Seq2SeqLMOutput

from .config import ModelConfig
from .decoders import build_decoder
from .encoders import build_encoder


@dataclass
class ConformerAEDWithCTCOutput(Seq2SeqLMOutput):
    """``Seq2SeqLMOutput`` extended with the CTC branch tensors.

    ``loss`` is the blended AED+CTC total (what Trainer uses for backprop).
    ``aed_loss`` / ``ctc_loss`` are the raw (unblended) components so a custom
    ``compute_loss`` can re-apply label smoothing to only the AED branch and
    then re-blend. ``ctc_logits`` + ``encoder_attention_mask`` are exposed so
    downstream eval callbacks can greedy-decode without another encoder pass.
    """

    aed_loss: Optional[torch.FloatTensor] = None
    ctc_loss: Optional[torch.FloatTensor] = None
    ctc_logits: Optional[torch.FloatTensor] = None
    encoder_attention_mask: Optional[torch.Tensor] = None


class ConformerAEDWithCTC(SpeechEncoderDecoderModel):
    """Hybrid CTC/AED wrapper around ``SpeechEncoderDecoderModel``.

    Adds a linear CTC head on top of the encoder's hidden states. When
    ``labels`` are provided the total loss is
    ``(1 - ctc_weight) * AED + ctc_weight * CTC``; otherwise behaves exactly
    like the parent (so ``generate()`` is unaffected). The CTC head reuses
    the tokenizer's existing ``<pad>`` id as the blank symbol — pad never
    appears in real targets, so no vocab change is needed.
    """

    def __init__(
        self,
        config=None,
        encoder=None,
        decoder=None,
        ctc_weight: float = 0.3,
        ctc_blank_id: int = 0,
    ):
        super().__init__(config=config, encoder=encoder, decoder=decoder)
        vocab_size = self.decoder.config.vocab_size
        hidden_size = self.encoder.config.hidden_size
        self.ctc_head = nn.Linear(hidden_size, vocab_size)
        self.ctc_weight = float(ctc_weight)
        self.ctc_blank_id = int(ctc_blank_id)
        # Persist on the config so save_pretrained / from_pretrained round-trip
        # the CTC knobs without needing a custom config class.
        self.config.ctc_weight = self.ctc_weight
        self.config.ctc_blank_id = self.ctc_blank_id

    def _compute_ctc_loss(
        self,
        ctc_logits: torch.Tensor,
        encoder_attn_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # (B, T, V) -> (T, B, V) log-probs. Cast to fp32: CTC's log-sum-exp
        # is unstable under fp16/bf16, and cuDNN's CTC kernel silently falls
        # back to CPU if fed half-precision.
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1).float()

        if encoder_attn_mask is not None:
            input_lengths = encoder_attn_mask.sum(-1).long()
        else:
            input_lengths = torch.full(
                (ctc_logits.size(0),),
                ctc_logits.size(1),
                dtype=torch.long,
                device=ctc_logits.device,
            )

        # Drop -100 (pad-ignore) before handing targets to CTC; CTC treats the
        # flat concatenation of all target sequences plus per-sample lengths.
        label_mask = labels.ne(-100)
        label_lengths = label_mask.sum(-1).long()
        flat_targets = labels.masked_select(label_mask).long()

        return F.ctc_loss(
            log_probs,
            flat_targets,
            input_lengths,
            label_lengths,
            blank=self.ctc_blank_id,
            reduction="mean",
            zero_infinity=True,
        )

    def forward(
        self,
        input_features=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        labels=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Run the encoder ourselves (when not already cached by ``generate``)
        # so we can tap the hidden states for CTC without a second forward.
        encoder_was_run_here = encoder_outputs is None
        if encoder_was_run_here:
            encoder_outputs = self.encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                return_dict=True,
            )
        encoder_hidden = encoder_outputs[0]

        encoder_attn_mask = None
        if attention_mask is not None:
            encoder_attn_mask = self.encoder._get_feature_vector_attention_mask(
                encoder_hidden.size(1), attention_mask
            )

        # Derive decoder_input_ids from labels ourselves when the caller didn't
        # provide them. SpeechEncoderDecoderModel.forward does the same
        # shift-right internally, but whether it fires is gated on transformers
        # internals that have been brittle across versions — when it doesn't,
        # BartDecoder gets called with both input_ids and inputs_embeds unset
        # and raises. Doing it up front here makes the decoder call deterministic.
        if decoder_input_ids is None and labels is not None:
            pad_id = self.config.pad_token_id
            bos_id = self.config.decoder_start_token_id
            shifted = labels.new_zeros(labels.shape)
            shifted[:, 1:] = labels[:, :-1].clone()
            shifted[:, 0] = bos_id
            decoder_input_ids = shifted.masked_fill(shifted == -100, pad_id)

        # AED path — delegate to the parent, reusing our encoder outputs.
        aed_outputs = super().forward(
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            labels=labels,
            return_dict=True,
            **kwargs,
        )
        aed_loss = aed_outputs.loss

        # CTC head. Skip during ``generate``'s incremental decoder steps (when
        # encoder_outputs was cached and no labels are requested) — the head
        # would just produce unused logits step after step.
        ctc_logits = None
        ctc_loss = None
        if encoder_was_run_here or labels is not None:
            ctc_logits = self.ctc_head(encoder_hidden)
            if labels is not None:
                ctc_loss = self._compute_ctc_loss(ctc_logits, encoder_attn_mask, labels)

        if aed_loss is not None and ctc_loss is not None:
            total_loss = (1.0 - self.ctc_weight) * aed_loss + self.ctc_weight * ctc_loss
        else:
            total_loss = aed_loss

        output = ConformerAEDWithCTCOutput(
            loss=total_loss,
            logits=aed_outputs.logits,
            past_key_values=aed_outputs.past_key_values,
            decoder_hidden_states=aed_outputs.decoder_hidden_states,
            decoder_attentions=aed_outputs.decoder_attentions,
            cross_attentions=aed_outputs.cross_attentions,
            encoder_last_hidden_state=aed_outputs.encoder_last_hidden_state,
            encoder_hidden_states=aed_outputs.encoder_hidden_states,
            encoder_attentions=aed_outputs.encoder_attentions,
            aed_loss=aed_loss,
            ctc_loss=ctc_loss,
            ctc_logits=ctc_logits,
            encoder_attention_mask=encoder_attn_mask,
        )
        return output if return_dict else output.to_tuple()


def build_model(mcfg: ModelConfig, tokenizer: PreTrainedTokenizerFast) -> SpeechEncoderDecoderModel:
    """Construct a randomly-initialized encoder+decoder ASR model.

    Both encoder and decoder families are looked up by their registered string
    keys (``mcfg.encoder_type`` / ``mcfg.decoder_type``); the downsampler stem
    inside the encoder is looked up independently (``mcfg.downsampler``). See
    ``conformer_asr/encoders/``, ``conformer_asr/decoders/``, and
    ``conformer_asr/downsamplers/`` for registered options.
    """
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None or bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must define pad/bos/eos tokens.")

    encoder = build_encoder(mcfg)
    decoder = build_decoder(
        mcfg,
        vocab_size=len(tokenizer),
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
    )

    if getattr(mcfg, "ctc_enabled", False):
        model = ConformerAEDWithCTC(
            encoder=encoder,
            decoder=decoder,
            ctc_weight=mcfg.ctc_weight,
            ctc_blank_id=pad_id,
        )
    else:
        model = SpeechEncoderDecoderModel(encoder=encoder, decoder=decoder)

    # Wire special tokens at the top-level config so generate() picks them up.
    model.config.pad_token_id = pad_id
    model.config.decoder_start_token_id = bos_id
    model.config.bos_token_id = bos_id
    model.config.eos_token_id = eos_id
    model.config.vocab_size = decoder.config.vocab_size
    model.config.max_length = mcfg.decoder_max_position_embeddings

    return model
