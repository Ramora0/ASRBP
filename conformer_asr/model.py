from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (
    BartConfig,
    BartForCausalLM,
    PreTrainedTokenizerFast,
    SpeechEncoderDecoderModel,
    Wav2Vec2ConformerConfig,
)
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutputWithCrossAttentions
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerEncoder,
    Wav2Vec2ConformerPreTrainedModel,
    Wav2Vec2ConformerSelfAttention,
)

from .config import ModelConfig


# HF's Wav2Vec2ConformerSelfAttention has no SDPA dispatch path — it always
# materializes the full (B, H, T, T) attention matrix via eager matmul/softmax.
# For T~1000 on V100 that's a serious HBM-bandwidth bottleneck. Route the
# rotary (and non-positional) path through F.scaled_dot_product_attention so
# the mem-efficient kernel can tile the attention. Relative-position path is
# left alone because it needs custom score composition SDPA can't express.
_ORIG_CONFORMER_SELF_ATTN_FORWARD = Wav2Vec2ConformerSelfAttention.forward


def _sdpa_self_attn_forward(
    self,
    hidden_states,
    attention_mask=None,
    relative_position_embeddings=None,
    output_attentions=False,
):
    # Fall back to eager when the caller wants attention weights (SDPA doesn't
    # surface them) or when we're on the Shaw-relative path.
    if output_attentions or self.position_embeddings_type == "relative":
        return _ORIG_CONFORMER_SELF_ATTN_FORWARD(
            self, hidden_states, attention_mask, relative_position_embeddings, output_attentions
        )

    batch_size, _, _ = hidden_states.size()
    query_key_states = hidden_states
    value_states = hidden_states

    if self.position_embeddings_type == "rotary":
        if relative_position_embeddings is None:
            raise ValueError(
                "`relative_position_embeddings` required when position_embeddings_type == 'rotary'"
            )
        query_key_states = self._apply_rotary_embedding(query_key_states, relative_position_embeddings)

    query = self.linear_q(query_key_states).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
    key = self.linear_k(query_key_states).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
    value = self.linear_v(value_states).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

    attn_output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=self.dropout.p if self.training else 0.0,
        scale=1.0 / math.sqrt(self.head_size),
    )
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
    attn_output = self.linear_out(attn_output)
    return attn_output, None


Wav2Vec2ConformerSelfAttention.forward = _sdpa_self_attn_forward


class MelConformerEncoder(Wav2Vec2ConformerPreTrainedModel):
    """Log-Mel + Conv2d-subsampling frontend feeding ``Wav2Vec2ConformerEncoder``.

    Replaces ``Wav2Vec2ConformerModel``'s raw-waveform CNN feature encoder +
    feature projection. Input is a ``(B, T_mel, n_mels)`` log-Mel spectrogram
    (see ``conformer_asr/features.py``). The stem is the standard ASR pattern
    used by ESPnet/SpeechBrain/NeMo/Whisper: two ``Conv2d(k=3, s=2, pad=0)``
    + ReLU → flatten → linear → dropout, giving ≈4× time downsampling. Then
    the existing Conformer blocks run on top.

    Slots into ``SpeechEncoderDecoderModel`` by exposing:
      - ``.config.hidden_size`` (inherited from ``Wav2Vec2ConformerConfig``)
      - ``forward(input_features, attention_mask=None, ...)`` returning a
        ``BaseModelOutput`` (wrapper reads ``encoder_outputs[0]``).
      - ``_get_feature_vector_attention_mask(feat_len, attention_mask, ...)``
        which downsamples a ``(B, T_mel)`` mask by the Conv2d time arithmetic.
    """

    def __init__(self, config: Wav2Vec2ConformerConfig):
        super().__init__(config)
        self.n_mels = int(config.n_mels)
        hidden = config.hidden_size

        self.subsample = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        # After two k=3 s=2 convs with no padding along the mel axis,
        # out_dim = ((n_mels - 1) // 2 - 1) // 2.
        mel_after = ((self.n_mels - 1) // 2 - 1) // 2
        if mel_after <= 0:
            raise ValueError(
                f"n_mels={self.n_mels} too small for two Conv2d(k=3,s=2) layers"
            )
        self.proj = nn.Linear(hidden * mel_after, hidden)
        self.dropout = nn.Dropout(config.hidden_dropout)

        # Learnable mask embedding for SpecAugment-style time masking
        # (mirrors ``Wav2Vec2ConformerModel.masked_spec_embed``).
        self.masked_spec_embed = nn.Parameter(torch.zeros(hidden).uniform_())

        self.encoder = Wav2Vec2ConformerEncoder(config)

        self.post_init()

    def _stem_output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """Per-sample valid output length after two Conv2d(k=3,s=2) along time.
        ``l -> (l - 1) // 2 -> ((l - 1) // 2 - 1) // 2``. Clamped at 0 so
        pathologically short inputs don't underflow to -1.
        """
        lengths = input_lengths
        for _ in range(2):
            lengths = torch.clamp((lengths - 1) // 2, min=0)
        return lengths

    def _get_feature_vector_attention_mask(
        self,
        feature_vector_length: int,
        attention_mask: torch.LongTensor,
        add_adapter=None,
    ) -> torch.BoolTensor:
        """Build a ``(B, feature_vector_length)`` **bool** mask from the
        pre-stem ``(B, T_mel)`` mask using the Conv2d time arithmetic.

        Bool dtype matches what the parent ``Wav2Vec2ConformerPreTrainedModel``
        returns — ``Wav2Vec2ConformerEncoder.forward`` does ``~attention_mask``
        which only gives the right (logical-NOT) answer on a bool tensor; on a
        long tensor ``~1`` is ``-2`` which is either an out-of-bounds or
        silently-wrapping negative index. ``add_adapter`` is accepted for
        interface compatibility and ignored.
        """
        input_lengths = attention_mask.sum(-1)
        output_lengths = self._stem_output_lengths(input_lengths)
        arange = torch.arange(feature_vector_length, device=attention_mask.device)
        return arange[None, :] < output_lengths[:, None]  # bool (B, feat_len)

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        """SpecAugment-style masking on the post-stem hidden states, matching
        ``Wav2Vec2ConformerModel._mask_hidden_states``."""
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states
        if not self.training:
            return hidden_states

        batch_size, sequence_length, hidden_size = hidden_states.size()

        if self.config.mask_time_prob > 0:
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(
                mask_time_indices, device=hidden_states.device, dtype=torch.bool
            )
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0:
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(
                mask_feature_indices, device=hidden_states.device, dtype=torch.bool
            )
            mask_feature_indices = mask_feature_indices[:, None, :].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    def forward(
        self,
        input_features: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # (B, T, n_mels) -> (B, 1, T, n_mels) -> subsample -> (B, H, T', M')
        x = input_features.unsqueeze(1)
        x = self.subsample(x)
        bsz, hidden, t_out, m_out = x.shape
        # (B, H, T', M') -> (B, T', H * M') -> (B, T', hidden)
        x = x.permute(0, 2, 1, 3).reshape(bsz, t_out, hidden * m_out)
        x = self.proj(x)
        x = self.dropout(x)

        # Downsample the pre-stem attention mask so the Conformer blocks see
        # the correct (B, T') mask for self-attention and SpecAugment only masks
        # over valid positions.
        encoder_attention_mask = None
        if attention_mask is not None:
            encoder_attention_mask = self._get_feature_vector_attention_mask(t_out, attention_mask)

        x = self._mask_hidden_states(x, attention_mask=encoder_attention_mask)

        return self.encoder(
            x,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class _CompatBartForCausalLM(BartForCausalLM):
    """Work around a long-standing interaction bug between
    ``SpeechEncoderDecoderModel`` and ``BartForCausalLM``: the outer model
    passes ``input_ids``, BartForCausalLM.forward internally also builds
    ``inputs_embeds`` and forwards both down to ``BartDecoder``, which rejects
    the combination. We bypass ``BartForCausalLM.forward`` entirely: pre-embed
    the ids ourselves and pass only ``inputs_embeds`` into the inner decoder,
    then apply the LM head. Loss is (re)computed here if ``labels`` is given,
    mirroring the parent's behavior.
    """

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        if input_ids is not None:
            bart_decoder = self.model.decoder
            embed_scale = getattr(bart_decoder, "embed_scale", 1.0)
            inputs_embeds = bart_decoder.embed_tokens(input_ids) * embed_scale
            input_ids = None

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model.decoder(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = CrossEntropyLoss()(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
            )

        if not return_dict:
            output = (logits,) + tuple(
                v for v in (
                    outputs.past_key_values,
                    outputs.hidden_states,
                    outputs.attentions,
                    outputs.cross_attentions,
                )
                if v is not None
            )
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )


def _build_encoder_config(mcfg: ModelConfig) -> Wav2Vec2ConformerConfig:
    cfg = Wav2Vec2ConformerConfig(
        hidden_size=mcfg.encoder_hidden_size,
        num_hidden_layers=mcfg.encoder_num_hidden_layers,
        num_attention_heads=mcfg.encoder_num_attention_heads,
        intermediate_size=mcfg.encoder_intermediate_size,
        conv_depthwise_kernel_size=mcfg.encoder_conv_depthwise_kernel_size,
        position_embeddings_type="rotary",
        mask_time_prob=mcfg.encoder_mask_time_prob,
        mask_feature_prob=mcfg.encoder_mask_feature_prob,
        hidden_dropout=mcfg.encoder_hidden_dropout,
        attention_dropout=mcfg.encoder_attention_dropout,
        activation_dropout=mcfg.encoder_activation_dropout,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        apply_spec_augment=True,
    )
    # Stashed on the config so save_pretrained / from_pretrained preserve it —
    # MelConformerEncoder reads n_mels out of config at construction time.
    cfg.n_mels = mcfg.n_mels
    return cfg


def _build_decoder_config(mcfg: ModelConfig, vocab_size: int, pad_id: int, bos_id: int, eos_id: int) -> BartConfig:
    return BartConfig(
        vocab_size=vocab_size,
        d_model=mcfg.decoder_d_model,
        decoder_layers=mcfg.decoder_layers,
        decoder_attention_heads=mcfg.decoder_attention_heads,
        decoder_ffn_dim=mcfg.decoder_ffn_dim,
        # BartForCausalLM still reads encoder_layers/heads for self-attention
        # inside BartDecoder when used standalone; set them to match the decoder
        # shape so the model instantiates without errors.
        encoder_layers=mcfg.decoder_layers,
        encoder_attention_heads=mcfg.decoder_attention_heads,
        encoder_ffn_dim=mcfg.decoder_ffn_dim,
        max_position_embeddings=mcfg.decoder_max_position_embeddings,
        dropout=mcfg.decoder_dropout,
        attention_dropout=mcfg.decoder_dropout,
        activation_dropout=mcfg.decoder_dropout,
        is_decoder=True,
        add_cross_attention=True,
        is_encoder_decoder=False,
        scale_embedding=True,
        pad_token_id=pad_id,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        decoder_start_token_id=bos_id,
    )


def build_model(mcfg: ModelConfig, tokenizer: PreTrainedTokenizerFast) -> SpeechEncoderDecoderModel:
    """Construct a randomly-initialized Conformer encoder + Bart decoder ASR model."""
    pad_id = tokenizer.pad_token_id
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None or bos_id is None or eos_id is None:
        raise ValueError("Tokenizer must define pad/bos/eos tokens.")

    enc_cfg = _build_encoder_config(mcfg)
    dec_cfg = _build_decoder_config(mcfg, len(tokenizer), pad_id, bos_id, eos_id)

    encoder = MelConformerEncoder(enc_cfg)
    decoder = _CompatBartForCausalLM(dec_cfg)
    model = SpeechEncoderDecoderModel(encoder=encoder, decoder=decoder)

    # Wire special tokens at the top-level config so generate() picks them up.
    model.config.pad_token_id = pad_id
    model.config.decoder_start_token_id = bos_id
    model.config.bos_token_id = bos_id
    model.config.eos_token_id = eos_id
    model.config.vocab_size = dec_cfg.vocab_size
    model.config.max_length = mcfg.decoder_max_position_embeddings

    return model
