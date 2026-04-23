from __future__ import annotations

import math

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (
    BartConfig,
    BartForCausalLM,
    PreTrainedTokenizerFast,
    SpeechEncoderDecoderModel,
    Wav2Vec2ConformerConfig,
    Wav2Vec2ConformerModel,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
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
    return Wav2Vec2ConformerConfig(
        hidden_size=mcfg.encoder_hidden_size,
        num_hidden_layers=mcfg.encoder_num_hidden_layers,
        num_attention_heads=mcfg.encoder_num_attention_heads,
        intermediate_size=mcfg.encoder_intermediate_size,
        conv_depthwise_kernel_size=mcfg.encoder_conv_depthwise_kernel_size,
        position_embeddings_type="rotary",
        mask_time_prob=mcfg.encoder_mask_time_prob,
        mask_feature_prob=mcfg.encoder_mask_feature_prob,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        feat_proj_dropout=0.0,
        layerdrop=0.0,
        apply_spec_augment=True,
    )


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

    encoder = Wav2Vec2ConformerModel(enc_cfg)
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
