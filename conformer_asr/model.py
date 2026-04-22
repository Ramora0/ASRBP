from __future__ import annotations

from transformers import (
    BartConfig,
    BartForCausalLM,
    PreTrainedTokenizerFast,
    SpeechEncoderDecoderModel,
    Wav2Vec2ConformerConfig,
    Wav2Vec2ConformerModel,
)

from .config import ModelConfig


class _CompatBartForCausalLM(BartForCausalLM):
    """Drop ``inputs_embeds`` when ``input_ids`` is also provided.

    SpeechEncoderDecoderModel in transformers >=4.51 passes both kwargs down
    to the decoder; BartDecoder raises on the combination. ``input_ids`` is
    authoritative — ``inputs_embeds`` would just be its embedding — so we
    prefer it.
    """

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if input_ids is not None and inputs_embeds is not None:
            inputs_embeds = None
        return super().forward(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )


def _build_encoder_config(mcfg: ModelConfig) -> Wav2Vec2ConformerConfig:
    return Wav2Vec2ConformerConfig(
        hidden_size=mcfg.encoder_hidden_size,
        num_hidden_layers=mcfg.encoder_num_hidden_layers,
        num_attention_heads=mcfg.encoder_num_attention_heads,
        intermediate_size=mcfg.encoder_intermediate_size,
        conv_depthwise_kernel_size=mcfg.encoder_conv_depthwise_kernel_size,
        position_embeddings_type="relative",
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
