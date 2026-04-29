"""Registry of speech encoders.

Add a new architecture by:
  1. Dropping a module in this package that exposes a factory callable
     matching the ``(mcfg, tokenizer_info) -> nn.Module`` shape below (or
     whatever shape fits — see ``_build_mel_conformer``).
  2. Registering it in ``ENCODERS`` below under a string key.
  3. Referencing that key from ``model.encoder_type`` in the YAML config.

``build_encoder`` dispatches on ``mcfg.encoder_type``. Each builder owns its
own config-to-module translation so different encoder families can read
different subsets of ``ModelConfig``.
"""
from __future__ import annotations

from typing import Callable

import torch.nn as nn
from transformers import Wav2Vec2ConformerConfig

from ..config import ModelConfig
from ..downsamplers import build_downsampler
from .mel_conformer import MelConformerEncoder
from .preproc import SpecAugment


def _build_conformer_config(mcfg: ModelConfig) -> Wav2Vec2ConformerConfig:
    cfg = Wav2Vec2ConformerConfig(
        hidden_size=mcfg.encoder_hidden_size,
        num_hidden_layers=mcfg.encoder_num_hidden_layers,
        num_attention_heads=mcfg.encoder_num_attention_heads,
        intermediate_size=mcfg.encoder_intermediate_size,
        conv_depthwise_kernel_size=mcfg.encoder_conv_depthwise_kernel_size,
        position_embeddings_type="rotary",
        hidden_dropout=mcfg.encoder_hidden_dropout,
        attention_dropout=mcfg.encoder_attention_dropout,
        activation_dropout=mcfg.encoder_activation_dropout,
        feat_proj_dropout=0.0,
        layerdrop=mcfg.encoder_layerdrop,
        # SpecAugment is done externally by the ``SpecAugment`` module on
        # pre-stem log-Mel features (see encoders/preproc.py), so disable
        # HF's post-stem ``_mask_hidden_states`` entirely to avoid double
        # masking if a future refactor re-wires it.
        apply_spec_augment=False,
        mask_time_prob=0.0,
        mask_feature_prob=0.0,
    )
    # Stashed on the config so save_pretrained / from_pretrained preserve it —
    # MelConformerEncoder reads n_mels out of config at construction time.
    cfg.n_mels = mcfg.n_mels
    return cfg


def _build_spec_augment(mcfg: ModelConfig) -> SpecAugment:
    return SpecAugment(
        time_masks=mcfg.spec_aug_time_masks,
        time_length_low=mcfg.spec_aug_time_length_low,
        time_length_high=mcfg.spec_aug_time_length_high,
        feature_masks=mcfg.spec_aug_feature_masks,
        feature_length_low=mcfg.spec_aug_feature_length_low,
        feature_length_high=mcfg.spec_aug_feature_length_high,
    )


def _build_mel_conformer(mcfg: ModelConfig) -> MelConformerEncoder:
    enc_cfg = _build_conformer_config(mcfg)
    downsampler = build_downsampler(
        mcfg.downsampler,
        n_mels=mcfg.n_mels,
        hidden=mcfg.encoder_hidden_size,
        dropout=mcfg.encoder_hidden_dropout,
    )
    spec_augment = _build_spec_augment(mcfg)
    return MelConformerEncoder(
        enc_cfg,
        downsampler=downsampler,
        spec_augment=spec_augment,
        cross_attn_layer_indices=mcfg.cross_attn_layer_indices,
        cross_attn_num_heads=mcfg.cross_attn_num_heads,
        cross_attn_dropout=mcfg.cross_attn_dropout,
    )


ENCODERS: dict[str, Callable[[ModelConfig], nn.Module]] = {
    "conformer": _build_mel_conformer,
}


def build_encoder(mcfg: ModelConfig) -> nn.Module:
    """Instantiate the encoder named by ``mcfg.encoder_type``."""
    if mcfg.encoder_type not in ENCODERS:
        known = ", ".join(sorted(ENCODERS))
        raise ValueError(f"Unknown encoder_type {mcfg.encoder_type!r}; known: {known}")
    return ENCODERS[mcfg.encoder_type](mcfg)


__all__ = ["MelConformerEncoder", "ENCODERS", "build_encoder"]
