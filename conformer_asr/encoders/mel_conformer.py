"""Log-Mel + pluggable-downsampler Conformer encoder.

Takes a ``(B, T_mel, n_mels)`` log-Mel spectrogram (see
``conformer_asr/features.py``), runs it through:

  1. ``InputNormalization`` — per-bin running mean/var, updated for the
     first N epochs then frozen (see ``encoders/preproc.py``).
  2. ``SpecAugment`` — pre-stem deterministic K time + K feature masks
     with zero-fill (see ``encoders/preproc.py``). No-op outside training
     or before ``spec_aug_warmup_steps``.
  3. A swappable ``Downsampler`` (defaults to 2× ``Conv2d(k=3, s=2)``).
  4. The existing HF ``Wav2Vec2ConformerEncoder`` blocks.

Slots into ``SpeechEncoderDecoderModel`` by exposing:
  - ``.config.hidden_size`` (inherited from ``Wav2Vec2ConformerConfig``)
  - ``forward(input_features, attention_mask=None, ...)`` returning a
    ``BaseModelOutput`` (the wrapper reads ``encoder_outputs[0]``).
  - ``_get_feature_vector_attention_mask(feat_len, attention_mask, ...)``
    which translates the pre-stem mask through the downsampler's time
    arithmetic so the transformer only attends to valid positions.

The downsampler and spec-augment modules are external state — pass them
into ``__init__`` (normally via ``build_encoder``). The encoder doesn't
care which concrete classes they are, just the interface contracts.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2ConformerConfig
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerEncoder,
    Wav2Vec2ConformerPreTrainedModel,
)

from ..downsamplers.base import Downsampler
from .preproc import InputNormalization, SpecAugment
from .sdpa_patch import install_sdpa_attention_patch

install_sdpa_attention_patch()


class MelConformerEncoder(Wav2Vec2ConformerPreTrainedModel):
    """Mel-input Conformer encoder with a pluggable ``Downsampler`` stem."""

    def __init__(
        self,
        config: Wav2Vec2ConformerConfig,
        downsampler: Downsampler,
        spec_augment: SpecAugment,
    ):
        super().__init__(config)
        self.n_mels = int(config.n_mels)

        self.input_norm = InputNormalization(n_features=self.n_mels)
        self.spec_augment = spec_augment
        self.downsampler = downsampler
        self.encoder = Wav2Vec2ConformerEncoder(config)

        self.post_init()

    def _get_feature_vector_attention_mask(
        self,
        feature_vector_length: int,
        attention_mask: torch.LongTensor,
        add_adapter=None,
    ) -> torch.BoolTensor:
        """Build a ``(B, feature_vector_length)`` **bool** mask from the
        pre-stem ``(B, T_mel)`` mask using the downsampler's time arithmetic.

        Bool dtype matches what the parent ``Wav2Vec2ConformerPreTrainedModel``
        returns — ``Wav2Vec2ConformerEncoder.forward`` does ``~attention_mask``
        which only gives the right (logical-NOT) answer on a bool tensor; on a
        long tensor ``~1`` is ``-2`` which is either an out-of-bounds or
        silently-wrapping negative index. ``add_adapter`` is accepted for
        interface compatibility and ignored.
        """
        input_lengths = attention_mask.sum(-1)
        output_lengths = self.downsampler.output_lengths(input_lengths)
        arange = torch.arange(feature_vector_length, device=attention_mask.device)
        return arange[None, :] < output_lengths[:, None]

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

        # 1. Per-bin normalization (stats update during training until frozen).
        # 2. Pre-stem SpecAugment (training only, zero-fill after normalization
        #    so masked regions land on the per-bin mean).
        x = self.input_norm(input_features, attention_mask=attention_mask)
        x = self.spec_augment(x, attention_mask=attention_mask)

        # 3. Downsampler stem: (B, T_mel, n_mels) → (B, T', hidden).
        # Dynamic downsamplers (e.g. boundary predictor) need per-sample valid
        # lengths to mask padding before pooling — pass them through; static
        # downsamplers ignore the kwarg.
        input_lengths = attention_mask.sum(-1) if attention_mask is not None else None
        x = self.downsampler(x, input_lengths=input_lengths)
        t_out = x.shape[1]

        # Downsample the pre-stem attention mask so the Conformer blocks see
        # the correct (B, T') mask for self-attention.
        encoder_attention_mask = None
        if attention_mask is not None:
            encoder_attention_mask = self._get_feature_vector_attention_mask(t_out, attention_mask)

        return self.encoder(
            x,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
