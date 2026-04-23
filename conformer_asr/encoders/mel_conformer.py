"""Log-Mel + pluggable-downsampler Conformer encoder.

Takes a ``(B, T_mel, n_mels)`` log-Mel spectrogram (see
``conformer_asr/features.py``), runs it through a swappable ``Downsampler``
(defaults to the standard 2× ``Conv2d(k=3, s=2)`` stem), then the existing HF
Conformer blocks.

Slots into ``SpeechEncoderDecoderModel`` by exposing:
  - ``.config.hidden_size`` (inherited from ``Wav2Vec2ConformerConfig``)
  - ``forward(input_features, attention_mask=None, ...)`` returning a
    ``BaseModelOutput`` (the wrapper reads ``encoder_outputs[0]``).
  - ``_get_feature_vector_attention_mask(feat_len, attention_mask, ...)``
    which translates the pre-stem mask through the downsampler's time
    arithmetic so the transformer only attends to valid positions.

The downsampler is external state — pass it into ``__init__`` (normally via
``build_encoder`` which in turn calls ``build_downsampler``). The encoder
intentionally does not know what *kind* of downsampler it has, just the
``Downsampler`` interface.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2ConformerConfig
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices
from transformers.models.wav2vec2_conformer.modeling_wav2vec2_conformer import (
    Wav2Vec2ConformerEncoder,
    Wav2Vec2ConformerPreTrainedModel,
)

from ..downsamplers.base import Downsampler
from .sdpa_patch import install_sdpa_attention_patch

install_sdpa_attention_patch()


class MelConformerEncoder(Wav2Vec2ConformerPreTrainedModel):
    """Mel-input Conformer encoder with a pluggable ``Downsampler`` stem."""

    def __init__(self, config: Wav2Vec2ConformerConfig, downsampler: Downsampler):
        super().__init__(config)
        self.n_mels = int(config.n_mels)
        hidden = config.hidden_size

        self.downsampler = downsampler

        # Learnable mask embedding for SpecAugment-style time masking
        # (mirrors ``Wav2Vec2ConformerModel.masked_spec_embed``).
        self.masked_spec_embed = nn.Parameter(torch.zeros(hidden).uniform_())

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

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        """SpecAugment-style masking on post-stem hidden states, matching
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

        x = self.downsampler(input_features)
        t_out = x.shape[1]

        # Downsample the pre-stem attention mask so the Conformer blocks see
        # the correct (B, T') mask for self-attention and SpecAugment only
        # masks over valid positions.
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
