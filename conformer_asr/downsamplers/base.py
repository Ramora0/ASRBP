"""Pluggable spectrogram → transformer-input "downsampler" interface.

The downsampler sits between the log-Mel features (``(B, T_mel, n_mels)``) and
the transformer encoder (``(B, T', hidden)``). Every downsampler must answer
two questions:

1. Given a batch of padded mel features, produce hidden states.
2. Given per-sample input lengths (in mel frames), predict the per-sample
   *output* lengths — so the encoder can build an attention mask that masks
   only valid positions.

Question 2 has to be answered without actually running the module (we need the
mask before / alongside the forward pass), so every implementation has to know
its own time arithmetic analytically.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Downsampler(nn.Module, ABC):
    """Abstract base class for spectrogram → transformer-input modules."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T_mel, n_mels) -> (B, T', hidden)``."""

    @abstractmethod
    def output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """Per-sample valid output length after the downsample stack.

        Pure-function time arithmetic — no module state, no forward pass. Used
        by the encoder to build a ``(B, T')`` attention mask from the original
        ``(B, T_mel)`` mask.
        """
