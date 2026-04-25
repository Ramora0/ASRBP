"""Pluggable spectrogram → transformer-input "downsampler" interface.

The downsampler sits between the log-Mel features (``(B, T_mel, n_mels)``) and
the transformer encoder (``(B, T', hidden)``). Every downsampler must answer
two questions:

1. Given a batch of padded mel features, produce hidden states.
2. Given per-sample input lengths (in mel frames), report the per-sample
   *output* lengths — so the encoder can build an attention mask that masks
   only valid positions.

For *static* downsamplers (e.g. fixed-stride convolutions) question 2 is
pure-function time arithmetic that doesn't need a forward pass. For *dynamic*
downsamplers (e.g. boundary-predictor pooling, where the per-sample output
length depends on data and model state) ``output_lengths`` is expected to
return values cached during the most-recent ``forward`` call. The encoder
always calls ``forward`` first and only then queries ``output_lengths``, so
the caching pattern is safe.

A downsampler may also carry an auxiliary training loss (e.g. the binomial
prior penalty on predicted boundary counts). It surfaces it via
``aux_loss()``; the model wrapper (``ConformerAEDWithCTC``) reads that after
the encoder's forward and adds it to the total loss for backprop. Static
downsamplers leave the default ``None`` return.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class Downsampler(nn.Module, ABC):
    """Abstract base class for spectrogram → transformer-input modules."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        """``(B, T_mel, n_mels) -> (B, T', hidden)``.

        ``input_lengths`` carries the per-sample valid mel-frame count for
        downsamplers that need it for masking (e.g. boundary-predictor
        pooling). Static downsamplers ignore it.
        """

    @abstractmethod
    def output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """Per-sample valid output length after the downsample stack.

        For static downsamplers this is pure-function time arithmetic — no
        module state, no forward pass. Dynamic downsamplers (e.g. boundary
        predictor) override to return values cached during the most-recent
        ``forward`` call.
        """

    def aux_loss(self) -> torch.Tensor | None:
        """Auxiliary loss from the most-recent forward pass.

        Returns ``None`` when the downsampler has no auxiliary objective
        (the default). Dynamic downsamplers override this to return a scalar
        tensor; the model wrapper adds it to the training total.
        """
        return None
