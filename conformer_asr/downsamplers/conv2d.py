"""Conv2d subsampling stem — the default ASR downsampler.

The first two Conv2d layers use kernel ``(3, 3)`` + stride ``(2, 2)`` — they
downsample both time and mel axes, matching the standard
ESPnet / SpeechBrain / NeMo / Whisper stem and yielding ~4× time reduction on
80-mel input. Any additional convs beyond the first two use kernel ``(3, 1)``
+ stride ``(2, 1)`` — genuinely time-only: no kernel spread over mel, so the
mel axis is left untouched. Extra convs are useful to push overall time
downsampling further (8× at 3 convs, 16× at 4) for very long utterances or to
trade compute for longer effective receptive field. All convs use no padding;
time-stride follows ``l_out = (l_in - 1) // 2`` uniformly.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Downsampler


class Conv2dDownsampler(Downsampler):
    """Configurable stack of Conv2d(k=3, s=2) layers with ReLU.

    Args:
        n_mels: Input mel dimension.
        hidden: Output channel / transformer hidden size.
        num_convs: Number of Conv2d layers. First two are stride ``(2, 2)``
            (time + mel), the rest are stride ``(2, 1)`` (time-only). Must be
            ``>= 1``.
        dropout: Dropout applied after the post-flatten linear projection.
    """

    def __init__(
        self,
        n_mels: int,
        hidden: int,
        num_convs: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_convs < 1:
            raise ValueError(f"num_convs must be >= 1, got {num_convs}")

        self.n_mels = int(n_mels)
        self.hidden = int(hidden)
        self.num_convs = int(num_convs)
        # Convs beyond the first two keep the mel axis intact (stride 1 along mel).
        self._time_mel_convs = min(2, self.num_convs)
        self._time_only_convs = max(0, self.num_convs - 2)

        layers: list[nn.Module] = []
        in_channels = 1
        for i in range(self.num_convs):
            if i < 2:
                # Standard 2D subsample: time + mel both stride 2.
                kernel, stride = (3, 3), (2, 2)
            else:
                # Time-only: kernel width 1 along mel so no spread / shrinkage
                # along the mel axis, stride 1 along mel confirms no downsample.
                kernel, stride = (3, 1), (2, 1)
            layers.append(nn.Conv2d(in_channels, hidden, kernel_size=kernel, stride=stride))
            layers.append(nn.ReLU())
            in_channels = hidden
        self.convs = nn.Sequential(*layers)

        mel_after = self._mel_after_convs(self.n_mels)
        if mel_after <= 0:
            raise ValueError(
                f"n_mels={self.n_mels} too small for {self._time_mel_convs} Conv2d(k=3,s=(2,2)) layer(s)"
            )
        self.proj = nn.Linear(hidden * mel_after, hidden)
        self.dropout = nn.Dropout(dropout)

    def _mel_after_convs(self, n_mels: int) -> int:
        """Mel-axis size after the first ``_time_mel_convs`` stride-(2,2) layers.

        Each ``(k=3, s=2)`` step: ``l -> (l - 1) // 2``. Subsequent convs use
        kernel width 1 along mel, so they don't change the dimension.
        """
        m = n_mels
        for _ in range(self._time_mel_convs):
            m = (m - 1) // 2
        return m

    def output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """Per-sample valid time length after ``num_convs`` stride-2 layers.

        Every conv uses stride 2 along time regardless of mel stride, so the
        time arithmetic is just ``l -> (l - 1) // 2`` repeated ``num_convs``
        times. Clamped at 0 so pathologically short inputs don't underflow.
        """
        lengths = input_lengths
        for _ in range(self.num_convs):
            lengths = torch.clamp((lengths - 1) // 2, min=0)
        return lengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, n_mels) -> (B, 1, T, n_mels) -> conv stack -> (B, H, T', M')
        x = x.unsqueeze(1)
        x = self.convs(x)
        bsz, hidden, t_out, m_out = x.shape
        # (B, H, T', M') -> (B, T', H * M') -> (B, T', hidden)
        x = x.permute(0, 2, 1, 3).reshape(bsz, t_out, hidden * m_out)
        x = self.proj(x)
        x = self.dropout(x)
        return x
