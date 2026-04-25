"""Conv2d subsampling stem — the default ASR downsampler.

A stack of Conv2d layers with per-layer ``[time, mel]`` strides specified in
the config. Kernel size is uniformly ``(3, 3)`` (the standard ASR/Whisper
choice). Padding is chosen per axis: 1 when that axis's stride is 1
(preserves dim, like Whisper), else 0 (matches the conventional no-pad
arithmetic for stride-2 subsampling).

Resulting per-layer output formula:
- stride 1, padding 1: ``l_out = l_in`` (dim preserved).
- stride ``s > 1``, padding 0: ``l_out = (l_in - 3) // s + 1``.

The standard ASR stem is two ``stride=(2, 2)`` layers — 4x time + ~4x mel
reduction on 80-mel input, matching ESPnet / SpeechBrain / NeMo / Whisper.
``stride=(2, 1)`` layers extend time downsampling without shrinking the mel
axis (8x at 3 layers, 16x at 4). Whisper-style context mixing — stride 1
with kernel 3 — preserves dim along that axis (kernel-3 with padding 1).
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .base import Downsampler


KERNEL = 3


def _normalize_strides(strides: Sequence[Sequence[int]]) -> list[tuple[int, int]]:
    if not strides:
        raise ValueError("strides must be a non-empty list of [time, mel] pairs")
    out: list[tuple[int, int]] = []
    for i, p in enumerate(strides):
        if len(p) != 2:
            raise ValueError(f"strides[{i}] must have length 2, got {list(p)!r}")
        t, m = int(p[0]), int(p[1])
        if t < 1 or m < 1:
            raise ValueError(f"strides[{i}] must be >= 1 in both axes, got ({t}, {m})")
        out.append((t, m))
    return out


def _padding_for_stride(s: int) -> int:
    """Padding rule: 1 when stride==1 (preserves dim), 0 otherwise."""
    return 1 if s == 1 else 0


def _length_after(l: int | torch.Tensor, s: int) -> int | torch.Tensor:
    """Single-axis output length for kernel=3, padding-by-stride rule."""
    if s == 1:
        return l
    if isinstance(l, torch.Tensor):
        return torch.clamp((l - KERNEL) // s + 1, min=0)
    return max((l - KERNEL) // s + 1, 0)


class Conv2dDownsampler(Downsampler):
    """Stack of Conv2d(kernel=3, stride=...) layers with stride-conditional padding.

    Args:
        n_mels: Input mel dimension.
        hidden: Output channel / transformer hidden size.
        strides: One ``[time, mel]`` pair per Conv2d layer. Must be non-empty.
        dropout: Dropout applied after the post-flatten linear projection.
    """

    def __init__(
        self,
        n_mels: int,
        hidden: int,
        strides: Sequence[Sequence[int]] = ((2, 2), (2, 2)),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_mels = int(n_mels)
        self.hidden = int(hidden)
        self.strides = _normalize_strides(strides)

        layers: list[nn.Module] = []
        in_channels = 1
        for t, m in self.strides:
            padding = (_padding_for_stride(t), _padding_for_stride(m))
            layers.append(
                nn.Conv2d(in_channels, hidden, kernel_size=KERNEL, stride=(t, m), padding=padding)
            )
            layers.append(nn.ReLU())
            in_channels = hidden
        self.convs = nn.Sequential(*layers)

        mel_after = self._mel_after_convs(self.n_mels)
        if mel_after <= 0:
            raise ValueError(
                f"n_mels={self.n_mels} too small for mel strides {[m for _, m in self.strides]}"
            )
        self.proj = nn.Linear(hidden * mel_after, hidden)
        self.dropout = nn.Dropout(dropout)

    def _mel_after_convs(self, n_mels: int) -> int:
        l = n_mels
        for _, m in self.strides:
            l = _length_after(l, m)
        return l

    def output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """Per-sample valid time length after the configured convs.

        Stride-1 axes preserve length (kernel 3, padding 1); stride-``s`` axes
        follow ``(l - 3) // s + 1`` (kernel 3, padding 0). Clamped at 0.
        """
        lengths = input_lengths
        for t, _ in self.strides:
            lengths = _length_after(lengths, t)
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
