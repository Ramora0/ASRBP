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
        ctc_tap_after_layer: int | None = None,
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

        # Optional intermediate tap. With ``ctc_tap_after_layer=N``, forward()
        # stashes a ``(B, T_at_N, hidden)`` tensor on ``_post_cnn_features``
        # after the first N strided conv layers, projected into ``hidden`` by
        # its own ``ctc_tap_proj`` (separate weights from the final ``proj``).
        # Lets a cnns/ config wear two hats: the encoder consumes the
        # full-stack output (deep / coarse), while a CTC head supervises the
        # intermediate output (shallow / fine).
        self.ctc_tap_after_layer: int | None
        if ctc_tap_after_layer is None:
            self.ctc_tap_after_layer = None
            self.ctc_tap_proj = None
        else:
            n = int(ctc_tap_after_layer)
            if not (1 <= n < len(self.strides)):
                raise ValueError(
                    f"ctc_tap_after_layer must be in [1, {len(self.strides)-1}] "
                    f"(strict — at the boundary it would equal the final output); "
                    f"got {n} with {len(self.strides)} layers."
                )
            self.ctc_tap_after_layer = n
            mel_at_tap = self.n_mels
            for _, m in self.strides[:n]:
                mel_at_tap = _length_after(mel_at_tap, m)
            if mel_at_tap <= 0:
                raise ValueError(
                    f"n_mels={self.n_mels} too small for mel strides "
                    f"{[m for _, m in self.strides[:n]]} at intermediate tap"
                )
            self.ctc_tap_proj = nn.Linear(hidden * int(mel_at_tap), hidden)

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

    def post_cnn_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """Per-sample valid time length at the intermediate CTC tap.

        Same arithmetic as ``output_lengths`` but stops after the first
        ``ctc_tap_after_layer`` strided layers. Only meaningful when the tap
        is configured.
        """
        if self.ctc_tap_after_layer is None:
            raise RuntimeError(
                "post_cnn_lengths called but ctc_tap_after_layer is None — "
                "Conv2dDownsampler has no intermediate tap configured."
            )
        lengths = input_lengths
        for t, _ in self.strides[: self.ctc_tap_after_layer]:
            lengths = _length_after(lengths, t)
        return lengths

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # ``input_lengths`` is unused — Conv2d output length is a pure function
        # of the input length (see ``output_lengths``); the encoder builds the
        # attention mask separately. Accepted only to match the base interface.
        del input_lengths
        # (B, T, n_mels) -> (B, 1, T, n_mels) -> conv stack -> (B, H, T', M')
        x = x.unsqueeze(1)
        if self.ctc_tap_after_layer is None:
            x = self.convs(x)
        else:
            # Iterate so we can tap the intermediate post-Nth-layer tensor.
            # ``self.convs`` is ``Sequential(Conv, ReLU, Conv, ReLU, ...)`` —
            # two children per strided layer.
            n_pre = self.ctc_tap_after_layer * 2
            children = list(self.convs.children())
            for layer in children[:n_pre]:
                x = layer(x)
            bsz, hidden_dim, t_tap, m_tap = x.shape
            tap = x.permute(0, 2, 1, 3).reshape(bsz, t_tap, hidden_dim * m_tap)
            self._post_cnn_features = self.ctc_tap_proj(tap)
            for layer in children[n_pre:]:
                x = layer(x)
        bsz, hidden, t_out, m_out = x.shape
        # (B, H, T', M') -> (B, T', H * M') -> (B, T', hidden)
        x = x.permute(0, 2, 1, 3).reshape(bsz, t_out, hidden * m_out)
        x = self.proj(x)
        x = self.dropout(x)
        return x
