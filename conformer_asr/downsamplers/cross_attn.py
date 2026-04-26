"""Cross-attention downsampler.

Full Conv2d stack (strided frontend + dilated layers, mel axis preserved
throughout) → flatten + project → pick every Nth → cross-attend the picked
subset to the full post-conv sequence, then hand the refined queries to the
encoder.

Pipeline inside ``forward``:
  1. ``convs``: a single ``nn.Sequential`` of Conv2d layers — same shape as
     ``Conv2dDownsampler`` plus extra dilated layers. The strided portion
     (default ``[[2,2],[2,2]]``) downsamples both time and mel, matching the
     c4x stem. The dilated portion (default ``dilations=(2, 2)``) keeps
     stride=1 on both axes — kernel-3 Conv2d with padding=(dilation, 1) and
     dilation=(dilation, 1) preserves both length AND mel dim while
     expanding the time-axis receptive field. Mel stays separate through
     every conv layer (4 layers total at default), so deep cross-mel
     interactions are learned just like c16x.
  2. Flatten the mel axis and apply a single ``Linear(hidden * mel_after,
     hidden)`` — placed after ALL convs, same as c16x's stem. Yields
     ``(B, T_cnn, hidden)`` at the post-frontend rate (4× downsampled
     by default, ≈25 Hz).
  3. Pick ``out[:, ::stride, :]`` to seed queries. With the default
     ``stride=4`` on top of the 4× conv stem, total downsampling is 16×
     (≈6.25 Hz on 100 Hz mels).
  4. ``num_layers`` pre-norm cross-attention blocks: queries = the picked
     subset (residually updated), keys/values = the full post-conv
     sequence. RoPE is applied to Q/K with positions taken from the
     **original post-frontend indices** — queries get ``[0, stride, 2*stride,
     ...]``, keys get ``[0, 1, ..., T_cnn-1]`` — so the relative offset
     under RoPE matches the true frame-rate gap. Padded positions are
     masked out of attention. The output projection is zero-init'd, so at
     step 0 the cross-attn contributes nothing — the model behaves as
     "full Conv2d stack → strided pick → conformer", which is structurally
     a 4-Conv2d stem plus a strided slice (the only difference vs c16x:
     the slice replaces c16x's last stride-2 conv).
  5. Return the refined queries.

Output length is a pure function of input length (only the strided convs
change rate; dilated layers preserve it; final ceil-divide by ``stride``).
This is a static downsampler. ``aux_loss`` stays at the base-class default
``None``.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Downsampler
from .conv2d import _length_after, _normalize_strides, _padding_for_stride


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """GPT-NeoX-style RoPE: rotate first half against second half.

    ``x``: ``(B, H, T, head_dim)``. ``cos``/``sin``: ``(1, 1, T, head_dim)``,
    where the last-axis halves are duplicates of the same per-pair frequency
    table — see ``_CrossAttnBlock._rope_cos_sin``.
    """
    return x * cos + _rotate_half(x) * sin


class _CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention sublayer with RoPE on Q/K (no FFN).

    The trailing position-wise FFN is omitted because the immediately-downstream
    Conformer layer starts with a macaron half-step FFN, which redoes the same
    work on the same residual stream.

    RoPE positions are passed in by the caller so we can encode the *original*
    post-frontend indices for both queries (``0, stride, ...``) and keys
    (``0, 1, ..., T_cnn-1``). That makes the Q·K relative offset under RoPE
    equal to the true frame-rate gap between a picked query and any key.
    """

    def __init__(
        self,
        hidden: int,
        num_heads: int,
        dropout: float,
        attn_dropout: float,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(f"hidden={hidden} not divisible by num_heads={num_heads}")
        head_dim = hidden // num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim={head_dim} must be even for RoPE")
        self.num_heads = int(num_heads)
        self.head_dim = head_dim
        self.attn_dropout = float(attn_dropout)

        self.norm_q = nn.LayerNorm(hidden)
        self.norm_kv = nn.LayerNorm(hidden)
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.resid_dropout = nn.Dropout(dropout)

        # RoPE inverse-frequency table (held in fp32; cos/sin are computed on
        # the fly per forward — the sequences here are short, so the cost is
        # negligible vs. precomputing a max-length cache).
        inv_freq = 1.0 / (
            float(rope_base)
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Zero-init the output projection so the cross-attn block contributes
        # nothing at step 0. The forward then equals the residual passthrough
        # (queries unchanged), letting the upstream conv stack — frontend +
        # dilation block — define the model's init behavior.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _rope_cos_sin(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # positions: (T,) long, on device. Returns cos, sin of shape
        # (1, 1, T, head_dim), matching dtype for the multiply against (B, H, T, head_dim).
        freqs = positions.to(self.inv_freq.dtype)[:, None] * self.inv_freq[None, :]  # (T, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim) — halves duplicated for _rotate_half
        cos = emb.cos().to(dtype)[None, None, :, :]
        sin = emb.sin().to(dtype)[None, None, :, :]
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_positions: torch.Tensor,
        k_positions: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, T_q, D = q.shape
        T_k = kv.shape[1]

        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)

        Q = self.q_proj(q_n).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv_n).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_n).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        cos_q, sin_q = self._rope_cos_sin(q_positions, Q.dtype)
        cos_k, sin_k = self._rope_cos_sin(k_positions, K.dtype)
        Q = _apply_rope(Q, cos_q, sin_q)
        K = _apply_rope(K, cos_k, sin_k)

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T_q, D)
        q = q + self.resid_dropout(self.out_proj(out))
        return q


class CrossAttnDownsampler(Downsampler):
    """Full Conv2d stack (mel preserved) → projection → strided pick → cross-attn.

    Args:
        n_mels: Mel-bin count.
        hidden: Channel count for every Conv2d layer AND the post-conv hidden
            size after projection (also the cross-attention dim).
        dropout: Dropout after the post-conv projection AND on the attention
            sublayer's residual branch.
        strides: Per-layer ``[time, mel]`` strides for the strided portion of
            the conv stack (same convention as ``Conv2dDownsampler``).
            Default ``[[2,2],[2,2]]`` matches c4x — 4× time reduction.
        dilations: Per-layer time-axis dilation factors for the post-strided
            Conv2d layers. Each is kernel-(3,3), stride-(1,1), padding=
            (dilation, 1), dilation=(dilation, 1) — preserves BOTH time
            (length) AND mel (dim) while expanding time RF. Default
            ``(2, 2)`` lifts each frame's time RF from 7 mel frames (c4x) to
            39 (slightly above c16x's 31). Mel stays separate through every
            conv layer, so cross-mel interactions are learned at depth like
            c16x. ``()`` disables this block.
        stride: Subsample factor applied after the conv stack to seed the
            cross-attention queries. With the default ``stride=4`` on top of
            the 4× strided stem, total compression is 16×.
        num_heads: Cross-attention heads.
        num_layers: Number of cross-attention blocks. Each block re-attends
            the (residually-updated) queries to the same key/value sequence.
        attn_dropout: Dropout applied inside ``scaled_dot_product_attention``.
    """

    def __init__(
        self,
        n_mels: int,
        hidden: int,
        dropout: float = 0.0,
        *,
        strides: Sequence[Sequence[int]] = ((2, 2), (2, 2)),
        dilations: Sequence[int] = (2, 2),
        stride: int = 4,
        num_heads: int = 4,
        num_layers: int = 1,
        attn_dropout: float = 0.0,
    ):
        super().__init__()
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.n_mels = int(n_mels)
        self.hidden = int(hidden)
        self.stride = int(stride)
        self.strides = _normalize_strides(strides)
        self.dilations = tuple(int(d) for d in dilations)
        for i, d in enumerate(self.dilations):
            if d < 1:
                raise ValueError(f"dilations[{i}] must be >= 1, got {d}")

        # Build one Conv2d stack: strided frontend, then dilated layers. Mel
        # axis is preserved through every layer (kernel-3 + stride-1 +
        # padding-1 on the mel side for the dilated portion; padding rule is
        # stride-conditional for the strided frontend). Final flatten +
        # projection happens AFTER the whole stack — same shape as c16x's
        # 4-Conv2d stem.
        layers: list[nn.Module] = []
        in_channels = 1
        for t, m in self.strides:
            t_pad = _padding_for_stride(t)
            m_pad = _padding_for_stride(m)
            layers.append(
                nn.Conv2d(
                    in_channels,
                    int(hidden),
                    kernel_size=3,
                    stride=(t, m),
                    padding=(t_pad, m_pad),
                )
            )
            layers.append(nn.ReLU())
            in_channels = int(hidden)
        for d in self.dilations:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    int(hidden),
                    kernel_size=3,
                    stride=1,
                    padding=(d, 1),
                    dilation=(d, 1),
                )
            )
            layers.append(nn.ReLU())
            in_channels = int(hidden)
        self.convs = nn.Sequential(*layers)

        # Mel after the strided portion (dilated layers preserve mel).
        mel_after = self.n_mels
        for _, m in self.strides:
            mel_after = _length_after(mel_after, m)
        if mel_after <= 0:
            raise ValueError(
                f"n_mels={self.n_mels} too small for mel strides {[m for _, m in self.strides]}"
            )
        self.mel_after = int(mel_after)

        # Single projection AFTER the whole conv stack (this is the *only*
        # point where mel collapses, mirroring c16x).
        self.proj = nn.Linear(int(hidden) * self.mel_after, int(hidden))
        self.proj_dropout = nn.Dropout(float(dropout))

        self.layers = nn.ModuleList([
            _CrossAttnBlock(
                hidden=int(hidden),
                num_heads=int(num_heads),
                dropout=float(dropout),
                attn_dropout=float(attn_dropout),
            )
            for _ in range(int(num_layers))
        ])

    def _post_conv_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """Per-sample valid time length after the conv stack.

        Only the strided layers change time length; dilated layers
        (stride=1, same-padding) preserve it.
        """
        lengths = input_lengths
        for t, _ in self.strides:
            lengths = _length_after(lengths, t)
        return lengths

    def output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """Conv-stack output lengths, then ceil-divide by ``stride``.

        Picking ``out[:, ::stride, :]`` from a length-``L`` sequence yields
        ``ceil(L / stride)`` elements; the same arithmetic gives the per-sample
        valid count.
        """
        post = self._post_conv_lengths(input_lengths)
        if self.stride == 1:
            return post
        return torch.div(post + self.stride - 1, self.stride, rounding_mode="floor")

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # 1. Full Conv2d stack: (B, T_mel, n_mels) -> (B, hidden, T_cnn, mel_after).
        x = x.unsqueeze(1)  # (B, 1, T_mel, n_mels)
        x = self.convs(x)
        bsz, hidden_dim, T_cnn, m_out = x.shape

        # 2. Flatten mel + project AFTER the whole stack (mel only collapses
        # here, mirroring c16x). (B, hidden, T_cnn, m_out) -> (B, T_cnn, hidden).
        h = x.permute(0, 2, 1, 3).reshape(bsz, T_cnn, hidden_dim * m_out)
        h = self.proj(h)
        h = self.proj_dropout(h)
        device = h.device

        # 3. Build a key-padding mask covering padded post-conv positions.
        # SDPA bool-mask convention: True = participates in attention.
        attn_mask: torch.Tensor | None = None
        if input_lengths is not None:
            post_lens = self._post_conv_lengths(input_lengths.to(device)).long()
            post_lens = post_lens.clamp(min=1, max=T_cnn)
            pos = torch.arange(T_cnn, device=device).unsqueeze(0)
            valid = pos < post_lens.unsqueeze(1)  # (B, T_cnn)
            attn_mask = valid[:, None, None, :]  # (B, 1, 1, T_k)

        # 4. Pick every ``stride``-th frame as queries. Track which post-conv
        # indices we picked — those are the RoPE positions for the queries,
        # so the relative offset Q[i] - K[j] under RoPE matches the true gap
        # between picked frame ``i*stride`` and key frame ``j``.
        if self.stride > 1:
            q = h[:, ::self.stride, :]
            q_positions = torch.arange(0, T_cnn, self.stride, device=device)
        else:
            q = h
            q_positions = torch.arange(T_cnn, device=device)
        k_positions = torch.arange(T_cnn, device=device)

        # 5. Cross-attend. With zero-init out_proj, this is the identity at
        # step 0 — the model behaves as "convs + strided pick + conformer"
        # until cross-attn weights move.
        for layer in self.layers:
            q = layer(q, h, q_positions, k_positions, attn_mask)

        return q
