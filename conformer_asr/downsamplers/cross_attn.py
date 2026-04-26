"""Cross-attention downsampler.

Conv2d frontend → dilated-conv RF expansion → pick every Nth post-frontend
vector → cross-attend that subset to the full post-frontend sequence to
inject temporal context, then hand the refined queries to the encoder.

Pipeline inside ``forward``:
  1. ``Conv2dDownsampler`` frontend bridges ``(B, T_mel, n_mels)`` to
     ``(B, T_cnn, hidden)`` (default: 4× time reduction, matching c4x).
  2. ``dilation_block``: same-padded dilated Conv1d layers that **preserve
     rate** while expanding per-frame receptive field. Default
     ``dilations=(2, 2)`` on top of the c4x frontend lifts each frame's RF
     from 7 mel frames (c4x) to 39 mel frames — slightly above c16x's RF of
     31. Combined with the zero-init on the cross-attn output projection
     below, this makes the downsampler a strict superset of a 16×-RF conv
     stack at init.
  3. Pick ``out[:, ::stride, :]`` from the dilation-block output to seed
     queries — these are the eventual encoder inputs after refinement.
     With the default ``stride=4`` on top of the 4× conv stem, total
     downsampling is 16× (≈6.25 Hz on 100 Hz mels).
  4. ``num_layers`` pre-norm cross-attention blocks: queries = the picked
     subset (residually updated), keys/values = the full post-dilation
     sequence. RoPE is applied to Q/K with positions taken from the
     **original post-frontend indices** — queries get ``[0, stride, 2*stride,
     ...]``, keys get ``[0, 1, ..., T_cnn-1]`` — so the relative offset
     under RoPE matches the true frame-rate gap. Padded positions are
     masked out of attention. The output projection is zero-init'd, so at
     step 0 the cross-attn contributes nothing — the model behaves as
     "frontend → dilations → strided pick → conformer", a strict superset
     of a 16× conv stack.
  5. Return the refined queries.

Output length is a pure function of input length (frontend arithmetic +
ceil-divide by ``stride``; dilations preserve length) — this is a static
downsampler. ``aux_loss`` stays at the base-class default ``None``.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Downsampler
from .conv2d import Conv2dDownsampler


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
    """Conv2d frontend → dilated RF expansion → strided pick → cross-attn.

    Args:
        n_mels: Mel-bin count (passed straight through to the frontend).
        hidden: Transformer hidden size (frontend output dim and cross-attn
            dim).
        dropout: Dropout after the frontend's projection AND on the
            attention sublayer's residual branch.
        strides: Per-layer ``[time, mel]`` strides for the frontend Conv2d
            stack (same convention as ``Conv2dDownsampler``). Default
            ``[[2,2],[2,2]]`` matches c4x — 4× time reduction.
        dilations: Per-layer dilation factors for the post-frontend
            ``Conv1d`` stack. Each layer is kernel-3, stride-1, padding =
            dilation, so length is preserved (rate doesn't change). Default
            ``(2, 2)`` lifts each frame's RF from 7 mel frames (c4x) to 39
            (slightly above c16x's 31), giving the picked queries a
            depth-rich representation at init. ``()`` disables the block.
        stride: Subsample factor applied after the dilation block. ``out[:,
            ::stride, :]`` from the post-dilation sequence seeds the queries
            for cross-attention. With the default ``stride=4`` on top of the
            4× conv stem, total compression is 16×.
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
        self.frontend = Conv2dDownsampler(
            n_mels=n_mels, hidden=hidden, strides=strides, dropout=dropout
        )
        self.hidden = int(hidden)
        self.stride = int(stride)

        # Same-padded dilated 1D convs: preserve length, expand RF. Output
        # length stays equal to the frontend's, so output_lengths arithmetic
        # is unaffected by this block.
        dilation_layers: list[nn.Module] = []
        for d in dilations:
            d_int = int(d)
            if d_int < 1:
                raise ValueError(f"dilations entries must be >= 1, got {d_int}")
            dilation_layers.append(
                nn.Conv1d(
                    int(hidden),
                    int(hidden),
                    kernel_size=3,
                    padding=d_int,
                    dilation=d_int,
                )
            )
            dilation_layers.append(nn.ReLU())
        self.dilation_block: nn.Module = (
            nn.Sequential(*dilation_layers) if dilation_layers else nn.Identity()
        )

        self.layers = nn.ModuleList([
            _CrossAttnBlock(
                hidden=int(hidden),
                num_heads=int(num_heads),
                dropout=float(dropout),
                attn_dropout=float(attn_dropout),
            )
            for _ in range(int(num_layers))
        ])

    def output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """Frontend output lengths, then ceil-divide by ``stride``.

        Picking ``out[:, ::stride, :]`` from a length-``L`` sequence yields
        ``ceil(L / stride)`` elements; the same arithmetic gives the per-sample
        valid count.
        """
        post = self.frontend.output_lengths(input_lengths)
        if self.stride == 1:
            return post
        return torch.div(post + self.stride - 1, self.stride, rounding_mode="floor")

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # 1. Conv2d frontend: (B, T_mel, n_mels) -> (B, T_cnn, hidden).
        h = self.frontend(x)

        # 2. Dilation block: same-padded dilated Conv1d → preserves length,
        # expands per-frame RF. Conv1d expects (B, C, T), so transpose around it.
        h = self.dilation_block(h.transpose(1, 2)).transpose(1, 2)

        B, T_cnn, _ = h.shape
        device = h.device

        # 3. Build a key-padding mask covering padded post-frontend positions.
        # SDPA bool-mask convention: True = participates in attention.
        attn_mask: torch.Tensor | None = None
        if input_lengths is not None:
            post_lens = self.frontend.output_lengths(input_lengths.to(device)).long()
            post_lens = post_lens.clamp(min=1, max=T_cnn)
            pos = torch.arange(T_cnn, device=device).unsqueeze(0)
            valid = pos < post_lens.unsqueeze(1)  # (B, T_cnn)
            attn_mask = valid[:, None, None, :]  # (B, 1, 1, T_k); broadcasts over heads + queries

        # 4. Pick every ``stride``-th frame as queries. Track which post-frontend
        # indices we picked — those are the RoPE positions for the queries, so
        # the relative offset Q[i] - K[j] under RoPE matches the true gap
        # between picked frame ``i*stride`` and key frame ``j``.
        if self.stride > 1:
            q = h[:, ::self.stride, :]
            q_positions = torch.arange(0, T_cnn, self.stride, device=device)
        else:
            q = h
            q_positions = torch.arange(T_cnn, device=device)
        k_positions = torch.arange(T_cnn, device=device)

        # 5. Cross-attend (one or more blocks). With zero-init out_proj, this
        # is the identity at step 0 — the model behaves as "frontend +
        # dilations + strided pick + conformer" until cross-attn weights move.
        for layer in self.layers:
            q = layer(q, h, q_positions, k_positions, attn_mask)

        return q
