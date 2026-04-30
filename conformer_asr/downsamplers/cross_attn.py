"""Cross-attention downsampler — feature extractor only.

Full Conv2d stack (strided frontend + dilated layers, mel axis preserved
throughout) → flatten + project → strided pick. Outputs the picked subset
directly; the cross-attention refinement *itself* lives in
``MelConformerEncoder`` as **interleaved** blocks injected between conformer
layers (see ``cross_attn_layer_indices``). This module's job is to produce:

  - the picked-rate query stream (the encoder's input)
  - the cached post-CNN feature map ``_post_cnn_features`` (K/V source for
    every interleaved XA block downstream)

Pipeline inside ``forward``:
  1. ``convs``: a single ``nn.Sequential`` of Conv2d layers — same shape as
     ``Conv2dDownsampler`` plus extra dilated layers. The strided portion
     (default ``[[2,2],[2,2]]``) downsamples both time and mel, matching the
     c4x stem. The dilated portion (default ``dilations=(2, 2)``) keeps
     stride=1 on both axes — kernel-3 Conv2d with padding=(0, 1) and
     dilation=(dilation, 1). Mel padding 1 preserves the mel dim; time
     padding 0 (valid conv) shrinks time by ``2*dilation`` per dilated
     layer. The valid-padding choice is what makes the dilated stack +
     final strided pick the exact à trous re-parameterization of a
     stride-2 cascade: with pad=0 the kept output positions read the same
     input windows as the equivalent strided-cascade outputs. Mel stays
     separate through every conv layer (4 layers total at default), so
     deep cross-mel interactions are learned just like c16x.
  2. Flatten the mel axis and apply a single ``Linear(hidden * mel_after,
     hidden)`` — placed after ALL convs, same as c16x's stem. Yields
     ``(B, T_cnn, hidden)`` at the post-frontend rate (4× downsampled
     by default, ≈25 Hz). Stash on ``_post_cnn_features``.
  3. Pick ``out[:, ::stride, :]`` to seed the encoder input. With the
     default ``stride=4`` on top of the 4× conv stem, total downsampling
     is 16× (≈6.25 Hz on 100 Hz mels). Return the picked tensor — no
     in-downsampler attention refinement.

Output length is a pure function of input length (strided convs apply
the standard ``(l-3)//s+1`` shrink; valid-pad dilated convs subtract
``2*dilation``; final ceil-divide by ``stride``).
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


class SharedKVProjector(nn.Module):
    """Encoder-level K/V projector shared across a group of cross-attn taps.

    Every ``CrossAttnBlock`` in a given group consumes the same K/V source
    (``downsampler._post_cnn_features``) at the same K positions
    (``arange(T_k)``), so applying ``LayerNorm + Linear(K) + Linear(V) +
    RoPE(K)`` per block was repeated work. Hoisting these into one module
    called once per encoder forward (per group) drops the per-tap marginal
    cost ~5×: K/V projections at the high rate ``T_k`` are the single
    biggest line item in a block, and at typical T_k = 4 × T_q they account
    for ~80% of one block's compute.

    The encoder owns a list of these projectors (``kv_projectors``); the
    contiguous depth-wise partition in ``block_to_group`` decides which
    projector each block reads from. ``n_groups == 1`` recovers the original
    "every tap shares one global K/V" design; ``n_groups == n_taps``
    recovers standard per-layer cross-attention.

    Per-block freedom on the *query* side (own ``norm_q``, ``q_proj``,
    ``out_proj``) is always preserved — taps within a group still attend
    to different "views" of the shared K/V via their own query function.

    Shape contract: ``forward(kv, k_positions)`` returns ``(K, V)`` of shape
    ``(B, num_heads, T_k, head_dim)`` with K already RoPE-rotated, ready to
    feed into ``F.scaled_dot_product_attention`` alongside a per-block Q.
    """

    def __init__(self, hidden: int, num_heads: int, rope_base: float = 10000.0):
        super().__init__()
        if hidden % num_heads != 0:
            raise ValueError(f"hidden={hidden} not divisible by num_heads={num_heads}")
        head_dim = hidden // num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim={head_dim} must be even for RoPE")
        self.num_heads = int(num_heads)
        self.head_dim = head_dim
        self.norm_kv = nn.LayerNorm(hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        inv_freq = 1.0 / (
            float(rope_base)
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _rope_cos_sin(
        self, positions: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # K positions in this codebase are always shared across the batch
        # (``arange(T_k)``), so only the 1D path is needed here.
        freqs = positions.to(self.inv_freq.dtype)[:, None] * self.inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype)[None, None, :, :]
        sin = emb.sin().to(dtype)[None, None, :, :]
        return cos, sin

    def forward(
        self, kv: torch.Tensor, k_positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T_k, _ = kv.shape
        kv_n = self.norm_kv(kv)
        K = self.k_proj(kv_n).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv_n).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        cos_k, sin_k = self._rope_cos_sin(k_positions, K.dtype)
        K = _apply_rope(K, cos_k, sin_k)
        return K, V


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention sublayer (Q-side only) with RoPE on Q.

    Used by ``MelConformerEncoder`` as an **interleaved** block inserted
    between conformer layers. K/V projections, K-side LayerNorm, and K-side
    RoPE have been factored out into ``SharedKVProjector`` and computed
    once per encoder forward — every tap reads the same ``_post_cnn_features``
    with identical k positions, so paying for K/V per block was repeated
    work. ``forward`` consumes precomputed ``K, V`` tensors of shape
    ``(B, H, T_k, head_dim)`` (K already RoPE-rotated).

    The trailing position-wise FFN is omitted because the immediately-
    downstream Conformer layer starts with a macaron half-step FFN, which
    redoes the same work on the same residual stream.

    RoPE on Q is applied here per-block; ``q_positions`` may be a shared 1D
    arange (static-stride downsamplers) or a per-sample 2D index (variable
    selectors like ``BPXADownsampler``). The Q·K relative offset under RoPE
    therefore tracks the true frame-rate gap between a picked query and any
    key — including the per-sample case where the picked positions vary.

    The output projection is zero-init'd: at step 0 the block contributes
    nothing, so a fresh model degrades to a plain conformer with the
    downsampler's strided-pick output as its input. The XA blocks then learn
    their contribution as training proceeds (track via
    ``train/xa_delta_ratio_l{i}`` in W&B).
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
        self.q_proj = nn.Linear(hidden, hidden)
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
        # positions: ``(T,)`` long for shared positions across the batch, OR
        # ``(B, T)`` long for per-sample positions (used by BP+XA's variable
        # selector — Q's RoPE position is the predicted post-CNN frame index,
        # different per sample).
        # Returns cos, sin shaped ``(1, 1, T, head_dim)`` (shared) or
        # ``(B, 1, T, head_dim)`` (per-batch), both broadcasting against
        # ``Q: (B, H, T, head_dim)``. dtype matches the multiply target.
        if positions.dim() == 1:
            freqs = positions.to(self.inv_freq.dtype)[:, None] * self.inv_freq[None, :]  # (T, head_dim/2)
            emb = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim) — halves duplicated for _rotate_half
            cos = emb.cos().to(dtype)[None, None, :, :]
            sin = emb.sin().to(dtype)[None, None, :, :]
        else:
            freqs = positions.to(self.inv_freq.dtype)[..., None] * self.inv_freq[None, None, :]  # (B, T, head_dim/2)
            emb = torch.cat([freqs, freqs], dim=-1)  # (B, T, head_dim)
            cos = emb.cos().to(dtype)[:, None, :, :]
            sin = emb.sin().to(dtype)[:, None, :, :]
        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        q_positions: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        B, T_q, D = q.shape

        q_n = self.norm_q(q)
        Q = self.q_proj(q_n).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)

        cos_q, sin_q = self._rope_cos_sin(q_positions, Q.dtype)
        Q = _apply_rope(Q, cos_q, sin_q)

        out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T_q, D)
        delta = self.resid_dropout(self.out_proj(out))
        # Stash batch-mean RMS of the residual contribution and the
        # pre-update query stream so the trainer can log how much each XA
        # layer is actually shifting the residual stream. Detached scalars
        # only — no graph retention. Fp32 for the reduction so bf16 autocast
        # doesn't clip the magnitude.
        with torch.no_grad():
            self._last_delta_rms = float(delta.detach().float().pow(2).mean().sqrt().item())
            self._last_q_rms = float(q.detach().float().pow(2).mean().sqrt().item())
        return q + delta


class CrossAttnDownsampler(Downsampler):
    """Full Conv2d stack (mel preserved) → projection → strided pick.

    Cross-attention refinement is no longer part of this module — it lives
    on ``MelConformerEncoder`` as interleaved blocks between conformer
    layers (see ``cross_attn_layer_indices`` on ``ModelConfig``). This
    downsampler produces both the encoder input (the picked stream) and
    the K/V source for those interleaved blocks (``_post_cnn_features``).

    Args:
        n_mels: Mel-bin count.
        hidden: Channel count for every Conv2d layer AND the post-conv hidden
            size after projection.
        dropout: Dropout after the post-conv projection.
        strides: Per-layer ``[time, mel]`` strides for the strided portion of
            the conv stack (same convention as ``Conv2dDownsampler``).
            Default ``[[2,2],[2,2]]`` matches c4x — 4× time reduction.
        dilations: Per-layer time-axis dilation factors for the post-strided
            Conv2d layers. Each is kernel-(3,3), stride-(1,1), padding=
            (0, 1), dilation=(dilation, 1) — valid on the time axis (shrinks
            time by ``2*dilation`` per layer), same on the mel axis
            (preserves mel dim). Valid time padding is what makes the
            full stack + strided pick a bit-exact à trous re-write of a
            stride-2 cascade. Default ``(2, 2)`` lifts each frame's time RF
            from 7 mel frames (c4x) to 39 (slightly above c16x's 31). Mel
            stays separate through every conv layer, so cross-mel
            interactions are learned at depth like c16x. ``()`` disables
            this block.
        stride: Subsample factor applied after the conv stack. With the
            default ``stride=4`` on top of the 4× strided stem, total
            compression is 16×.
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
    ):
        super().__init__()
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
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
                    padding=(0, 1),
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

    def _post_conv_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """Per-sample valid time length after the conv stack.

        Strided layers shrink by ``(l-3)//s+1`` (kernel 3, pad 0). Dilated
        layers are stride 1 with pad=(0,1) — valid on the time axis — so
        each shrinks time by ``2*dilation``.
        """
        lengths = input_lengths
        for t, _ in self.strides:
            lengths = _length_after(lengths, t)
        for d in self.dilations:
            shrink = 2 * int(d)
            if isinstance(lengths, torch.Tensor):
                lengths = torch.clamp(lengths - shrink, min=0)
            else:
                lengths = max(lengths - shrink, 0)
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

    def post_cnn_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """Per-sample valid time length at the post-conv / pre-pick stage.

        Exposed so an outer CTC head can supervise the post-CNN tensor
        (``ctc_input='post_cnn'``) and build a matching attention mask. Equal
        to ``output_lengths`` when ``stride == 1`` (no extra pick stage).
        """
        return self._post_conv_lengths(input_lengths)

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

        # Stash the post-CNN / pre-pick tensor. Two consumers downstream:
        #   - ``ConformerAEDWithCTC`` with ``ctc_input='post_cnn'`` taps it
        #     for a higher-rate CTC head.
        #   - ``MelConformerEncoder`` reads it as the K/V source for every
        #     interleaved cross-attn block in the conformer stack.
        # Plain attribute, kept out of state_dict.
        self._post_cnn_features = h

        # 3. Pick every ``stride``-th post-conv frame as the encoder input.
        # The interleaved cross-attn blocks downstream re-attend the
        # conformer's residual stream against the full ``h`` — which is why
        # the K/V cache above must stay live through the whole encoder pass.
        if self.stride > 1:
            return h[:, ::self.stride, :]
        return h
