"""Boundary-predictor downsampler.

Conv2d frontend → MLP-predicted segment boundaries → mean pooling. Ported
from ``SpeechbrainBP/boundary_predictor.py`` and folded into this codebase's
``Downsampler`` interface.

Pipeline inside ``forward``:
  1. Run the Conv2d stack only (no flatten-projection yet) to produce the
     pre-projection features ``(B, T_cnn, hidden * mel_after)``. The
     frontend's ``proj`` linear is deferred until step 6 — ``proj`` is
     linear so order-of-ops with the mean-pool is mathematically
     irrelevant, and deferring lets BP score boundaries on the ``mel_after``×
     richer pre-projection feature space rather than the rank-≤``hidden``
     view the proj would otherwise collapse it into.
  2. A 2-layer MLP with a small bottleneck (``bp_hidden``, default
     ``hidden // 4``) scores each frame on those rich features, sigmoid →
     per-frame boundary probability. The bottleneck keeps the BP MLP
     param count bounded — the input dim is ~``mel_after``× larger than
     in the original (post-projection) design.
  3. During training: ``RelaxedBernoulli`` rsample for differentiable soft
     boundaries, threshold at 0.5 for the hard pass, straight-through
     estimator wires gradients through the soft path. During eval: hard
     threshold on the sigmoid probability directly (no sampling noise).
  4. Force the last valid frame of each sample to be a boundary, mask
     padded frames to zero — guarantees at least one segment per sample.
  5. Mean-pool the pre-projection features by segment id (cumsum-derived).
  6. Apply the frontend's deferred ``proj`` + ``dropout`` to the pooled
     segments to land in ``(B, S, hidden)`` for the encoder.

Two pieces of state are written to the module on every ``forward`` and
consumed by the rest of the model:

  - ``output_lengths``: per-sample boundary count, returned by
    ``output_lengths()`` so the Conformer encoder's attention mask and the
    CTC head's input-lengths argument both see the right valid range.
  - ``aux_loss``: a scalar negative-log-prob under
    ``Binomial(n=actual_lens, p=prior)`` of the realized boundary count,
    scaled by ``loss_weight``. ``ConformerAEDWithCTC`` reads it via
    ``aux_loss()`` and adds it to the training total. Eval-time and
    forced-mode forwards return ``None`` (no aux loss).
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from .base import Downsampler
from .conv2d import Conv2dDownsampler


def _segment_indicator(boundaries: torch.Tensor) -> torch.Tensor | None:
    """Cumsum-based segment assignment.

    Returns ``[B, L, S]`` where ``out[b, t, s] == 0`` iff position ``t`` in
    sample ``b`` belongs to segment ``s``. ``S`` is the batch-max number of
    segments. Returns ``None`` if no boundary fired anywhere in the batch
    (caller falls back to an empty pooled output).
    """
    n_segments = int(boundaries.sum(dim=-1).max().item())
    if n_segments == 0:
        return None
    seg_idx = torch.arange(n_segments, device=boundaries.device)
    cum = boundaries.cumsum(1) - boundaries
    return seg_idx.view(1, 1, -1) - cum.unsqueeze(-1)


class BoundaryPredictorDownsampler(Downsampler):
    """Conv2d frontend → boundary-predictor mean pooling.

    Args:
        n_mels: Mel-bin count (passed straight through to the frontend).
        hidden: Transformer hidden size (frontend output dim and MLP width).
        dropout: Dropout after the frontend's projection (frontend kwarg).
        strides: Per-layer ``[time, mel]`` strides for the frontend Conv2d
            stack (same convention as ``Conv2dDownsampler``). Default
            ``[[1,2],[2,2]]`` matches the project's Whisper-style 2× stem
            (one stride-1 context-mixing layer + one stride-2 subsampling
            layer); paired with the default ``prior=0.25`` this targets
            ~8× total downsampling (2× conv × 4× BP).
        prior: Target boundary fraction in the binomial prior loss. With
            the default Whisper-style 2× frontend, ``0.25`` ⇒ ~8× total;
            ``0.5`` ⇒ ~4× total.
        temp: ``RelaxedBernoulli`` temperature for the soft-sample path
            during training. Higher ⇒ softer / noisier; lower ⇒ closer to
            the hard threshold (and lower-variance gradients). ``0.1`` is
            sharp — close to a deterministic threshold but still
            differentiable for the straight-through path.
        mlp_dropout: Dropout in front of the boundary MLP.
        loss_weight: Scalar multiplier applied to the binomial prior loss
            before it's surfaced via ``aux_loss()``. ``10.0`` matches the
            SpeechbrainBP recipe.
        boundary_mode: ``"learned"`` (default), ``"all"`` (every frame is
            a boundary — pass-through, debug), or ``"alternating"`` (every
            other frame — fixed 2× compression for sanity checks). Only
            ``"learned"`` carries the binomial loss.
        bp_hidden: Bottleneck width of the boundary MLP. The MLP is
            ``Linear(hidden * mel_after → bp_hidden) → GELU →
            Linear(bp_hidden → 1)``; keeping ``bp_hidden`` modest is what
            stops the first layer from blowing up, since its input dim is
            ``mel_after``× larger than ``hidden``. Default
            ``max(hidden // 4, 16)``.
    """

    def __init__(
        self,
        n_mels: int,
        hidden: int,
        dropout: float = 0.0,
        *,
        strides: Sequence[Sequence[int]] = ((1, 2), (2, 2)),
        prior: float = 0.25,
        temp: float = 0.1,
        mlp_dropout: float = 0.1,
        loss_weight: float = 10.0,
        boundary_mode: str = "learned",
        bp_hidden: int | None = None,
        init_prior_bias: bool = True,
        bp_isolated: bool = False,
    ):
        super().__init__()
        if boundary_mode not in {"learned", "all", "alternating"}:
            raise ValueError(
                f"boundary_mode must be one of {{learned, all, alternating}}, "
                f"got {boundary_mode!r}"
            )
        self.frontend = Conv2dDownsampler(
            n_mels=n_mels, hidden=hidden, strides=strides, dropout=dropout
        )
        self.hidden = int(hidden)
        self.prior = float(prior)
        self.temp = float(temp)
        self.loss_weight = float(loss_weight)
        self.boundary_mode = boundary_mode
        # Diagnostic switch: when True, the BP MLP is trained ONLY by the
        # binomial aux loss. The main-task gradient is cut at two places:
        # (1) ``boundaries.detach()`` before the mean-pool so the encoder's
        # gradient can't reach soft → logit → MLP; (2) ``h.detach()`` at the
        # MLP input so the aux gradient can't smear back into the conv stack.
        # Forward path is unchanged. Used to disambiguate "indirect path
        # overpowers binomial" from "binomial gradient is structurally bad
        # under STE+RelaxedBernoulli".
        self.bp_isolated = bool(bp_isolated)

        # BP scores boundaries on the conv stack's pre-projection features
        # (``hidden * mel_after`` per frame). Bottleneck the MLP at
        # ``bp_hidden`` so the first layer's params stay bounded — input
        # dim is ~``mel_after``× larger than the original post-projection
        # design, so a full-width MLP would explode.
        flat_dim = self.frontend.proj.in_features
        bp_hidden = int(bp_hidden) if bp_hidden is not None else max(self.hidden // 4, 16)
        self.bp_hidden = bp_hidden
        # LayerNorm the raw conv features before the BP MLP scores them.
        # The conv stack has no built-in normalization and its activation
        # scale drifts during training, which would otherwise feed the
        # scorer a non-stationary input distribution.
        self.pre_mlp_norm = nn.LayerNorm(flat_dim)
        self.boundary_mlp = nn.Sequential(
            nn.Linear(flat_dim, bp_hidden),
            nn.GELU(),
            nn.Linear(bp_hidden, 1),
        )
        self.init_prior_bias_enabled = bool(init_prior_bias)
        if self.init_prior_bias_enabled:
            self._init_prior_bias()
        self.mlp_dropout = nn.Dropout(p=mlp_dropout)
        # LayerNorm the pooled output before it enters the encoder. Without
        # this, the residual-stream scale at the encoder input is a direct
        # function of the compression rate (mean-pool variance ~ 1/N), which
        # creates a feedback loop: longer segments → smoother tokens → from-
        # scratch encoder prefers them → STE gradient rewards fewer
        # boundaries → compression collapses to ~1.
        self.post_pool_norm = nn.LayerNorm(self.hidden)

        self._cached_lengths: torch.LongTensor | None = None
        self._cached_aux_loss: torch.Tensor | None = None
        self._cached_post_frontend_lens: torch.LongTensor | None = None
        self._cached_input_lengths: torch.LongTensor | None = None

    def _init_prior_bias(self) -> None:
        # Start the boundary predictor at exactly `prior` regardless of the
        # frontend's init scale: zero the final weight and set the bias to
        # logit(prior), so every per-frame logit is `bias` at step 0 and
        # sigmoid(bias) == prior. Bias-only init would only hit prior on
        # average — the realized rate would drift with the variance of the
        # frontend's output. Single output unit ⇒ no symmetry issue.
        with torch.no_grad():
            p = min(max(self.prior, 1e-4), 1 - 1e-4)
            self.boundary_mlp[-1].weight.zero_()
            self.boundary_mlp[-1].bias.fill_(math.log(p / (1.0 - p)))

    def post_parent_init(self) -> None:
        # The parent encoder's HF post_init() recursively re-inits every
        # nn.Linear under this module — including boundary_mlp[-1] — wiping
        # the prior-targeted init done in __init__. Re-apply it here, unless
        # the user explicitly disabled the prior-bias init (debugging the
        # role of the initial bias in collapse dynamics).
        if self.init_prior_bias_enabled:
            self._init_prior_bias()

    # ------------------------------------------------------------- public API

    def aux_loss(self) -> torch.Tensor | None:
        return self._cached_aux_loss

    def last_stats(self) -> dict | None:
        """Boundary-rate / compression / aux-loss snapshot from the most-recent
        ``forward``. Returns ``None`` if forward hasn't run yet. All values are
        Python scalars so callers (e.g. the trainer's batched logger) can
        accumulate them across steps without touching CUDA tensors.

        Keys:
          - ``aux_loss``: ``loss_weight * binomial_NLL`` (or ``None`` outside
            training / in forced-boundary modes — same as ``aux_loss()``).
          - ``n_boundaries``: total predicted boundaries across the batch.
          - ``n_post_frontend``: total valid frames after the Conv2d frontend
            across the batch (denominator for ``realized_prior``).
          - ``n_input``: total valid mel-frame inputs across the batch
            (denominator for end-to-end compression).
        """
        if self._cached_lengths is None:
            return None
        aux = self._cached_aux_loss
        return {
            "aux_loss": float(aux.detach().float().item()) if aux is not None else None,
            "n_boundaries": int(self._cached_lengths.sum().item()),
            "n_post_frontend": (
                int(self._cached_post_frontend_lens.sum().item())
                if self._cached_post_frontend_lens is not None
                else 0
            ),
            "n_input": (
                int(self._cached_input_lengths.sum().item())
                if self._cached_input_lengths is not None
                else 0
            ),
        }

    def output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """Dynamic — returns the per-sample boundary counts cached by the
        most-recent ``forward``. Falls back to the frontend's static
        upper-bound when called before any forward (e.g. instrumentation).

        Under ``generate()`` with beam search, the decoder's cross-attention
        mask is built from a beam-expanded ``attention_mask`` (``B`` →
        ``B * num_beams`` along dim 0), but the encoder ran once on the
        un-expanded batch, so the cache holds ``B`` entries. We detect that
        case (matching length up to an integer multiplier) and
        ``repeat_interleave`` the cached lengths to match — HF beam-expands
        by the same pattern, so the per-beam alignment is preserved.
        """
        cached = self._cached_lengths
        if cached is None:
            return self.frontend.output_lengths(input_lengths)
        n_in = input_lengths.shape[0]
        n_cache = cached.shape[0]
        if n_in == n_cache:
            return cached
        if n_in > 0 and n_in % n_cache == 0:
            return cached.repeat_interleave(n_in // n_cache)
        # Mismatched in a way we don't recognize; fall back to the static
        # upper bound rather than crashing on a shape mismatch downstream.
        return self.frontend.output_lengths(input_lengths)

    def set_prior(self, prior: float) -> None:
        self.prior = float(prior)

    def set_temperature(self, temp: float) -> None:
        self.temp = float(temp)

    # ------------------------------------------------------------- internals

    def _apply_length_mask(
        self,
        boundaries: torch.Tensor,
        valid_mask: torch.Tensor,
        actual_lens: torch.Tensor,
        T_cnn: int,
    ) -> torch.Tensor:
        # Zero out boundaries in padded positions, then force the last valid
        # frame to be a boundary so every sample has at least one segment.
        B = boundaries.shape[0]
        boundaries = boundaries * valid_mask
        last_valid_idx = torch.clamp(actual_lens - 1, min=0, max=T_cnn - 1)
        batch_idx = torch.arange(B, device=boundaries.device)
        boundaries = boundaries.clone()
        boundaries[batch_idx, last_valid_idx] = 1.0
        return boundaries

    def _learned_boundaries(
        self,
        h: torch.Tensor,
        valid_mask: torch.Tensor,
        actual_lens: torch.Tensor,
        T_cnn: int,
    ) -> torch.Tensor:
        # Run the boundary head in fp32 so RelaxedBernoulli is stable under
        # bf16/fp16 autocast (the distribution rejects half-precision probs).
        h_in = h.detach() if self.bp_isolated else h
        logits = self.boundary_mlp(self.mlp_dropout(self.pre_mlp_norm(h_in))).squeeze(-1).float()
        probs = torch.sigmoid(logits)

        if self.training:
            dist = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp, probs=probs,
            )
            soft = dist.rsample()
        else:
            soft = probs

        hard_samples = (soft > 0.5).float()

        soft = self._apply_length_mask(soft, valid_mask, actual_lens, T_cnn)
        hard_samples = self._apply_length_mask(hard_samples, valid_mask, actual_lens, T_cnn)
        # Straight-through: forward path uses the hard threshold, backward
        # path flows through the soft (RelaxedBernoulli sample) values.
        return hard_samples - soft.detach() + soft

    def _forced_boundaries(
        self,
        T_cnn: int,
        every_n: int,
        valid_mask: torch.Tensor,
        actual_lens: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if every_n == 1:
            b = torch.ones(batch_size, T_cnn, device=device)
        else:
            b = torch.zeros(batch_size, T_cnn, device=device)
            b[:, every_n - 1::every_n] = 1.0
        return self._apply_length_mask(b, valid_mask, actual_lens, T_cnn)

    def _mean_pool(self, boundaries: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        B, _, D = h.shape
        ind = _segment_indicator(boundaries)
        if ind is None:
            return torch.zeros(B, 0, D, device=h.device, dtype=h.dtype)
        weights = 1 - ind
        weights = weights.masked_fill(ind != 0, 0.0)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-9)
        # einsum: (B, L, S) x (B, L, D) -> (B, S, D). Cast weights to h.dtype
        # so the matmul stays in the encoder's autocast precision.
        return torch.einsum("bls,bld->bsd", weights.to(h.dtype), h)

    def _binomial_loss(
        self,
        boundaries: torch.Tensor,
        actual_lens: torch.Tensor,
    ) -> torch.Tensor:
        # Per-sample -log P(num_boundaries | Binomial(n=actual_lens, p=prior)),
        # length-normalized then averaged over the batch. Clamp the prior
        # away from {0, 1} — at the endpoints the Binomial is degenerate
        # (log_prob is -inf for any k != n*prior) and would NaN-poison the
        # loss for typical batches where k drifts off the boundary.
        clamped_prior = min(max(self.prior, 1e-4), 1 - 1e-4)
        binomial = torch.distributions.binomial.Binomial(
            total_count=actual_lens.float(),
            probs=torch.tensor([clamped_prior], device=boundaries.device),
        )
        num_boundaries = boundaries.sum(dim=1)
        per_sample = -binomial.log_prob(num_boundaries) / actual_lens.float().clamp(min=1)
        return per_sample.mean()

    # --------------------------------------------------------------- forward

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # 1. Conv2d stack only — pre-projection features. Bypassing
        # ``frontend.forward`` lets BP score boundaries on the rich
        # ``hidden * mel_after`` features; the proj + dropout are
        # deferred to step 6 (proj is linear, so doing it after the
        # mean-pool gives an identical encoder input but a richer signal
        # for the BP scorer).
        conv_out = self.frontend.convs(x.unsqueeze(1))
        B, n_chan, T_cnn, m_out = conv_out.shape
        h_flat = conv_out.permute(0, 2, 1, 3).reshape(B, T_cnn, n_chan * m_out)
        device = h_flat.device

        # 2. Per-sample valid lengths after the conv stack.
        if input_lengths is None:
            actual_lens = torch.full((B,), T_cnn, dtype=torch.long, device=device)
        else:
            actual_lens = self.frontend.output_lengths(input_lengths.to(device)).long()
            actual_lens = torch.clamp(actual_lens, min=1, max=T_cnn)

        pos = torch.arange(T_cnn, device=device).unsqueeze(0)
        valid_mask = (pos < actual_lens.unsqueeze(1)).float()

        # 3. Predict / force boundaries on the pre-projection features.
        if self.boundary_mode == "all":
            boundaries = self._forced_boundaries(T_cnn, 1, valid_mask, actual_lens, B, device)
        elif self.boundary_mode == "alternating":
            boundaries = self._forced_boundaries(T_cnn, 2, valid_mask, actual_lens, B, device)
        else:
            boundaries = self._learned_boundaries(h_flat, valid_mask, actual_lens, T_cnn)

        # 4. Mean-pool the pre-projection features by segment id, then
        # apply the deferred proj + LayerNorm + dropout to land in (B, S, hidden).
        # When ``bp_isolated`` is on, route a detached copy into the pool so
        # the main-loss gradient stops at the segment assignments — only the
        # un-detached ``boundaries`` reaches the binomial below.
        boundaries_for_pool = boundaries.detach() if self.bp_isolated else boundaries
        pooled_flat = self._mean_pool(boundaries_for_pool, h_flat)
        pooled = self.frontend.proj(pooled_flat)
        pooled = self.post_pool_norm(pooled.float()).to(pooled.dtype)
        pooled = self.frontend.dropout(pooled)

        # 5. Cache per-sample output lengths, post-frontend lengths, and
        # (training-only) aux loss. The post-frontend / input-length caches
        # are read by ``last_stats()`` for boundary-rate logging.
        self._cached_lengths = boundaries.sum(dim=1).long().clamp(min=1)
        self._cached_post_frontend_lens = actual_lens.detach().long()
        if input_lengths is not None:
            self._cached_input_lengths = input_lengths.detach().long()
        else:
            self._cached_input_lengths = None
        if self.training and self.boundary_mode == "learned":
            self._cached_aux_loss = self.loss_weight * self._binomial_loss(boundaries, actual_lens)
        else:
            self._cached_aux_loss = None

        return pooled
