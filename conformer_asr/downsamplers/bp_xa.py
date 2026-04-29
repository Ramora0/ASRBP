"""BP variable selector + cross-attention framework — content-aware token picking.

Same architecture as ``CrossAttnDownsampler`` (conv stack + projection +
post-CNN K/V cache feeding the encoder's interleaved cross-attention blocks),
but the uniform strided pick ``h[:, ::stride, :]`` is replaced with a learned
per-frame keep-probability scorer (BP-style). The encoder's interleaved XA
blocks are unchanged — they read the same post-CNN K/V cache, but Q's RoPE
positions are now the actual predicted frame indices (per-sample, variable
count) instead of the fixed ``[0, stride, 2*stride, ...]`` pattern.

This is an ADDITIVE feature alongside ``CrossAttnDownsampler``; the strided-
sampling path keeps working unchanged. The only encoder hook is a
``getattr(downsampler, "_kept_indices", None)`` check that falls back to the
stride-based positions when not present.

Pipeline:
  1. Conv stack (strided + dilated, mel preserved) → projection →
     ``h: (B, T_cnn, hidden)``. Stash on ``_post_cnn_features`` for the
     encoder's interleaved XA taps.
  2. BP scorer (LayerNorm → MLP → sigmoid) on post-projection features.
  3. ``RelaxedBernoulli`` rsample (training) / hard threshold (eval), with
     length masking and forced last-valid-frame keep. Straight-through gate.
  4. Sort-based gather: ``kept_idx[b, k]`` = post-CNN frame index of the
     k-th kept token in sample b, padded with non-kept indices for samples
     with fewer kept tokens than the batch maximum.
  5. ``queries = h.gather(...) * gate.gather(...)`` — STE multiply: forward
     equals plain gathering at kept positions (gate=1) and zero elsewhere
     (gate=0); backward routes the soft probability gradient back to the
     scorer.
  6. Stash ``_kept_indices`` so the encoder's interleaved XA reads them as
     Q's per-sample RoPE positions.

Aux loss: same binomial prior NLL as ``BoundaryPredictorDownsampler``.
``last_stats()``: BP-shaped (``aux_loss``, ``n_boundaries``,
``n_post_frontend``, ``n_input``). XA delta_rms keys come from the
encoder's interleaved blocks unchanged.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

from .base import Downsampler
from .conv2d import _length_after, _normalize_strides, _padding_for_stride


class BPXADownsampler(Downsampler):
    """BP-predicted token selection on the CrossAttn-style XA framework.

    Args:
        n_mels: Mel-bin count.
        hidden: Channel count for every Conv2d layer AND post-conv hidden size.
        dropout: Dropout after the post-conv projection.
        strides: Per-layer ``[time, mel]`` strides for the strided portion of
            the conv stack. Default ``[[2,2],[2,2]]`` matches c4x — 4× time
            reduction.
        dilations: Per-layer time-axis dilation factors for the post-strided
            layers (each kernel-(3,3), stride-(1,1), padding=(0,1), valid on
            time, same on mel). Default ``(1, 2)`` gives the dense 31-frame
            post-CNN RF used by xa_16x.
        prior: Target keep-fraction in the binomial prior loss. With the
            default 4× conv stem, ``prior=0.25`` ⇒ ~16× total compression.
        temp: ``RelaxedBernoulli`` temperature for the soft-sample path
            (training only). Lower ⇒ closer to a hard threshold; higher ⇒
            more sampling noise.
        mlp_dropout: Dropout in front of the boundary MLP.
        loss_weight: Scalar multiplier applied to the binomial NLL before it
            surfaces via ``aux_loss()``. ``10.0`` matches the BP recipe.
        boundary_mode: ``"learned"`` (default), ``"all"`` (every valid frame
            kept — pass-through past the conv stage; debug), or
            ``"alternating"`` (every other valid frame kept — fixed 2×
            picking past the conv stage; sanity check). Only ``"learned"``
            carries the aux loss.
        bp_hidden: Bottleneck width of the boundary MLP. Default
            ``max(hidden // 4, 16)``.
    """

    def __init__(
        self,
        n_mels: int,
        hidden: int,
        dropout: float = 0.0,
        *,
        strides: Sequence[Sequence[int]] = ((2, 2), (2, 2)),
        dilations: Sequence[int] = (1, 2),
        prior: float = 0.25,
        temp: float = 0.1,
        mlp_dropout: float = 0.1,
        loss_weight: float = 10.0,
        boundary_mode: str = "learned",
        bp_hidden: int | None = None,
    ):
        super().__init__()
        if boundary_mode not in {"learned", "all", "alternating"}:
            raise ValueError(
                f"boundary_mode must be one of {{learned, all, alternating}}, "
                f"got {boundary_mode!r}"
            )
        self.n_mels = int(n_mels)
        self.hidden = int(hidden)
        self.strides = _normalize_strides(strides)
        self.dilations = tuple(int(d) for d in dilations)
        for i, d in enumerate(self.dilations):
            if d < 1:
                raise ValueError(f"dilations[{i}] must be >= 1, got {d}")
        self.prior = float(prior)
        self.temp = float(temp)
        self.loss_weight = float(loss_weight)
        self.boundary_mode = boundary_mode
        # Average compression past the conv stage. Exposed as ``stride`` so
        # ``MelConformerEncoder._encoder_with_xa`` has a sensible fallback
        # when ``_kept_indices`` isn't set yet (instrumentation, edge cases).
        # The encoder reads ``_kept_indices`` first.
        self.stride = max(1, int(round(1.0 / max(self.prior, 1e-6))))

        # Conv stack: same shape as CrossAttnDownsampler (strided frontend +
        # dilated layers, mel preserved through every layer; flatten +
        # projection only at the end).
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
                f"n_mels={self.n_mels} too small for mel strides "
                f"{[m for _, m in self.strides]}"
            )
        self.mel_after = int(mel_after)

        self.proj = nn.Linear(int(hidden) * self.mel_after, int(hidden))
        self.proj_dropout = nn.Dropout(float(dropout))

        # BP scorer on POST-projection features (single-path forward —
        # simpler than BP's pre-projection scorer; trades a slightly weaker
        # signal for a shared representation across the scorer and the
        # downstream encoder/XA blocks).
        bp_hidden_dim = (
            int(bp_hidden) if bp_hidden is not None else max(int(hidden) // 4, 16)
        )
        self.bp_hidden = bp_hidden_dim
        self.pre_mlp_norm = nn.LayerNorm(int(hidden))
        self.boundary_mlp = nn.Sequential(
            nn.Linear(int(hidden), bp_hidden_dim),
            nn.GELU(),
            nn.Linear(bp_hidden_dim, 1),
        )
        self._init_prior_bias()
        self.mlp_dropout = nn.Dropout(p=float(mlp_dropout))

        # Per-forward state
        self._cached_lengths: torch.LongTensor | None = None
        self._cached_aux_loss: torch.Tensor | None = None
        self._cached_post_frontend_lens: torch.LongTensor | None = None
        self._cached_input_lengths: torch.LongTensor | None = None
        self._kept_indices: torch.LongTensor | None = None

    def _init_prior_bias(self) -> None:
        # Start the keep predictor at exactly ``prior``: zero the final-layer
        # weight, set bias to logit(prior). Bias-only would only hit the
        # target rate on average; zeroing the weight ensures every per-frame
        # logit is the bias at step 0 and sigmoid(bias) == prior exactly.
        with torch.no_grad():
            p = min(max(self.prior, 1e-4), 1 - 1e-4)
            self.boundary_mlp[-1].weight.zero_()
            self.boundary_mlp[-1].bias.fill_(math.log(p / (1.0 - p)))

    def post_parent_init(self) -> None:
        # MelConformerEncoder.post_init() recurses _init_weights and
        # overwrites the prior-targeted final-layer init. Re-apply.
        self._init_prior_bias()

    # --------------------------------------------------------- public API

    def aux_loss(self) -> torch.Tensor | None:
        return self._cached_aux_loss

    def last_stats(self) -> dict | None:
        if self._cached_lengths is None:
            return None
        aux = self._cached_aux_loss
        return {
            "aux_loss": (
                float(aux.detach().float().item()) if aux is not None else None
            ),
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

    def output_lengths(
        self, input_lengths: torch.LongTensor
    ) -> torch.LongTensor:
        cached = self._cached_lengths
        if cached is None:
            # Static fallback: the post-conv length (worst-case all kept).
            return self._post_conv_lengths(input_lengths)
        n_in = input_lengths.shape[0]
        n_cache = cached.shape[0]
        if n_in == n_cache:
            return cached
        # Beam expansion under generate(): encoder ran once on B inputs,
        # decoder calls with B*num_beams. ``repeat_interleave`` matches HF's
        # beam expansion pattern.
        if n_in > 0 and n_in % n_cache == 0:
            return cached.repeat_interleave(n_in // n_cache)
        return self._post_conv_lengths(input_lengths)

    def post_cnn_lengths(
        self, input_lengths: torch.LongTensor
    ) -> torch.LongTensor:
        return self._post_conv_lengths(input_lengths)

    def set_prior(self, prior: float) -> None:
        self.prior = float(prior)

    def set_temperature(self, temp: float) -> None:
        self.temp = float(temp)

    # ----------------------------------------------------------- internals

    def _post_conv_lengths(
        self, input_lengths: torch.LongTensor
    ) -> torch.LongTensor:
        """Per-sample valid time length after the conv stack.

        Strided layers shrink by ``(l-3)//s+1`` (kernel 3, pad 0); dilated
        layers are stride 1 with pad=(0, 1) on time, shrinking time by
        ``2*dilation`` per layer.
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

    def _apply_length_mask(
        self,
        keep: torch.Tensor,
        valid_mask: torch.Tensor,
        actual_lens: torch.Tensor,
        T_cnn: int,
    ) -> torch.Tensor:
        # Same shape as BP's: zero out padded positions, force last valid
        # frame as kept so every sample has at least one query.
        B = keep.shape[0]
        keep = keep * valid_mask
        last_valid_idx = torch.clamp(actual_lens - 1, min=0, max=T_cnn - 1)
        batch_idx = torch.arange(B, device=keep.device)
        keep = keep.clone()
        keep[batch_idx, last_valid_idx] = 1.0
        return keep

    def _learned_keeps(
        self,
        h: torch.Tensor,
        valid_mask: torch.Tensor,
        actual_lens: torch.Tensor,
        T_cnn: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(hard, soft)``, both shape ``(B, T_cnn)`` fp32.

        ``hard`` is the binary keep mask used by the gather. ``soft`` is the
        differentiable signal — RelaxedBernoulli sample (training) or
        sigmoid prob (eval) — used to build the STE gate downstream and to
        compute the binomial NLL aux loss.

        Training: ``hard = (RelaxedBernoulli.rsample() > 0.5)``. The sample
        adds exploration noise and the realized rate matches ``prior`` on
        average, with variance — the binomial NLL pulls the scorer toward
        producing well-calibrated probabilities over time.

        Eval: ``hard = top-K`` by probability per sample, with
        ``K = round(actual_len * prior)``. Deterministic and rate-stable
        regardless of scorer calibration. Threshold-at-0.5 (BP's eval path)
        breaks at cold start: with prior=0.25, the initialized prob is
        0.25 everywhere ⇒ 0 kept, only the forced-last-valid frame
        survives. BP papers over this with mean-pool-of-everything; we
        actually lose audio, so use top-K for a stable eval count.
        """
        # bf16 hazard: RelaxedBernoulli rejects half-precision probs. Force
        # logits to fp32 before sigmoid + sampling.
        logits = (
            self.boundary_mlp(self.mlp_dropout(self.pre_mlp_norm(h)))
            .squeeze(-1)
            .float()
        )
        probs = torch.sigmoid(logits)
        if self.training:
            dist = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=self.temp,
                probs=probs,
            )
            soft = dist.rsample()
            hard = (soft > 0.5).float()
        else:
            soft = probs
            # Per-sample top-K. K = round(actual_len * prior), clamped to
            # [1, actual_len] so empty / all-frame edge cases stay sane.
            k_per_sample = torch.clamp(
                torch.round(actual_lens.float() * self.prior).long(),
                min=1,
            )
            k_per_sample = torch.minimum(k_per_sample, actual_lens)
            # Push padded positions to the bottom of the sort by giving them
            # a sentinel below any sigmoid output (sigmoid ∈ (0, 1)).
            scored = probs * valid_mask + (-1.0) * (1.0 - valid_mask)
            _, sorted_idx = scored.sort(dim=1, descending=True)
            ranks = torch.arange(T_cnn, device=probs.device).unsqueeze(0)
            keep_by_rank = (ranks < k_per_sample.unsqueeze(1)).float()
            hard = torch.zeros_like(probs)
            hard.scatter_(1, sorted_idx, keep_by_rank)
        soft = self._apply_length_mask(soft, valid_mask, actual_lens, T_cnn)
        hard = self._apply_length_mask(hard, valid_mask, actual_lens, T_cnn)
        return hard, soft

    def _forced_keeps(
        self,
        T_cnn: int,
        every_n: int,
        valid_mask: torch.Tensor,
        actual_lens: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if every_n == 1:
            k = torch.ones(batch_size, T_cnn, device=device)
        else:
            k = torch.zeros(batch_size, T_cnn, device=device)
            k[:, every_n - 1::every_n] = 1.0
        return self._apply_length_mask(k, valid_mask, actual_lens, T_cnn)

    def _binomial_loss(
        self,
        hard: torch.Tensor,
        actual_lens: torch.Tensor,
    ) -> torch.Tensor:
        binomial = torch.distributions.binomial.Binomial(
            total_count=actual_lens.float(),
            probs=torch.tensor([self.prior], device=hard.device),
        )
        n_kept = hard.sum(dim=1)
        per_sample = -binomial.log_prob(n_kept) / actual_lens.float().clamp(min=1)
        return per_sample.mean()

    # ------------------------------------------------------------- forward

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # 1. Full Conv2d stack: (B, T_mel, n_mels) -> (B, hidden, T_cnn, mel_after).
        x = x.unsqueeze(1)
        x = self.convs(x)
        bsz, hidden_dim, T_cnn, m_out = x.shape

        # 2. Flatten + project AFTER the whole stack — same as cross_attn.
        h = x.permute(0, 2, 1, 3).reshape(bsz, T_cnn, hidden_dim * m_out)
        h = self.proj(h)
        h = self.proj_dropout(h)

        # 3. Stash post-CNN features for the encoder's interleaved XA blocks
        # to read as K/V — same contract as CrossAttnDownsampler.
        self._post_cnn_features = h
        device = h.device

        # 4. Per-sample valid lengths after the conv stack.
        if input_lengths is None:
            actual_lens = torch.full(
                (bsz,), T_cnn, dtype=torch.long, device=device
            )
        else:
            actual_lens = self._post_conv_lengths(input_lengths.to(device)).long()
            actual_lens = torch.clamp(actual_lens, min=1, max=T_cnn)
        pos = torch.arange(T_cnn, device=device).unsqueeze(0)
        valid_mask = (pos < actual_lens.unsqueeze(1)).float()

        # 5. Predict / force per-frame keep mask.
        if self.boundary_mode == "all":
            hard = self._forced_keeps(T_cnn, 1, valid_mask, actual_lens, bsz, device)
            soft = hard.detach()
        elif self.boundary_mode == "alternating":
            hard = self._forced_keeps(T_cnn, 2, valid_mask, actual_lens, bsz, device)
            soft = hard.detach()
        else:
            hard, soft = self._learned_keeps(h, valid_mask, actual_lens, T_cnn)
        # Straight-through gate. Forward = hard (0/1); backward routes the
        # soft sample's gradient through the scorer. Stays in fp32 here;
        # cast at the gather-multiply site.
        gate = hard - soft.detach() + soft  # (B, T_cnn) fp32

        # 6. Sort-based gather to get the kept frame indices.
        # ``hard.sort(descending=True, stable=True)`` puts 1s first in
        # original-index order; the top max_keep entries are the kept
        # indices, padded with non-kept indices for samples with fewer
        # kept tokens than the batch maximum.
        kept_counts = hard.sum(dim=1).long()  # (B,)
        max_keep = (
            int(kept_counts.max().item()) if kept_counts.numel() > 0 else 0
        )
        if max_keep == 0:
            # Defensive: ``_apply_length_mask`` forces the last valid frame
            # as kept, so this branch shouldn't fire on a non-empty batch.
            self._kept_indices = torch.zeros(
                (bsz, 0), dtype=torch.long, device=device
            )
            self._cached_lengths = kept_counts.clamp(min=1)
            self._cached_post_frontend_lens = actual_lens.detach().long()
            self._cached_input_lengths = (
                input_lengths.detach().long() if input_lengths is not None else None
            )
            self._cached_aux_loss = None
            return torch.zeros(
                (bsz, 0, self.hidden), dtype=h.dtype, device=device
            )
        _, sorted_idx = hard.sort(dim=1, descending=True, stable=True)
        kept_idx = sorted_idx[:, :max_keep]  # (B, max_keep) long

        # 7. Gather features at kept indices, multiply by gate to wire the
        # STE gradient back through the scorer. Real kept positions:
        # gate=1 in forward (passthrough), soft gradient flows back. Padding
        # positions (for samples with fewer than max_keep kept): gate=0 in
        # forward (zeros the query), encoder masks via ``output_lengths``.
        kept_idx_expanded = kept_idx.unsqueeze(-1).expand(-1, -1, self.hidden)
        queries = torch.gather(h, dim=1, index=kept_idx_expanded)
        gate_at_kept = gate.gather(1, kept_idx)  # (B, max_keep) fp32
        queries = queries * gate_at_kept.unsqueeze(-1).to(queries.dtype)

        # 8. Stash per-sample picked indices for the encoder's interleaved
        # XA blocks to use as Q's RoPE positions. Read by
        # ``MelConformerEncoder._encoder_with_xa``.
        self._kept_indices = kept_idx

        # 9. Cache lengths + (training-only) aux loss.
        self._cached_lengths = kept_counts.clamp(min=1)
        self._cached_post_frontend_lens = actual_lens.detach().long()
        self._cached_input_lengths = (
            input_lengths.detach().long() if input_lengths is not None else None
        )
        if self.training and self.boundary_mode == "learned":
            self._cached_aux_loss = self.loss_weight * self._binomial_loss(
                hard, actual_lens
            )
        else:
            self._cached_aux_loss = None

        return queries
