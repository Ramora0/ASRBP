"""Pre-stem feature preprocessing for ``MelConformerEncoder``.

Two modules that operate on raw log-Mel features ``(B, T_mel, n_mels)``
before the downsampler stem:

- ``InputNormalization`` — per-bin running mean/var, in the style of
  SpeechBrain's ``InputNormalization(norm_type='global',
  update_until_epoch=N)``. Stats update during training until the
  ``frozen`` flag is flipped by ``FreezeInputNormCallback`` (see
  ``scripts/train.py``), then stay fixed for the rest of training and
  for all downstream eval. The running buffers ride along in the model
  state_dict, so ``evaluate.py`` inherits them via ``load_state_dict``.

- ``SpecAugment`` — deterministic ``K`` time masks of length uniform in
  ``[tl, th]`` + ``K`` feature masks of length uniform in ``[fl, fh]``,
  applied to the **pre-stem** 100 Hz log-Mel features (matches
  Park et al. 2019 and the SB LibriSpeech recipe). Masked regions are
  zero-filled — intended to run **after** ``InputNormalization``, so
  zero = per-bin mean.

Placing both operations pre-stem matches SB's ``fea_augment`` pipeline
(applied before the CNN frontend); the Wav2Vec2Conformer stock masking
hook (``_mask_hidden_states``) was post-stem over hidden channels,
which regularizes latent feature dimensions rather than the mel axis
that SpecAugment is designed for.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InputNormalization(nn.Module):
    """Per-feature running mean/var normalization with an external freeze flag.

    Like ``BatchNorm1d(affine=False, track_running_stats=True)`` but with
    pooled-variance accumulation (stable under arbitrary batch sizes across
    training) and with the update decision decoupled from ``.train()/.eval()``
    — you toggle ``self.frozen`` so the encoder can freeze stats mid-training
    while still computing gradients through the normalization.

    Padded frames are excluded from the batch moments via ``attention_mask``.
    Under DDP ``broadcast_buffers=True`` means rank 0's buffers are the
    authoritative copy at the start of each forward, so effectively only
    rank 0's updates persist — an unbiased sample since the shards are drawn
    from the same distribution.
    """

    def __init__(self, n_features: int, eps: float = 1e-5):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(n_features))
        self.register_buffer("running_var", torch.ones(n_features))
        self.register_buffer("n_seen", torch.zeros((), dtype=torch.long))
        self.eps = eps
        self.frozen = False

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor, attention_mask: torch.Tensor | None) -> None:
        # x: (B, T, F). Drop padded frames before computing batch moments;
        # otherwise zero-padded tails bias the running mean toward zero.
        if attention_mask is None:
            flat = x.reshape(-1, x.size(-1))
        else:
            flat = x[attention_mask.bool()]
        n_new = int(flat.size(0))
        if n_new == 0:
            return
        # Compute in fp32 — running stats drift badly under bf16/fp16.
        flat = flat.float()
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        n_old = int(self.n_seen.item())
        total = n_old + n_new
        delta = batch_mean - self.running_mean
        new_mean = self.running_mean + delta * (n_new / total)
        # Chan et al. pooled variance (two-pass merge):
        # σ² = (n_a σ²_a + n_b σ²_b) / n + (n_a n_b / n²) (μ_a - μ_b)²
        new_var = (
            (n_old / total) * self.running_var
            + (n_new / total) * batch_var
            + (n_old * n_new / (total * total)) * delta.pow(2)
        )
        self.running_mean.copy_(new_mean)
        self.running_var.copy_(new_var)
        self.n_seen.fill_(total)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.training and not self.frozen:
            self._update_stats(x.detach(), attention_mask)
        mean = self.running_mean.to(x.dtype)
        std = (self.running_var + self.eps).sqrt().to(x.dtype)
        return (x - mean) / std


class SpecAugment(nn.Module):
    """Deterministic K time masks + K feature masks on pre-stem log-Mel.

    For each sample in ``(B, T, F)``:
      - draws ``time_masks`` time-axis mask lengths uniform in
        ``[time_length_low, time_length_high]``, and places each at a
        uniform-random valid start;
      - does the same on the feature (mel-bin) axis with
        ``[feature_length_low, feature_length_high]``.

    Fills the union of all masked positions with zero. Intended to run
    after ``InputNormalization``, so zero corresponds to the per-bin
    mean. No-op outside training mode, or when ``self.active`` is
    ``False`` (``SpecAugWarmupCallback`` flips this on after
    ``spec_aug_warmup_steps`` global steps).
    """

    def __init__(
        self,
        time_masks: int,
        time_length_low: int,
        time_length_high: int,
        feature_masks: int,
        feature_length_low: int,
        feature_length_high: int,
    ):
        super().__init__()
        self.time_masks = int(time_masks)
        self.time_length_low = int(time_length_low)
        self.time_length_high = int(time_length_high)
        self.feature_masks = int(feature_masks)
        self.feature_length_low = int(feature_length_low)
        self.feature_length_high = int(feature_length_high)
        self.active = True

    @staticmethod
    def _build_axis_mask(
        batch_size: int,
        axis_size: int,
        n_masks: int,
        length_low: int,
        length_high: int,
        valid_lengths: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a ``(B, axis_size)`` bool mask — True where the axis should be zeroed."""
        mask = torch.zeros((batch_size, axis_size), dtype=torch.bool, device=device)
        if n_masks <= 0 or length_high <= 0:
            return mask
        # randint needs high > low; guard against degenerate configs (low == high)
        # by clamping the upper bound to low + 1 (torch.randint's high is exclusive).
        hi = max(length_low + 1, length_high + 1)
        lo = max(1, length_low)
        arange = torch.arange(axis_size, device=device)
        for _ in range(n_masks):
            lengths = torch.randint(low=lo, high=hi, size=(batch_size,), device=device)
            if valid_lengths is not None:
                # Time axis: sample start ∈ [0, valid_len - length) so masks stay on
                # non-padded frames. Feature axis passes ``valid_lengths=None``.
                limits = valid_lengths.to(device) - lengths
            else:
                limits = torch.full((batch_size,), axis_size, device=device) - lengths
            limits = torch.clamp(limits, min=1)
            starts = (torch.rand((batch_size,), device=device) * limits.float()).long()
            lower = starts.unsqueeze(1)
            upper = (starts + lengths).unsqueeze(1)
            mask = mask | ((arange.unsqueeze(0) >= lower) & (arange.unsqueeze(0) < upper))
        return mask

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self.training or not self.active:
            return x
        B, T, F = x.shape
        device = x.device
        valid_lengths = attention_mask.sum(-1) if attention_mask is not None else None

        time_mask = self._build_axis_mask(
            batch_size=B,
            axis_size=T,
            n_masks=self.time_masks,
            length_low=self.time_length_low,
            length_high=self.time_length_high,
            valid_lengths=valid_lengths,
            device=device,
        )
        feat_mask = self._build_axis_mask(
            batch_size=B,
            axis_size=F,
            n_masks=self.feature_masks,
            length_low=self.feature_length_low,
            length_high=self.feature_length_high,
            valid_lengths=None,
            device=device,
        )
        # Union: zero-fill positions hit by either time OR feature masks.
        # time_mask: (B, T) -> (B, T, 1); feat_mask: (B, F) -> (B, 1, F).
        combined = time_mask.unsqueeze(-1) | feat_mask.unsqueeze(1)
        return x.masked_fill(combined, 0.0)


__all__ = ["InputNormalization", "SpecAugment"]
