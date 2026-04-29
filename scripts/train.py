"""Train the Conformer AED model on LibriSpeech with ``Seq2SeqTrainer``."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _is_main_process() -> bool:
    """True on rank 0 under torchrun / single-GPU runs.

    ``RANK`` is set by ``torchrun`` / SLURM-launched DDP. Unset on single-GPU
    runs (which are always rank 0). We can't use ``torch.distributed`` here
    because the process group isn't initialized yet — Trainer does that later
    in ``TrainingArguments._setup_devices``.
    """
    return int(os.environ.get("RANK", "0")) == 0


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

# --- HF cache bootstrap: MUST run before any ``datasets`` / ``transformers`` / ---
# --- ``conformer_asr.*`` import, otherwise HF snapshots the wrong cache paths. ---
from bootstrap_cache import bootstrap_cache_from_argv  # noqa: E402

_resolved_cache = bootstrap_cache_from_argv()
print(f"HF cache_dir (bootstrapped): {_resolved_cache}")
# -------------------------------------------------------------------------------

from transformers import (  # noqa: E402
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
from transformers.trainer_callback import ProgressCallback  # noqa: E402

from conformer_asr.config import (  # noqa: E402
    autocast_dtype,
    load_config,
    resolve_grad_accum,
    resolve_precision,
)
from conformer_asr.data import (  # noqa: E402
    DataCollatorSpeechSeq2SeqWithPadding,
    RandomSpeedVariantSampler,
    _preprocess_cache_dir,
    _preprocess_cache_key,
    load_librispeech,
    preprocess_dataset,
    setup_cache_dir,
)
from conformer_asr.metrics import build_compute_metrics  # noqa: E402
from conformer_asr.model import build_model  # noqa: E402
from conformer_asr.tokenizer import load_tokenizer  # noqa: E402
from conformer_asr.wandb_utils import (  # noqa: E402
    CTCEvalCallback,
    EpochLoggerCallback,
    PredictionsTableCallback,
    SWACallback,
    init_wandb,
    wandb_is_enabled,
)


class StepTimerCallback(TrainerCallback):
    """Per-step latency breakdown: total wall time vs. compute vs. data-wait.

    On a healthy GPU-bound run, ``data_wait_ms`` percentiles should sit near 0
    — workers stay ahead of the GPU. When the percentiles climb (especially
    p95/p99) the dataloader is starving the GPU; common culprits are network
    storage latency, too-small ``prefetch_factor``, or worker oversubscription.

    Implementation: ``on_step_begin`` and ``on_step_end`` bracket the compute
    portion of each micro-step. The gap between ``on_step_end`` of step N and
    ``on_step_begin`` of step N+1 is data-wait + Python overhead. We log
    p50 / p95 / p99 of each over the last ``logging_steps`` boundary, both
    via ``tqdm.write`` and to ``wandb`` if the run is online.
    """

    def __init__(self) -> None:
        import time

        self._t = time.perf_counter
        self._last_begin: float | None = None
        self._last_end: float | None = None
        self._compute_ms: list[float] = []
        self._wait_ms: list[float] = []
        self._total_ms: list[float] = []
        # Mirror trainer's logging cadence; updated on first on_step_begin.
        self._log_every: int = 50

    @staticmethod
    def _pct(xs: list[float], q: float) -> float:
        if not xs:
            return 0.0
        s = sorted(xs)
        idx = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
        return s[idx]

    def on_step_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            now = self._t()
            if self._last_end is not None:
                self._wait_ms.append((now - self._last_end) * 1000.0)
            if self._last_begin is not None:
                self._total_ms.append((now - self._last_begin) * 1000.0)
            self._last_begin = now
            self._log_every = max(10, int(getattr(args, "logging_steps", 50) or 50))

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        now = self._t()
        if self._last_begin is not None:
            self._compute_ms.append((now - self._last_begin) * 1000.0)
        self._last_end = now
        if len(self._compute_ms) >= self._log_every:
            p50_c = self._pct(self._compute_ms, 0.50)
            p95_c = self._pct(self._compute_ms, 0.95)
            p99_c = self._pct(self._compute_ms, 0.99)
            p50_w = self._pct(self._wait_ms, 0.50)
            p95_w = self._pct(self._wait_ms, 0.95)
            p99_w = self._pct(self._wait_ms, 0.99)
            p50_t = self._pct(self._total_ms, 0.50)
            p95_t = self._pct(self._total_ms, 0.95)
            line = (
                f"[step-timer N={len(self._compute_ms)}] "
                f"compute_ms p50={p50_c:.1f}/p95={p95_c:.1f}/p99={p99_c:.1f}  "
                f"wait_ms p50={p50_w:.1f}/p95={p95_w:.1f}/p99={p99_w:.1f}  "
                f"total_ms p50={p50_t:.1f}/p95={p95_t:.1f}"
            )
            try:
                from tqdm.auto import tqdm

                tqdm.write(line)
            except Exception:
                print(line)
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "perf/compute_ms_p50": p50_c,
                            "perf/compute_ms_p95": p95_c,
                            "perf/compute_ms_p99": p99_c,
                            "perf/wait_ms_p50": p50_w,
                            "perf/wait_ms_p95": p95_w,
                            "perf/wait_ms_p99": p99_w,
                            "perf/total_ms_p50": p50_t,
                            "perf/total_ms_p95": p95_t,
                        },
                        step=state.global_step,
                    )
            except Exception:
                pass
            self._compute_ms.clear()
            self._wait_ms.clear()
            self._total_ms.clear()


class EmptyCacheCallback(TrainerCallback):
    """Flush the CUDA caching allocator around validation.

    Training accumulates many variable-size cached blocks under ``group_by_length``.
    When eval then asks for a large contiguous KV cache for ``generate()`` +
    ``PredictionsTableCallback``, the allocator can't find a big enough block
    in the cached pool and grows the heap — so peak memory creeps up every
    epoch even though live usage is stable. ``empty_cache`` returns unused
    cached blocks to the driver; ``gc.collect`` first so any Python-referenced
    tensors are dropped before we release.
    """

    def _flush(self):
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_epoch_end(self, args, state, control, **kwargs):
        # Fires before eval (eval_strategy="epoch" runs after on_epoch_end).
        self._flush()

    def on_evaluate(self, args, state, control, **kwargs):
        # Fires after eval — reclaim the KV cache / generation workspace
        # before the next epoch's training steps re-allocate activations.
        self._flush()


class DualProgressCallback(TrainerCallback):
    """Two tqdm bars: overall training progress + current epoch.

    Replaces HF's default ``ProgressCallback`` (remove that one before adding
    this, or you get three bars). Also forwards HF's log dict through
    ``tqdm.write`` so per-logging-step loss lines don't shred the bars.
    """

    def __init__(self) -> None:
        self.overall_bar = None
        self.epoch_bar = None
        self.eval_bar = None
        self._last_global_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        from tqdm.auto import tqdm

        self.overall_bar = tqdm(
            total=state.max_steps,
            desc="train",
            position=0,
            leave=True,
            dynamic_ncols=True,
            unit="step",
        )
        self._last_global_step = state.global_step  # nonzero on resume
        self.overall_bar.update(state.global_step)

    def on_epoch_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        from tqdm.auto import tqdm

        if self.epoch_bar is not None:
            self.epoch_bar.close()
        total_epochs = max(1, int(args.num_train_epochs))
        steps_per_epoch = max(1, state.max_steps // total_epochs)
        epoch_idx = (int(state.epoch) if state.epoch is not None else 0) + 1
        self.epoch_bar = tqdm(
            total=steps_per_epoch,
            desc=f"epoch {epoch_idx}/{total_epochs}",
            position=1,
            leave=False,
            dynamic_ncols=True,
            unit="step",
        )

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        delta = state.global_step - self._last_global_step
        self._last_global_step = state.global_step
        if self.overall_bar is not None:
            self.overall_bar.update(delta)
        if self.epoch_bar is not None:
            self.epoch_bar.update(delta)

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_bar is not None:
            self.epoch_bar.close()
            self.epoch_bar = None

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        # Fires per eval batch. Replaces HF's default ProgressCallback eval bar
        # (which we removed to avoid dueling train bars).
        if not state.is_world_process_zero:
            return
        from tqdm.auto import tqdm

        if self.eval_bar is None:
            total = len(eval_dataloader) if eval_dataloader is not None and hasattr(eval_dataloader, "__len__") else None
            self.eval_bar = tqdm(
                total=total,
                desc="eval",
                position=2,
                leave=False,
                dynamic_ncols=True,
                unit="batch",
            )
        self.eval_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if self.eval_bar is not None:
            self.eval_bar.close()
            self.eval_bar = None

    def on_train_end(self, args, state, control, **kwargs):
        if self.epoch_bar is not None:
            self.epoch_bar.close()
            self.epoch_bar = None
        if self.eval_bar is not None:
            self.eval_bar.close()
            self.eval_bar = None
        if self.overall_bar is not None:
            self.overall_bar.close()
            self.overall_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.overall_bar is None:
            return
        shown = {k: v for k, v in logs.items() if k != "total_flos"}
        self.overall_bar.write(str(shown))


class EpochCheckpointRenameCallback(TrainerCallback):
    """Rename ``checkpoint-<global_step>`` → ``checkpoint-<epoch>`` after save.

    HF Trainer names checkpoints by global_step, but under ``save_strategy='epoch'``
    the epoch index is the meaningful identifier. We keep the ``checkpoint-``
    prefix and a trailing integer so ``_rotate_checkpoints`` (which globs
    ``checkpoint-*`` and sorts by the trailing number) still prunes correctly.
    Also updates ``state.best_model_checkpoint`` so ``load_best_model_at_end``
    can find the renamed directory.

    Rank-0 only: under DDP, HF fires ``on_save`` on every rank, and non-zero
    ranks exit ``_save_checkpoint`` well before rank 0 finishes its heavy writes
    (model → optimizer → scheduler → trainer_state). If a non-zero rank renames
    the directory mid-flight, rank 0's ``torch.save(scheduler.state_dict(), …)``
    lands on a vanished path and the run crashes. Keeping the rename on rank 0
    serializes it after rank 0's own ``_save_checkpoint`` returns.
    """

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        from pathlib import Path as _Path

        step_dir = _Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not step_dir.exists():
            return
        epoch = int(round(state.epoch or 0))
        epoch_dir = _Path(args.output_dir) / f"checkpoint-{epoch}"
        # Avoid clobbering an existing epoch dir from a resumed run that
        # re-saves the same epoch boundary.
        if epoch_dir.exists():
            return
        step_dir.rename(epoch_dir)
        if state.best_model_checkpoint and _Path(state.best_model_checkpoint) == step_dir:
            state.best_model_checkpoint = str(epoch_dir)


class OneShotEvalCallback(TrainerCallback):
    """Fire eval exactly once at a given ``global_step``.

    Useful for sanity-checking the full eval pipeline (generate → gather →
    metrics → WER) well before the first real epoch-boundary eval would fire.
    Sets ``control.should_evaluate = True`` on the target step, then stays
    dormant so the normal epoch-boundary eval cadence is unaffected.
    """

    def __init__(self, target_step: int) -> None:
        self.target_step = int(target_step)
        self._fired = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self._fired and state.global_step >= self.target_step:
            control.should_evaluate = True
            self._fired = True
        return control


def _unwrap_encoder(model):
    """Peel DDP / SpeechEncoderDecoderModel wrappers down to ``MelConformerEncoder``."""
    m = model
    while hasattr(m, "module"):
        m = m.module
    return getattr(m, "encoder", None)


class FreezeInputNormCallback(TrainerCallback):
    """Freeze ``InputNormalization`` running stats at a configured epoch.

    Mirrors SB's ``InputNormalization(update_until_epoch=N)``: stats
    accumulate for the first N epochs of training (during which the encoder
    is also learning a crude alignment), then are frozen for the rest of
    training and all downstream eval. The frozen running_mean / running_var
    buffers ride along in the model state_dict.
    """

    def __init__(self, freeze_after_epochs: int) -> None:
        self.freeze_after = int(freeze_after_epochs)
        self._done = False

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self._done or model is None:
            return
        if (state.epoch or 0.0) < self.freeze_after:
            return
        enc = _unwrap_encoder(model)
        norm = getattr(enc, "input_norm", None) if enc is not None else None
        if norm is not None and not norm.frozen:
            norm.frozen = True
            self._done = True
            n_seen = int(norm.n_seen.item())
            print(f"[input-norm] frozen at epoch {state.epoch:.2f} (n_seen={n_seen} frames)")


class SpecAugWarmupCallback(TrainerCallback):
    """Gate ``SpecAugment`` off until ``warmup_steps`` global steps have passed.

    ``SpecAugment.active`` defaults to True; this callback flips it off at the
    start of training and back on once ``state.global_step`` crosses the
    threshold. ``warmup_steps <= 0`` skips the gating entirely (no-op).
    """

    def __init__(self, warmup_steps: int) -> None:
        self.warmup = int(warmup_steps)
        self._activated = False

    def _set_active(self, model, active: bool) -> None:
        enc = _unwrap_encoder(model)
        aug = getattr(enc, "spec_augment", None) if enc is not None else None
        if aug is not None:
            aug.active = active

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.warmup <= 0 or model is None:
            return
        if state.global_step < self.warmup:
            self._set_active(model, False)
        else:
            self._set_active(model, True)
            self._activated = True

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self._activated or self.warmup <= 0 or model is None:
            return
        if state.global_step >= self.warmup:
            self._set_active(model, True)
            self._activated = True
            print(f"[spec-augment] activated at step {state.global_step}")


class SpeedAugSeq2SeqTrainer(Seq2SeqTrainer):
    """``Seq2SeqTrainer`` variant that subsamples the speed-perturbed cache.

    When ``n_speed_variants > 1`` the preprocessed train split contains N
    contiguous variant rows per clip (row ``k * n + v`` = clip ``k`` at speed
    slot ``v``). This trainer wraps the normal train sampler in
    ``RandomSpeedVariantSampler`` so each epoch keeps exactly one variant per
    clip (uniform random), restoring epoch length to ``N`` instead of ``n * N``.
    ``n_speed_variants == 1`` is a pass-through: no wrapping, stock sampler.
    """

    def __init__(self, *args, n_speed_variants: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self._n_speed_variants = int(n_speed_variants)

    def _get_train_sampler(self, *args, **kwargs):
        # Seq2SeqTrainer._get_train_sampler signature drifts across transformers
        # 4.x point releases (some take ``train_dataset``, some don't) — forward
        # whatever we're given straight through so the wrapper stays compatible.
        base = super()._get_train_sampler(*args, **kwargs)
        if self._n_speed_variants > 1 and base is not None:
            return RandomSpeedVariantSampler(base, self._n_speed_variants)
        return base


class HybridSeq2SeqTrainer(SpeedAugSeq2SeqTrainer):
    """``Seq2SeqTrainer`` that keeps labels in the forward pass so the model
    can compute both AED and CTC losses in one shot.

    The stock ``Trainer.compute_loss`` pops ``labels`` from the batch before
    calling the model whenever a ``label_smoother`` is active — that would
    starve the CTC branch of its targets. We override to:
      1. Leave labels in ``inputs`` so ``ConformerAEDWithCTC`` can compute the
         raw CTC loss internally.
      2. Re-apply label smoothing to *just* the AED branch on the way out
         (the model's own AED loss is unsmoothed) and re-blend with the raw
         CTC loss using the model's ``ctc_weight``.

    A model without a CTC head falls back to the normal AED-only path so this
    trainer is safe to use regardless of ``ctc_enabled``.

    Also tracks per-step AED / CTC loss sums so they can be surfaced as
    ``train/aed_loss`` and ``train/ctc_loss`` on the next ``log()`` boundary,
    mirroring how HF's built-in ``tr_loss`` accumulator drives ``train/loss``.

    When the encoder uses a boundary-predictor downsampler (or any downsampler
    that exposes ``last_stats()``), the trainer also accumulates BP
    diagnostics across the logging window and emits them under a ``bp/``
    section (so they group into their own panel in wandb). Train-time keys:
    ``bp/aux_loss``, ``bp/realized_prior``, ``bp/compression`` (post-
    frontend), ``bp/total_compression`` (mel → post-BP), plus the per-frame
    distribution diagnostics (``bp/bias``, ``bp/logit_mean``, ``bp/logit_std``,
    ``bp/prob_mean``, ``bp/frac_sat_low``, ``bp/frac_sat_high``) and gradient
    norms captured after backward (``bp/bias_grad``, ``bp/mlp_last_w_grad``,
    ``bp/mlp_first_w_grad``, ``bp/pre_norm_grad``, ``bp/post_norm_grad``).
    Static downsamplers have no ``last_stats()`` so these keys simply don't
    appear.

    The same accumulators run during evaluation (driven by ``prediction_step``
    rather than ``compute_loss``); empirical compression at eval time is
    surfaced under the same section as ``bp/eval_realized_prior`` /
    ``bp/eval_compression`` / ``bp/eval_total_compression``. Eval-mode
    forwards in ``BoundaryPredictorDownsampler`` skip the binomial loss, so
    no ``bp/eval_aux_loss`` companion key is emitted.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aed_loss_sum = 0.0
        self._ctc_loss_sum = 0.0
        self._loss_component_count = 0
        # CTC viability: # samples where ``input_len >= label_len + n_dup``,
        # over the same logging window as ``aed/ctc_loss``. At low downsample
        # rates this stays at 1.0; at high rates it drops as long-target /
        # short-input clips fall under the CTC alignment floor.
        self._ctc_viable_count = 0
        self._ctc_viable_total = 0
        self._bp_aux_loss_sum = 0.0
        self._bp_n_boundaries = 0
        self._bp_n_post_frontend = 0
        self._bp_n_input = 0
        self._bp_step_count = 0
        # Per-frame distribution diagnostics (learned-mode only). Surfaced
        # as ``bp/bias``, ``bp/logit_mean``, etc. on log boundary.
        self._bp_diag_step_count = 0
        self._bp_bias_sum = 0.0
        self._bp_logit_mean_sum = 0.0
        self._bp_logit_std_sum = 0.0
        self._bp_prob_mean_sum = 0.0
        self._bp_frac_sat_low_sum = 0.0
        self._bp_frac_sat_high_sum = 0.0
        # Gradient-norm diagnostics: captured AFTER super().training_step has
        # run backward, so all .grad fields are populated. Lets us see whether
        # the binomial loss is actually reaching the BP parameters.
        self._bp_grad_step_count = 0
        self._bp_bias_grad_sum = 0.0
        self._bp_mlp_first_grad_sum = 0.0
        self._bp_mlp_last_w_grad_sum = 0.0
        self._bp_pre_norm_grad_sum = 0.0
        self._bp_post_norm_grad_sum = 0.0
        # Eval-time BP accumulators. Reset on every ``evaluate()`` call so the
        # numbers come out per-eval rather than rolling across the whole run.
        self._eval_bp_n_boundaries = 0
        self._eval_bp_n_post_frontend = 0
        self._eval_bp_n_input = 0
        # XA (cross-attn downsampler) per-layer residual-magnitude accumulators.
        # Lazy-initialized on first stats hit since layer count is config-dependent.
        # Reference accumulators track the conformer layer's own residual delta at
        # the same depth — emitted as ``train/xa_relative_strength_l{i}`` so the
        # per-tap signal is self-calibrated against an in-stack baseline.
        self._xa_delta_rms_sum: list[float] = []
        self._xa_q_rms_sum: list[float] = []
        self._xa_ref_delta_rms_sum: list[float] = []
        # Per-tap reference counter — independent of ``_xa_step_count`` because
        # the conformer layer right before a tap can be layer-dropped
        # (10% / layer / step), in which case its delta isn't recorded.
        self._xa_ref_step_count: list[int] = []
        self._xa_step_count = 0

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)

        ctc_loss = getattr(outputs, "ctc_loss", None)
        aed_raw_loss = getattr(outputs, "aed_loss", None)
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        ctc_weight = float(getattr(unwrapped, "ctc_weight", 0.0))

        if self.label_smoother is not None and labels is not None:
            # Label smoother operates on ``outputs.logits`` (AED decoder logits)
            # — it doesn't touch the CTC branch, which stays at raw CE.
            aed_loss = self.label_smoother(outputs, labels, shift_labels=False)
        else:
            # No smoothing: reuse the model's own raw AED loss if available,
            # otherwise fall back to ``outputs.loss`` (AED-only models).
            aed_loss = aed_raw_loss if aed_raw_loss is not None else outputs.loss

        if ctc_loss is not None and ctc_weight > 0.0:
            loss = (1.0 - ctc_weight) * aed_loss + ctc_weight * ctc_loss
        else:
            loss = aed_loss

        # Accumulate component sums for the next ``log()`` emission. Detaching
        # first so we don't retain autograd graphs across steps.
        if aed_loss is not None:
            self._aed_loss_sum += float(aed_loss.detach().float().item())
        if ctc_loss is not None:
            self._ctc_loss_sum += float(ctc_loss.detach().float().item())
        self._loss_component_count += 1

        # CTC viability rollup. ``ctc_viable`` is a (B,) bool tensor stamped
        # by ``ConformerAEDWithCTC._compute_ctc_loss``; sum / total gives the
        # fraction of samples in this window where CTC's input-length floor
        # is met. Skipped silently for AED-only models (no CTC head).
        ctc_viable = getattr(outputs, "ctc_viable", None)
        if ctc_viable is not None:
            self._ctc_viable_count += int(ctc_viable.sum().item())
            self._ctc_viable_total += int(ctc_viable.numel())

        # Auxiliary stats. The downsampler exposes ``last_stats()`` for BP-
        # style boundary metrics; the encoder exposes it for the per-tap
        # cross-attn delta RMS used to track interleaved-XA learning. Each
        # owns its own key namespace, so we just merge whatever's there and
        # branch on key presence below.
        encoder = getattr(unwrapped, "encoder", None)
        for obj in (getattr(encoder, "downsampler", None), encoder):
            stats_fn = getattr(obj, "last_stats", None) if obj is not None else None
            stats = stats_fn() if stats_fn is not None else None
            if stats is None:
                continue
            if "n_boundaries" in stats:
                if stats.get("aux_loss") is not None:
                    self._bp_aux_loss_sum += stats["aux_loss"]
                self._bp_n_boundaries += stats["n_boundaries"]
                self._bp_n_post_frontend += stats["n_post_frontend"]
                self._bp_n_input += stats["n_input"]
                self._bp_step_count += 1
                # Per-frame distribution diagnostics (learned-mode only).
                if "bp_bias" in stats:
                    self._bp_bias_sum += stats["bp_bias"]
                    self._bp_logit_mean_sum += stats["bp_logit_mean"]
                    self._bp_logit_std_sum += stats["bp_logit_std"]
                    self._bp_prob_mean_sum += stats["bp_prob_mean"]
                    self._bp_frac_sat_low_sum += stats["bp_frac_sat_low"]
                    self._bp_frac_sat_high_sum += stats["bp_frac_sat_high"]
                    self._bp_diag_step_count += 1
            if "xa_delta_rms" in stats:
                deltas = stats["xa_delta_rms"]
                q_rms = stats["xa_q_rms"]
                ref_d = stats.get("xa_ref_layer_delta_rms") or [None] * len(deltas)
                if not self._xa_delta_rms_sum:
                    self._xa_delta_rms_sum = [0.0] * len(deltas)
                    self._xa_q_rms_sum = [0.0] * len(deltas)
                    self._xa_ref_delta_rms_sum = [0.0] * len(deltas)
                    self._xa_ref_step_count = [0] * len(deltas)
                for i, (d, q, rd) in enumerate(zip(deltas, q_rms, ref_d)):
                    self._xa_delta_rms_sum[i] += float(d)
                    self._xa_q_rms_sum[i] += float(q)
                    if rd is not None:
                        self._xa_ref_delta_rms_sum[i] += float(rd)
                        self._xa_ref_step_count[i] += 1
                self._xa_step_count += 1

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        # Run forward+backward via the parent. Gradients are populated on
        # all params after this call, but optimizer.step()/zero_grad() are
        # still in the future (the outer trainer loop owns those). That's
        # the right window to peek at the BP's grad fields.
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self._capture_bp_grad_norms(model)
        return loss

    def _capture_bp_grad_norms(self, model) -> None:
        """Read .grad off select BP parameters and add their norms to the
        rolling sum. No-op for non-BP downsamplers and for the steps where
        gradient accumulation hasn't yet attached a grad.
        """
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        downsampler = getattr(getattr(unwrapped, "encoder", None), "downsampler", None)
        if downsampler is None or not hasattr(downsampler, "boundary_mlp"):
            return
        bias = downsampler.boundary_mlp[-1].bias
        last_w = downsampler.boundary_mlp[-1].weight
        first_w = downsampler.boundary_mlp[0].weight
        pre_w = getattr(downsampler, "pre_mlp_norm", None)
        post_w = getattr(downsampler, "post_pool_norm", None)
        # Some grads can be None if the param was layer-dropped or sees no
        # gradient flow this step; skip in that case rather than crashing.
        captured_any = False
        if bias.grad is not None:
            self._bp_bias_grad_sum += float(bias.grad.detach().norm().item())
            captured_any = True
        if last_w.grad is not None:
            self._bp_mlp_last_w_grad_sum += float(last_w.grad.detach().norm().item())
            captured_any = True
        if first_w.grad is not None:
            self._bp_mlp_first_grad_sum += float(first_w.grad.detach().norm().item())
            captured_any = True
        if pre_w is not None and pre_w.weight.grad is not None:
            self._bp_pre_norm_grad_sum += float(pre_w.weight.grad.detach().norm().item())
        if post_w is not None and post_w.weight.grad is not None:
            self._bp_post_norm_grad_sum += float(post_w.weight.grad.detach().norm().item())
        if captured_any:
            self._bp_grad_step_count += 1

    def _accumulate_eval_bp_stats(self, model) -> None:
        """Read ``last_stats()`` off the model's downsampler and add to the
        eval-time accumulators. No-op for downsamplers without ``last_stats``.
        """
        unwrapped = model
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        downsampler = getattr(getattr(unwrapped, "encoder", None), "downsampler", None)
        last_stats_fn = getattr(downsampler, "last_stats", None) if downsampler is not None else None
        if last_stats_fn is None:
            return
        stats = last_stats_fn()
        if stats is None:
            return
        self._eval_bp_n_boundaries += stats["n_boundaries"]
        self._eval_bp_n_post_frontend += stats["n_post_frontend"]
        self._eval_bp_n_input += stats["n_input"]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Eval driver: ``Seq2SeqTrainer.prediction_step`` runs the encoder via
        # ``generate()`` first (then any prediction-loss forward). We tap the
        # downsampler cache after super returns — by that point the most-recent
        # encoder forward has populated ``last_stats()`` for this batch.
        result = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
        self._accumulate_eval_bp_stats(model)
        return result

    def evaluate(self, *args, **kwargs):
        # Reset eval accumulators at the *start* of every evaluate() call so the
        # numbers reflect this eval pass and don't bleed across epoch boundaries.
        self._eval_bp_n_boundaries = 0
        self._eval_bp_n_post_frontend = 0
        self._eval_bp_n_input = 0
        return super().evaluate(*args, **kwargs)

    def log(self, logs, *args, **kwargs):
        # HF fires ``log()`` for both training (``loss`` key) and eval
        # (``eval_loss`` key). Only inject / reset on training log boundaries —
        # eval-time CTC loss is reported separately by CTCEvalCallback.
        if "loss" in logs and self._loss_component_count > 0:
            denom = float(self._loss_component_count)
            logs["train/aed_loss"] = self._aed_loss_sum / denom
            logs["train/ctc_loss"] = self._ctc_loss_sum / denom
            if self._ctc_viable_total > 0:
                logs["train/ctc_viable_pct"] = (
                    self._ctc_viable_count / float(self._ctc_viable_total)
                )
            self._aed_loss_sum = 0.0
            self._ctc_loss_sum = 0.0
            self._loss_component_count = 0
            self._ctc_viable_count = 0
            self._ctc_viable_total = 0
            if self._bp_step_count > 0:
                logs["bp/aux_loss"] = self._bp_aux_loss_sum / float(self._bp_step_count)
                if self._bp_n_post_frontend > 0:
                    logs["bp/realized_prior"] = (
                        self._bp_n_boundaries / float(self._bp_n_post_frontend)
                    )
                    logs["bp/compression"] = (
                        self._bp_n_post_frontend / max(1, self._bp_n_boundaries)
                    )
                if self._bp_n_input > 0:
                    logs["bp/total_compression"] = (
                        self._bp_n_input / max(1, self._bp_n_boundaries)
                    )
                self._bp_aux_loss_sum = 0.0
                self._bp_n_boundaries = 0
                self._bp_n_post_frontend = 0
                self._bp_n_input = 0
                self._bp_step_count = 0
            if self._bp_diag_step_count > 0:
                d = float(self._bp_diag_step_count)
                logs["bp/bias"] = self._bp_bias_sum / d
                logs["bp/logit_mean"] = self._bp_logit_mean_sum / d
                logs["bp/logit_std"] = self._bp_logit_std_sum / d
                logs["bp/prob_mean"] = self._bp_prob_mean_sum / d
                logs["bp/frac_sat_low"] = self._bp_frac_sat_low_sum / d
                logs["bp/frac_sat_high"] = self._bp_frac_sat_high_sum / d
                self._bp_diag_step_count = 0
                self._bp_bias_sum = 0.0
                self._bp_logit_mean_sum = 0.0
                self._bp_logit_std_sum = 0.0
                self._bp_prob_mean_sum = 0.0
                self._bp_frac_sat_low_sum = 0.0
                self._bp_frac_sat_high_sum = 0.0
            if self._bp_grad_step_count > 0:
                gd = float(self._bp_grad_step_count)
                logs["bp/bias_grad"] = self._bp_bias_grad_sum / gd
                logs["bp/mlp_last_w_grad"] = self._bp_mlp_last_w_grad_sum / gd
                logs["bp/mlp_first_w_grad"] = self._bp_mlp_first_grad_sum / gd
                logs["bp/pre_norm_grad"] = self._bp_pre_norm_grad_sum / gd
                logs["bp/post_norm_grad"] = self._bp_post_norm_grad_sum / gd
                self._bp_grad_step_count = 0
                self._bp_bias_grad_sum = 0.0
                self._bp_mlp_last_w_grad_sum = 0.0
                self._bp_mlp_first_grad_sum = 0.0
                self._bp_pre_norm_grad_sum = 0.0
                self._bp_post_norm_grad_sum = 0.0
            if self._xa_step_count > 0:
                denom_xa = float(self._xa_step_count)
                for i in range(len(self._xa_delta_rms_sum)):
                    d_mean = self._xa_delta_rms_sum[i] / denom_xa
                    q_mean = self._xa_q_rms_sum[i] / denom_xa
                    if q_mean > 0:
                        # Absolute ratio: XA's contribution to the residual
                        # stream entering the block.
                        logs[f"train/xa_delta_ratio_l{i}"] = d_mean / q_mean
                    # Relative strength: XA's contribution as a fraction of
                    # the conformer layer's own residual delta at the same
                    # depth. Self-calibrating — > 0.25 means XA is doing real
                    # work relative to a full conformer layer; < 0.05 means
                    # XA is small change vs the layer's own contribution.
                    n_ref = self._xa_ref_step_count[i]
                    if n_ref > 0:
                        rd_mean = self._xa_ref_delta_rms_sum[i] / float(n_ref)
                        if rd_mean > 0:
                            logs[f"train/xa_relative_strength_l{i}"] = d_mean / rd_mean
                self._xa_delta_rms_sum = [0.0] * len(self._xa_delta_rms_sum)
                self._xa_q_rms_sum = [0.0] * len(self._xa_q_rms_sum)
                self._xa_ref_delta_rms_sum = [0.0] * len(self._xa_ref_delta_rms_sum)
                self._xa_ref_step_count = [0] * len(self._xa_ref_step_count)
                self._xa_step_count = 0
        # Eval log boundary — emit empirical compression rate over the eval set.
        # The downsampler's eval-mode forward zeros out the binomial loss, so
        # there's no ``bp/eval_aux_loss`` companion key — only the realized
        # rate / compression are meaningful at eval time. Eval keys live in
        # the same ``bp/`` section as train-time keys, prefixed with ``eval_``
        # so a single panel shows train vs eval side-by-side.
        if "eval_loss" in logs and self._eval_bp_n_post_frontend > 0:
            logs["bp/eval_realized_prior"] = (
                self._eval_bp_n_boundaries / float(self._eval_bp_n_post_frontend)
            )
            logs["bp/eval_compression"] = (
                self._eval_bp_n_post_frontend / max(1, self._eval_bp_n_boundaries)
            )
            if self._eval_bp_n_input > 0:
                logs["bp/eval_total_compression"] = (
                    self._eval_bp_n_input / max(1, self._eval_bp_n_boundaries)
                )
        return super().log(logs, *args, **kwargs)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/cnns/c4x.yaml")
    p.add_argument("--subset", choices=["clean100", "clean460", "all960"])
    p.add_argument("--output_dir")
    p.add_argument("--num_train_epochs", type=float)
    p.add_argument("--per_device_train_batch_size", type=int)
    p.add_argument("--per_device_eval_batch_size", type=int)
    p.add_argument("--gradient_accumulation_steps", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--warmup_steps", type=int)
    p.add_argument("--tokenizer_dir")
    p.add_argument("--cache_dir", help="overrides data.cache_dir")
    p.add_argument(
        "--speed_perturbations",
        type=float,
        nargs="+",
        help="Kaldi-style train-time speed factors, e.g. --speed_perturbations 0.9 1.0 1.1. "
        "Pass a single 1.0 to disable.",
    )
    p.add_argument("--report_to", help="e.g. 'wandb', 'tensorboard', or 'wandb,tensorboard'")
    # wandb overrides
    p.add_argument("--wandb_project", dest="project")
    p.add_argument("--wandb_entity", dest="entity")
    p.add_argument("--wandb_run_name", dest="run_name")
    p.add_argument("--wandb_group", dest="group")
    p.add_argument("--wandb_tags", dest="tags", help="comma-separated")
    p.add_argument("--wandb_notes", dest="notes")
    p.add_argument("--no_wandb", action="store_true", help="Disable wandb regardless of config")
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument(
        "--early_eval_frac",
        type=float,
        default=None,
        help="Fraction of epoch 1 at which to fire a one-shot sanity-check eval (e.g. 0.05).",
    )
    return p.parse_args()


def _flatten_overrides(args: argparse.Namespace) -> dict:
    overrides = {
        k: v
        for k, v in vars(args).items()
        if k not in {"config", "resume_from_checkpoint", "no_wandb", "early_eval_frac"}
    }
    # `tags` comes in as a comma-separated string
    tags = overrides.get("tags")
    if isinstance(tags, str):
        overrides["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    return overrides


def main() -> None:
    args = parse_args()
    overrides = _flatten_overrides(args)
    cfg = load_config(args.config, overrides=overrides)

    if args.no_wandb:
        cfg.wandb.enabled = False
        # Strip 'wandb' from report_to so HF's WandbCallback doesn't fire either.
        parts = [p.strip() for p in cfg.train.report_to.split(",") if p.strip() and p.strip() != "wandb"]
        cfg.train.report_to = ",".join(parts) if parts else "none"

    # Cache was already redirected at import time by bootstrap_cache_from_argv().
    # Keep setup_cache_dir() as a belt-and-braces no-op for env vars that might
    # have been set differently by the CLI (e.g. --cache_dir via argparse).
    setup_cache_dir(cfg.data.cache_dir)

    # V100 / Volta doesn't support bf16 — fall back to fp16 automatically.
    resolve_precision(cfg.train)

    # Derive gradient_accumulation_steps from effective_batch_size and the
    # runtime WORLD_SIZE if it wasn't pinned in the YAML / on the CLI. Keeps
    # effective batch constant across 1 / 2 / 4 / 8-GPU runs.
    resolve_grad_accum(cfg.train, _world_size())
    if _is_main_process():
        print(
            f"[batch] effective_batch_size={cfg.train.effective_batch_size}  "
            f"per_device={cfg.train.per_device_train_batch_size}  "
            f"world_size={_world_size()}  "
            f"gradient_accumulation_steps={cfg.train.gradient_accumulation_steps}"
        )
        print(
            f"[dataloader] num_workers={cfg.train.dataloader_num_workers}  "
            f"prefetch_factor={cfg.train.dataloader_prefetch_factor}  "
            f"cache_dir={cfg.data.cache_dir}"
        )

    # Let fp16/bf16 matmuls use reduced-precision intermediate accumulation.
    # ~1-2% throughput win on V100 fp16, imperceptible numerical impact for ASR.
    import torch

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    print(f"Loading tokenizer (local_path={cfg.data.tokenizer_dir}, cache_dir={cfg.data.cache_dir})")
    tokenizer = load_tokenizer(cfg.data.tokenizer_dir, cache_dir=cfg.data.cache_dir)

    # Under DDP, N ranks calling preprocess_dataset() concurrently would race on
    # save_to_disk() and duplicate 100+ GB of feature extraction. Require the
    # cache to be pre-baked via scripts/preprocess.py before multi-GPU launch;
    # all ranks then just load_from_disk the finished arrow shards.
    if _world_size() > 1:
        key = _preprocess_cache_key(cfg.model, tokenizer, cfg.data)
        save_dir = _preprocess_cache_dir(cfg.data, key)
        if save_dir is None or not save_dir.exists():
            raise RuntimeError(
                f"Preprocessed cache not found at {save_dir}. "
                "Run `python scripts/preprocess.py --config <config>` on a single "
                "process before launching multi-GPU training."
            )

    # Short-circuit when the preprocessed cache already exists: load it
    # directly and skip ``load_librispeech``, which otherwise calls
    # ``datasets.load_dataset`` for each split and re-downloads the raw
    # 1.2 TB audio archive whenever the HF datasets cache is empty (e.g.
    # on a fresh per-node /tmp). The preprocessed shards are the only thing
    # the trainer actually consumes downstream.
    pre_key = _preprocess_cache_key(cfg.model, tokenizer, cfg.data)
    pre_dir = _preprocess_cache_dir(cfg.data, pre_key)
    if pre_dir is not None and pre_dir.exists():
        from datasets import load_from_disk

        print(f"[preprocess] loading cached dataset from {pre_dir} (raw download skipped)")
        ds = load_from_disk(str(pre_dir))
    else:
        print("Loading LibriSpeech …")
        ds = load_librispeech(cfg.data)
        print("Preprocessing dataset (this caches to disk after first run)")
        ds = preprocess_dataset(ds, cfg.model, tokenizer, cfg.data)

    # Stored Arrow type is ``list<list<float>>``; default __getitem__ would
    # box every leaf float into a CPython PyFloat (~130K allocs / row), then
    # ``torch.as_tensor`` walks the nested list to copy it back out — pure
    # CPython object churn that dominates the dataloader CPU budget. The
    # numpy formatter calls PyArrow's C++ kernel to emit a flat float32 array
    # directly, so the collator's ``torch.as_tensor`` becomes a near-zero-copy
    # ndarray view. Net: ~5-6× faster per row, GPU stays fed.
    ds = ds.with_format("numpy")

    model = build_model(cfg.model, tokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.1f}M ({n_trainable / 1e6:.1f}M trainable)")

    # Initialize wandb BEFORE constructing the Trainer so HF's WandbCallback
    # picks up our run (instead of starting its own). Rank-0 only under DDP —
    # HF's WandbCallback also no-ops on non-zero ranks, so the other ranks never
    # touch wandb.
    wandb_run = init_wandb(
        cfg,
        extra_config={
            "n_parameters": n_params,
            "n_trainable_parameters": n_trainable,
            "train_dataset_size": len(ds["train"]),
            "eval_dataset_size": len(ds["validation"]),
            "tokenizer_vocab_size": len(tokenizer),
            "resume_from_checkpoint": args.resume_from_checkpoint,
        },
    ) if _is_main_process() else None
    if wandb_run is not None:
        wandb_run.summary["n_parameters"] = n_params
        wandb_run.summary["n_trainable_parameters"] = n_trainable
        wandb_run.summary["train_dataset_size"] = len(ds["train"])

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
        n_mels=cfg.model.n_mels,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.train.output_dir,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        warmup_steps=cfg.train.warmup_steps,
        num_train_epochs=cfg.train.num_train_epochs,
        weight_decay=cfg.train.weight_decay,
        adam_beta1=cfg.train.adam_beta1,
        adam_beta2=cfg.train.adam_beta2,
        adam_epsilon=cfg.train.adam_epsilon,
        max_grad_norm=cfg.train.max_grad_norm,
        label_smoothing_factor=cfg.train.label_smoothing_factor,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=cfg.train.logging_steps,
        save_total_limit=cfg.train.save_total_limit,
        bf16=cfg.train.bf16,
        fp16=cfg.train.fp16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        report_to=cfg.train.report_to,
        run_name=cfg.wandb.run_name,  # used by wandb via HF integration
        seed=cfg.train.seed,
        predict_with_generate=True,
        generation_max_length=cfg.train.generation_max_length,
        generation_num_beams=cfg.train.generation_num_beams,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=cfg.train.dataloader_num_workers,
        dataloader_persistent_workers=True,
        dataloader_prefetch_factor=cfg.train.dataloader_prefetch_factor,
        remove_unused_columns=False,
        group_by_length=cfg.train.group_by_length,
        length_column_name="input_length",
        optim="adamw_torch_fused",
        # A few params inside ``Wav2Vec2ConformerEncoder`` (e.g. the
        # ``pos_conv_embed`` submodule — conv + layernorm ≈ 3 params) are
        # instantiated unconditionally but only used on non-rotary position-
        # embedding paths. We're on rotary, so they never feed into loss, and
        # DDP crashes on step 2 without this flag. Small per-step scan cost.
        ddp_find_unused_parameters=True,
    )

    # Order matters: EmptyCacheCallback must run AFTER PredictionsTableCallback
    # and CTCEvalCallback (HF fires callbacks in list order), so we flush the
    # allocator once all eval passes (main, preds table, CTC) are done.
    callbacks = []
    if wandb_is_enabled(cfg):
        callbacks.append(EpochLoggerCallback())
        if cfg.wandb.log_preds_table:
            callbacks.append(
                PredictionsTableCallback(
                    tokenizer=tokenizer,
                    eval_dataset=ds["validation"],
                    data_collator=collator,
                    n_samples=cfg.wandb.log_preds_n,
                )
            )
    if cfg.model.ctc_enabled:
        callbacks.append(
            CTCEvalCallback(
                tokenizer=tokenizer,
                eval_dataset=ds["validation"],
                data_collator=collator,
                batch_size=cfg.train.per_device_eval_batch_size,
                autocast_dtype=autocast_dtype(cfg.train),
            )
        )
    if args.early_eval_frac is not None and args.early_eval_frac > 0:
        # Compute target step from effective batch size. World size and
        # grad_accum both scale effective batch, so steps_per_epoch divides by
        # the product. Clamp to >=1 so frac=0.001 on tiny subsets still fires.
        eff_batch = (
            cfg.train.per_device_train_batch_size
            * max(1, _world_size())
            * max(1, cfg.train.gradient_accumulation_steps)
        )
        steps_per_epoch = max(1, len(ds["train"]) // eff_batch)
        target_step = max(1, int(steps_per_epoch * args.early_eval_frac))
        print(
            f"[early-eval] firing one-shot eval at step {target_step} "
            f"({args.early_eval_frac:.1%} of epoch 1, ~{steps_per_epoch} steps/epoch)"
        )
        callbacks.append(OneShotEvalCallback(target_step=target_step))
    callbacks.append(StepTimerCallback())
    callbacks.append(EmptyCacheCallback())
    # Must run AFTER EmptyCacheCallback has done its work but order here doesn't
    # matter; on_save only needs the saved directory to exist on disk.
    callbacks.append(EpochCheckpointRenameCallback())
    callbacks.append(FreezeInputNormCallback(cfg.model.input_normalize_freeze_epochs))
    if cfg.model.spec_aug_warmup_steps > 0:
        callbacks.append(SpecAugWarmupCallback(cfg.model.spec_aug_warmup_steps))
    if cfg.train.swa_enabled:
        callbacks.append(
            SWACallback(
                start_frac=cfg.train.swa_start_frac,
                save_dir=Path(cfg.train.output_dir) / "final-swa",
            )
        )

    # Distinct speeds in the cache — if >1, the train split has that many
    # variant rows per clip (laid out contiguously) and the Trainer will
    # subsample to one per clip per epoch via RandomSpeedVariantSampler.
    n_speed_variants = len({round(float(s), 4) for s in cfg.data.speed_perturbations})
    trainer_cls = HybridSeq2SeqTrainer if cfg.model.ctc_enabled else SpeedAugSeq2SeqTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        compute_metrics=build_compute_metrics(tokenizer),
        processing_class=tokenizer,
        callbacks=callbacks or None,
        n_speed_variants=n_speed_variants,
    )
    # Swap HF's single-bar ProgressCallback for our two-bar overall+epoch view.
    trainer.remove_callback(ProgressCallback)
    trainer.add_callback(DualProgressCallback())
    # SpeechEncoderDecoderModel.forward accepts **kwargs, which Trainer reads
    # as "model accepts num_items_in_batch" and injects the kwarg into inputs.
    # SED then forwards it to BartForCausalLM.forward, which doesn't accept it.
    # Opt out of loss-kwarg forwarding — gradient accumulation still works.
    trainer.model_accepts_loss_kwargs = False

    if wandb_run is not None and cfg.wandb.watch_model:
        import wandb

        wandb.watch(model, log="gradients", log_freq=max(500, cfg.train.logging_steps * 5))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = Path(cfg.train.output_dir) / "final"
    # ``trainer.save_model`` is DDP-aware (saves on rank 0 only); guard the
    # tokenizer + artifact upload explicitly so non-zero ranks don't race on
    # the same output directory.
    trainer.save_model(str(final_dir))
    is_main = trainer.is_world_process_zero()
    if is_main:
        tokenizer.save_pretrained(str(final_dir))

        # If SWA ran, make ``final-swa/`` a drop-in checkpoint dir by copying
        # over the config + tokenizer metadata ``trainer.save_model`` wrote to
        # ``final/``. Only the weights themselves differ between the two dirs.
        swa_dir = Path(cfg.train.output_dir) / "final-swa"
        if cfg.train.swa_enabled and (swa_dir / "model.safetensors").exists():
            import shutil

            for name in (
                "config.json",
                "generation_config.json",
                "training_args.bin",
                "sentencepiece.model",
                "tokenizer_info.json",
            ):
                src = final_dir / name
                if src.exists():
                    shutil.copy2(src, swa_dir / name)

    # Post-training evaluation. evaluate.py είναι intentionally wandb-free and
    # single-process; we shell out from rank 0, strip torchrun's distributed env
    # vars so the subprocess doesn't try to re-init NCCL, then push the final
    # WER onto this run's wandb summary.
    final_wer: float | None = None
    if is_main:
        # Free trainer-held GPU memory (model + AdamW state ≈ 3× model size)
        # before the subprocess spins up a second model copy on the same GPU.
        import gc

        del trainer
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        eval_json = Path(cfg.train.output_dir) / "final_wer.json"
        sub_env = os.environ.copy()
        sub_env["WORLD_SIZE"] = "1"
        sub_env["RANK"] = "0"
        sub_env["LOCAL_RANK"] = "0"
        for _k in ("MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID"):
            sub_env.pop(_k, None)

        cmd = [
            sys.executable,
            str(Path(__file__).parent / "evaluate.py"),
            "--config", args.config,
            "--checkpoint", str(final_dir),
            "--split", cfg.data.test_split,
            "--output_json", str(eval_json),
        ]
        print(f"[post-train-eval] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, env=sub_env)
            with open(eval_json) as fh:
                results = json.load(fh)
            final_wer = float(results[-1]["wer"])
            print(f"[post-train-eval] final WER on {cfg.data.test_split}: {final_wer:.4f}")
        except (subprocess.CalledProcessError, OSError, KeyError, ValueError) as exc:
            # Don't crash training because eval failed — weights are already saved.
            print(f"[post-train-eval] skipped: {exc}")

    if wandb_run is not None and is_main:
        import wandb

        if final_wer is not None:
            key = f"final_wer/{cfg.data.test_split}"
            wandb_run.summary[key] = final_wer
            wandb.log({key: final_wer})

        # Save the final model as a wandb Artifact for easy downstream eval.
        artifact = wandb.Artifact(
            name=f"{cfg.wandb.project}-final",
            type="model",
            metadata={
                "n_parameters": n_params,
                "subset": cfg.data.subset,
                "num_train_epochs": cfg.train.num_train_epochs,
                "final_wer": final_wer,
            },
        )
        artifact.add_dir(str(final_dir))
        wandb_run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
