"""Train the Conformer AED model on LibriSpeech with ``Seq2SeqTrainer``."""
from __future__ import annotations

import argparse
import os
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

from conformer_asr.config import autocast_dtype, load_config, resolve_precision  # noqa: E402
from conformer_asr.data import (  # noqa: E402
    DataCollatorSpeechSeq2SeqWithPadding,
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
    init_wandb,
    wandb_is_enabled,
)


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

    def on_train_end(self, args, state, control, **kwargs):
        if self.epoch_bar is not None:
            self.epoch_bar.close()
            self.epoch_bar = None
        if self.overall_bar is not None:
            self.overall_bar.close()
            self.overall_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.overall_bar is None:
            return
        shown = {k: v for k, v in logs.items() if k != "total_flos"}
        self.overall_bar.write(str(shown))


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


class HybridSeq2SeqTrainer(Seq2SeqTrainer):
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
    """

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

        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/conformer_small.yaml")
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

    # Let fp16/bf16 matmuls use reduced-precision intermediate accumulation.
    # ~1-2% throughput win on V100 fp16, imperceptible numerical impact for ASR.
    import torch

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    print(f"Loading tokenizer from {cfg.data.tokenizer_dir}")
    tokenizer = load_tokenizer(cfg.data.tokenizer_dir)

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

    print("Loading LibriSpeech …")
    ds = load_librispeech(cfg.data)

    print("Preprocessing dataset (this caches to disk after first run)")
    ds = preprocess_dataset(ds, cfg.model, tokenizer, cfg.data)

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
    callbacks.append(EmptyCacheCallback())

    trainer_cls = HybridSeq2SeqTrainer if cfg.model.ctc_enabled else Seq2SeqTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        compute_metrics=build_compute_metrics(tokenizer),
        processing_class=tokenizer,
        callbacks=callbacks or None,
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
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_dir))

    if wandb_run is not None and trainer.is_world_process_zero():
        import wandb

        # Save the final model as a wandb Artifact for easy downstream eval.
        artifact = wandb.Artifact(
            name=f"{cfg.wandb.project}-final",
            type="model",
            metadata={
                "n_parameters": n_params,
                "subset": cfg.data.subset,
                "num_train_epochs": cfg.train.num_train_epochs,
            },
        )
        artifact.add_dir(str(final_dir))
        wandb_run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
