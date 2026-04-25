"""Helpers for Weights & Biases logging.

Centralizes run initialization + callbacks so both train.py and evaluate.py use
the same project/entity/tagging conventions.
"""
from __future__ import annotations

import os
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .config import Config, WandbConfig, autocast_dtype


def _parse_report_to(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


def wandb_is_enabled(cfg: Config) -> bool:
    if not cfg.wandb.enabled:
        return False
    return "wandb" in _parse_report_to(cfg.train.report_to)


def init_wandb(cfg: Config, extra_config: dict[str, Any] | None = None, job_type: str = "train"):
    """Initialize a wandb run before constructing the HF Trainer.

    The HF ``WandbCallback`` detects the existing ``wandb.run`` and reuses it,
    so we get a single run with both our metadata and the Trainer's auto-logged
    metrics (loss, eval/*, learning_rate, epoch, …).

    Returns the ``wandb`` module if enabled, else ``None``.
    """
    if not wandb_is_enabled(cfg):
        return None

    import wandb  # local import so the package is optional for users who don't log

    # Make sure the HF WandbCallback uses the same project we configure here.
    os.environ.setdefault("WANDB_PROJECT", cfg.wandb.project)
    if cfg.wandb.entity:
        os.environ.setdefault("WANDB_ENTITY", cfg.wandb.entity)
    # Don't let HF try to save checkpoints as wandb artifacts by default (heavy).
    os.environ.setdefault("WANDB_LOG_MODEL", "false")

    run_config = cfg.to_dict()
    if extra_config:
        run_config["_extra"] = extra_config

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        group=cfg.wandb.group,
        tags=cfg.wandb.tags or None,
        notes=cfg.wandb.notes or None,
        job_type=job_type,
        config=run_config,
        reinit=False,
    )

    # Define `epoch` as a first-class metric with global_step as the x-axis so
    # it plots cleanly alongside loss / eval metrics.
    wandb.define_metric("train/global_step")
    wandb.define_metric("epoch", step_metric="train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("eval/*", step_metric="train/global_step")

    return run


class EpochLoggerCallback(TrainerCallback):
    """Logs the current (possibly fractional) epoch to wandb on every logging step.

    Trainer already includes ``epoch`` in its log dict, so wandb sees it via
    the default callback. But that only fires at ``logging_steps`` cadence;
    this callback also mirrors epoch to ``wandb.summary`` so it's always
    visible on the run overview page, and logs it explicitly at eval time.
    """

    def __init__(self) -> None:
        self._wandb = None

    def _ensure_wandb(self):
        if self._wandb is None:
            try:
                import wandb

                if wandb.run is not None:
                    self._wandb = wandb
            except ImportError:
                self._wandb = None
        return self._wandb

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs,
    ):
        wandb = self._ensure_wandb()
        if wandb is None or state.epoch is None:
            return
        wandb.run.summary["epoch"] = state.epoch
        wandb.run.summary["global_step"] = state.global_step

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ):
        wandb = self._ensure_wandb()
        if wandb is None or state.epoch is None:
            return
        wandb.log(
            {
                "epoch": state.epoch,
                "train/global_step": state.global_step,
            },
            step=state.global_step,
        )


class PredictionsTableCallback(TrainerCallback):
    """Logs a small table of (reference, prediction) pairs at each evaluation.

    Cheap and very useful for eyeballing what the model is learning beyond WER.
    """

    def __init__(
        self,
        tokenizer,
        eval_dataset,
        data_collator,
        n_samples: int = 32,
    ) -> None:
        import random

        self._wandb = None
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        rng = random.Random(0)
        indices = list(range(len(eval_dataset)))
        rng.shuffle(indices)
        self.sample_indices = indices[:n_samples]
        self.sample_features = [eval_dataset[i] for i in self.sample_indices]

    def _ensure_wandb(self):
        if self._wandb is None:
            try:
                import wandb

                if wandb.run is not None:
                    self._wandb = wandb
            except ImportError:
                self._wandb = None
        return self._wandb

    @staticmethod
    def _flush_cuda_cache():
        """Return cached blocks from the PyTorch caching allocator to the driver.

        The main eval loop leaves a heap full of variable-size KV-cache and
        attention-scratch blocks. Generating on 16 more samples right after can
        push peak memory into OOM on V100 32GB under DDP. ``empty_cache`` frees
        those blocks so the upcoming generate sees a clean pool.
        """
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ):
        wandb = self._ensure_wandb()
        model = kwargs.get("model")
        if wandb is None or model is None:
            return

        import torch

        # Flush the allocator before we do our own generate — the main eval
        # loop just filled the cache with blocks sized for its own batches.
        self._flush_cuda_cache()

        model.eval()
        device = next(model.parameters()).device
        batch = self.data_collator(self.sample_features)
        input_features = batch["input_features"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        gen_kwargs = {"max_length": args.generation_max_length, "num_beams": 1}
        try:
            with torch.no_grad():
                generated = model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )
            labels = batch["labels"]
            # -100 → pad for decoding
            labels = labels.masked_fill(labels == -100, self.tokenizer.pad_token_id)
            preds = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            table = wandb.Table(columns=["step", "epoch", "reference", "prediction"])
            for ref, pred in zip(refs, preds):
                table.add_data(state.global_step, state.epoch, ref, pred)
            wandb.log(
                {"eval/sample_predictions": table, "epoch": state.epoch},
                step=state.global_step,
            )
        finally:
            # Drop our own generate's KV cache before the next epoch's
            # training steps start allocating activations.
            del batch, input_features
            if attention_mask is not None:
                del attention_mask
            self._flush_cuda_cache()


class CTCEvalCallback(TrainerCallback):
    """Log CTC-only WER over the validation set at each eval step.

    Runs an encoder-only forward pass (no autoregressive generate) per batch,
    taps ``ctc_logits`` + ``encoder_attention_mask`` from the model output,
    greedy-decodes, and reports ``eval/ctc_wer`` alongside the AED WER that
    Seq2SeqTrainer already logs. Exists as a separate callback — rather than
    being folded into ``compute_metrics`` — because the Trainer's eval loop
    only surfaces ``generate()`` outputs, not encoder hidden states.

    CTC WER is usually 2–5× higher than AED WER on from-scratch training, but
    it's the right signal for diagnosing whether the *encoder* alignment is
    learning: if AED WER drops while CTC WER stays flat, the decoder is
    memorizing rather than being told where to listen.
    """

    def __init__(
        self,
        tokenizer,
        eval_dataset,
        data_collator,
        batch_size: int = 8,
        autocast_dtype=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.batch_size = int(batch_size)
        self.autocast_dtype = autocast_dtype
        self._wandb = None

    def _ensure_wandb(self):
        if self._wandb is None:
            try:
                import wandb

                if wandb.run is not None:
                    self._wandb = wandb
            except ImportError:
                self._wandb = None
        return self._wandb

    @staticmethod
    def _unwrap(model):
        # DDP / accelerate wrap the model; reach the raw ``ConformerAEDWithCTC``
        # so we can read ``ctc_blank_id`` and know the ``ctc_head`` is present.
        while hasattr(model, "module"):
            model = model.module
        return model

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, float] | None = None,
        **kwargs,
    ):
        # Rank-0 only — mirrors ``PredictionsTableCallback`` and avoids
        # cross-rank duplication of work.
        if not state.is_world_process_zero:
            return

        model = kwargs.get("model")
        if model is None:
            return
        unwrapped = self._unwrap(model)
        if not hasattr(unwrapped, "ctc_head"):
            return  # model built without CTC — nothing to report

        import torch

        from .metrics import compute_wer, ctc_greedy_decode

        was_training = unwrapped.training
        unwrapped.eval()
        device = next(unwrapped.parameters()).device

        preds_all: list[str] = []
        refs_all: list[str] = []
        # Weighted average: each batch contributes ctc_loss × n_frames so the
        # dataset-level mean matches what an unbatched pass would produce
        # (otherwise short-audio batches would be under-weighted vs long ones).
        ctc_loss_sum = 0.0
        ctc_loss_frames = 0

        use_autocast = (
            device.type == "cuda"
            and self.autocast_dtype is not None
            and self.autocast_dtype != torch.float32
        )

        n = len(self.eval_dataset)
        try:
            with torch.no_grad():
                for start in range(0, n, self.batch_size):
                    end = min(start + self.batch_size, n)
                    batch_feats = [self.eval_dataset[i] for i in range(start, end)]
                    batch = self.data_collator(batch_feats)
                    input_features = batch["input_features"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    # Forward decoder_input_ids explicitly. The collator
                    # precomputes it (shift_right of labels); relying on
                    # SpeechEncoderDecoderModel.forward to derive it from
                    # ``labels`` is brittle across transformers versions —
                    # when it silently doesn't fire, BartDecoder gets called
                    # with both inputs unset and raises.
                    decoder_input_ids = batch["decoder_input_ids"].to(device)

                    autocast_ctx = (
                        torch.autocast(device_type="cuda", dtype=self.autocast_dtype)
                        if use_autocast
                        else torch.autocast(device_type="cpu", enabled=False)
                    )
                    with autocast_ctx:
                        outputs = unwrapped(
                            input_features=input_features,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            labels=labels,
                        )

                    ctc_logits = outputs.ctc_logits
                    enc_mask = outputs.encoder_attention_mask
                    input_lengths = enc_mask.sum(-1).long() if enc_mask is not None else None
                    batch_frames = int(input_lengths.sum().item()) if input_lengths is not None else ctc_logits.size(1) * ctc_logits.size(0)

                    if outputs.ctc_loss is not None and batch_frames > 0:
                        ctc_loss_sum += float(outputs.ctc_loss.detach().float().item()) * batch_frames
                        ctc_loss_frames += batch_frames

                    pred_ids = ctc_greedy_decode(
                        ctc_logits.float(),
                        input_lengths=input_lengths,
                        blank=unwrapped.ctc_blank_id,
                    )
                    preds = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

                    label_ids = labels.masked_fill(labels == -100, self.tokenizer.pad_token_id)
                    refs = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

                    preds_all.extend(preds)
                    refs_all.extend(refs)
        finally:
            if was_training:
                unwrapped.train()

        wer = compute_wer(preds_all, refs_all)
        ctc_loss_mean = (ctc_loss_sum / ctc_loss_frames) if ctc_loss_frames > 0 else None

        if metrics is not None:
            metrics["eval/ctc_wer"] = float(wer)
            if ctc_loss_mean is not None:
                metrics["eval/ctc_loss"] = float(ctc_loss_mean)

        wandb = self._ensure_wandb()
        if wandb is not None:
            payload = {
                "eval/ctc_wer": float(wer),
                "epoch": state.epoch,
                "train/global_step": state.global_step,
            }
            if ctc_loss_mean is not None:
                payload["eval/ctc_loss"] = float(ctc_loss_mean)
            wandb.log(payload, step=state.global_step)
        loss_str = f" ctc_loss={ctc_loss_mean:.4f}" if ctc_loss_mean is not None else ""
        print(
            f"[ctc-eval] step={state.global_step} epoch={state.epoch} "
            f"ctc_wer={wer:.4f}{loss_str}"
        )


class SWACallback(TrainerCallback):
    """Stochastic Weight Averaging via ``torch.optim.swa_utils.AveragedModel``.

    Maintains an equal-weight running mean of the model weights across the
    final ``(1 - start_frac)`` fraction of training epochs, sampled once per
    epoch at ``on_epoch_end``. At ``on_train_end`` the averaged weights are
    written to ``<save_dir>/pytorch_model.bin`` (and a marker ``swa_info.json``)
    so ``evaluate.py --checkpoint <save_dir>`` can pick them up directly.

    Construction is deferred to the first epoch crossing ``start_frac`` so the
    initial weights (which never enter the average) don't sit in GPU memory
    during the first ~75% of training.
    """

    def __init__(self, start_frac: float, save_dir):
        from pathlib import Path

        self.start_frac = float(start_frac)
        self.save_dir = Path(save_dir)
        self.swa_model = None
        self._updates = 0

    @staticmethod
    def _unwrap(model):
        # DDP/compile wrappers expose the underlying module at ``.module``.
        inner = model
        while hasattr(inner, "module") and not isinstance(inner, type):
            nxt = getattr(inner, "module")
            if nxt is inner:
                break
            inner = nxt
        return inner

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ) -> None:
        import torch

        if model is None or state.epoch is None or args.num_train_epochs <= 0:
            return
        progress = state.epoch / args.num_train_epochs
        if progress < self.start_frac:
            return

        raw_model = self._unwrap(model)
        if self.swa_model is None:
            from torch.optim.swa_utils import AveragedModel

            self.swa_model = AveragedModel(raw_model)
            print(
                f"[swa] starting weight averaging at epoch {state.epoch:.2f} / "
                f"{args.num_train_epochs} (start_frac={self.start_frac})"
            )
        self.swa_model.update_parameters(raw_model)
        self._updates += 1

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        import json

        import torch

        # Save only on rank 0 to avoid DDP races. Under non-DDP, local_rank is -1.
        if args.local_rank not in (-1, 0):
            return
        if self.swa_model is None or self._updates == 0:
            print(
                f"[swa] start_frac={self.start_frac} was never reached "
                f"(epochs={args.num_train_epochs}); no SWA weights saved."
            )
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        state_dict = self.swa_model.module.state_dict()
        # Save as safetensors to match the format ``trainer.save_model`` writes
        # for ``final/``, so evaluate.py's safetensors-first loader picks it up
        # and the two checkpoint dirs are interchangeable. Tie-shared tensors
        # (e.g. ``decoder.lm_head.weight`` ↔ embedding) must share storage or
        # ``save_file`` refuses; make each a contiguous clone.
        from safetensors.torch import save_file

        state_dict = {k: v.detach().contiguous().clone() for k, v in state_dict.items()}
        save_file(state_dict, str(self.save_dir / "model.safetensors"))
        with open(self.save_dir / "swa_info.json", "w") as f:
            json.dump(
                {
                    "num_updates": self._updates,
                    "start_frac": self.start_frac,
                    "num_train_epochs": float(args.num_train_epochs),
                },
                f,
                indent=2,
            )
        print(
            f"[swa] wrote {self._updates}-epoch-averaged weights to "
            f"{self.save_dir}/model.safetensors"
        )
