"""Helpers for Weights & Biases logging.

Centralizes run initialization + callbacks so both train.py and evaluate.py use
the same project/entity/tagging conventions.
"""
from __future__ import annotations

import os
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from .config import Config, WandbConfig


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

        model.eval()
        device = next(model.parameters()).device
        batch = self.data_collator(self.sample_features)
        input_features = batch["input_features"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        gen_kwargs = {"max_length": args.generation_max_length, "num_beams": 1}
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
