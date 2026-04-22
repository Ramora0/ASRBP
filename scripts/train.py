"""Train the Conformer AED model on LibriSpeech with ``Seq2SeqTrainer``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import (  # noqa: E402
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Wav2Vec2FeatureExtractor,
)

from conformer_asr.config import load_config  # noqa: E402
from conformer_asr.data import (  # noqa: E402
    DataCollatorSpeechSeq2SeqWithPadding,
    load_librispeech,
    preprocess_dataset,
    setup_cache_dir,
)
from conformer_asr.metrics import build_compute_metrics  # noqa: E402
from conformer_asr.model import build_model  # noqa: E402
from conformer_asr.tokenizer import load_tokenizer  # noqa: E402
from conformer_asr.wandb_utils import (  # noqa: E402
    EpochLoggerCallback,
    PredictionsTableCallback,
    init_wandb,
    wandb_is_enabled,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/conformer_small.yaml")
    p.add_argument("--subset", choices=["clean100", "clean460", "all960"])
    p.add_argument("--output_dir")
    p.add_argument("--max_steps", type=int)
    p.add_argument("--eval_steps", type=int)
    p.add_argument("--save_steps", type=int)
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
    return p.parse_args()


def _flatten_overrides(args: argparse.Namespace) -> dict:
    overrides = {k: v for k, v in vars(args).items() if k not in {"config", "resume_from_checkpoint", "no_wandb"}}
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

    # Point HF caches at scratch BEFORE importing/loading anything else that
    # might lazily trigger a download.
    setup_cache_dir(cfg.data.cache_dir)
    print(f"HF cache_dir: {cfg.data.cache_dir}")

    print(f"Loading tokenizer from {cfg.data.tokenizer_dir}")
    tokenizer = load_tokenizer(cfg.data.tokenizer_dir)

    print("Loading LibriSpeech …")
    ds = load_librispeech(cfg.data)

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=cfg.data.sampling_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    print("Preprocessing dataset (this caches to disk after first run)")
    ds = preprocess_dataset(ds, feature_extractor, tokenizer, cfg.data)

    model = build_model(cfg.model, tokenizer)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params / 1e6:.1f}M ({n_trainable / 1e6:.1f}M trainable)")

    # Initialize wandb BEFORE constructing the Trainer so HF's WandbCallback
    # picks up our run (instead of starting its own).
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
    )
    if wandb_run is not None:
        wandb_run.summary["n_parameters"] = n_params
        wandb_run.summary["n_trainable_parameters"] = n_trainable
        wandb_run.summary["train_dataset_size"] = len(ds["train"])

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.train.output_dir,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        warmup_steps=cfg.train.warmup_steps,
        max_steps=cfg.train.max_steps,
        weight_decay=cfg.train.weight_decay,
        adam_beta1=cfg.train.adam_beta1,
        adam_beta2=cfg.train.adam_beta2,
        adam_epsilon=cfg.train.adam_epsilon,
        label_smoothing_factor=cfg.train.label_smoothing_factor,
        lr_scheduler_type="inverse_sqrt",
        eval_strategy="steps",
        eval_steps=cfg.train.eval_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        logging_steps=cfg.train.logging_steps,
        save_total_limit=cfg.train.save_total_limit,
        bf16=cfg.train.bf16,
        fp16=cfg.train.fp16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        group_by_length=cfg.train.group_by_length,
        length_column_name="input_length",
        report_to=cfg.train.report_to,
        run_name=cfg.wandb.run_name,  # used by wandb via HF integration
        seed=cfg.train.seed,
        predict_with_generate=True,
        generation_max_length=cfg.train.generation_max_length,
        generation_num_beams=cfg.train.generation_num_beams,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

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

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        compute_metrics=build_compute_metrics(tokenizer),
        processing_class=feature_extractor,
        callbacks=callbacks or None,
    )

    if wandb_run is not None and cfg.wandb.watch_model:
        import wandb

        wandb.watch(model, log="gradients", log_freq=max(500, cfg.train.logging_steps * 5))

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    final_dir = Path(cfg.train.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    if wandb_run is not None:
        import wandb

        # Save the final model as a wandb Artifact for easy downstream eval.
        artifact = wandb.Artifact(
            name=f"{cfg.wandb.project}-final",
            type="model",
            metadata={
                "n_parameters": n_params,
                "subset": cfg.data.subset,
                "max_steps": cfg.train.max_steps,
            },
        )
        artifact.add_dir(str(final_dir))
        wandb_run.log_artifact(artifact)
        wandb.finish()


if __name__ == "__main__":
    main()
