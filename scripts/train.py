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
)
from conformer_asr.metrics import build_compute_metrics  # noqa: E402
from conformer_asr.model import build_model  # noqa: E402
from conformer_asr.tokenizer import load_tokenizer  # noqa: E402


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
    p.add_argument("--resume_from_checkpoint", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k not in {"config", "resume_from_checkpoint"}}
    cfg = load_config(args.config, overrides=overrides)

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
    print(f"Model parameters: {n_params / 1e6:.1f}M")

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

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        compute_metrics=build_compute_metrics(tokenizer),
        processing_class=feature_extractor,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(Path(cfg.train.output_dir) / "final"))
    tokenizer.save_pretrained(str(Path(cfg.train.output_dir) / "final"))


if __name__ == "__main__":
    main()
