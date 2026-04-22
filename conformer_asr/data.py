from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerFast, Wav2Vec2FeatureExtractor

from .config import DataConfig
from .tokenizer import normalize_text


_TRAIN_SUBSETS: dict[str, list[str]] = {
    "clean100": ["train.clean.100"],
    "clean460": ["train.clean.100", "train.clean.360"],
    "all960": ["train.clean.100", "train.clean.360", "train.other.500"],
}


def setup_cache_dir(cache_dir: str | None) -> str | None:
    """Direct HF downloads (datasets + transformers hub) into ``cache_dir``.

    Call this once, as early as possible in every entrypoint — subprocesses
    spawned later (e.g. HF ``datasets`` workers) inherit these env vars, so
    arrow shards, audio files, and any model artifacts all land on scratch
    instead of $HOME.
    """
    if not cache_dir:
        return None
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(cache_dir) / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(cache_dir) / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_dir) / "transformers"))
    return cache_dir


def _load_split(dataset_id: str, split: str, cache_dir: str | None):
    return load_dataset(
        dataset_id,
        split=split,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )


def load_librispeech(cfg: DataConfig) -> DatasetDict:
    """Load train / validation / test splits according to the chosen subset."""
    if cfg.subset not in _TRAIN_SUBSETS:
        raise ValueError(
            f"Unknown subset '{cfg.subset}'. Choose from {list(_TRAIN_SUBSETS)}."
        )
    setup_cache_dir(cfg.cache_dir)
    train_parts = [_load_split(cfg.dataset_id, s, cfg.cache_dir) for s in _TRAIN_SUBSETS[cfg.subset]]
    train = train_parts[0] if len(train_parts) == 1 else concatenate_datasets(train_parts)
    validation = _load_split(cfg.dataset_id, cfg.eval_split, cfg.cache_dir)
    test = _load_split(cfg.dataset_id, cfg.test_split, cfg.cache_dir)

    ds = DatasetDict({"train": train, "validation": validation, "test": test})
    ds = ds.cast_column("audio", Audio(sampling_rate=cfg.sampling_rate))
    return ds


def _duration_seconds(example: dict[str, Any], sampling_rate: int) -> float:
    audio = example["audio"]
    return len(audio["array"]) / sampling_rate


def preprocess_dataset(
    ds: DatasetDict,
    feature_extractor: Wav2Vec2FeatureExtractor,
    tokenizer: PreTrainedTokenizerFast,
    cfg: DataConfig,
) -> DatasetDict:
    """Filter by duration, extract audio features, tokenize labels."""

    sr = cfg.sampling_rate
    min_len = int(cfg.min_audio_seconds * sr)
    max_len = int(cfg.max_audio_seconds * sr)

    def is_valid_length(example):
        n = len(example["audio"]["array"])
        return min_len <= n <= max_len

    # Only filter training clips by max length — we still want to score all eval audio.
    ds["train"] = ds["train"].filter(is_valid_length, num_proc=cfg.num_proc)

    def prepare(batch):
        audio = batch["audio"]
        inputs = feature_extractor(
            audio["array"],
            sampling_rate=sr,
            return_tensors=None,
        )
        batch["input_values"] = inputs["input_values"][0]
        batch["input_length"] = len(batch["input_values"])
        text = normalize_text(batch["text"])
        batch["labels"] = tokenizer(text).input_ids
        return batch

    remove_cols = {
        split: [c for c in ds[split].column_names if c not in {"input_length"}]
        for split in ds
    }
    for split in ds:
        ds[split] = ds[split].map(
            prepare,
            remove_columns=remove_cols[split],
            num_proc=cfg.num_proc,
            desc=f"preprocess {split}",
        )
    return ds


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads audio inputs and tokenized labels for ``SpeechEncoderDecoderModel``.

    Audio is padded via ``Wav2Vec2FeatureExtractor`` (float tensor + attention mask).
    Labels are right-padded with -100 so CE loss ignores pad positions. The BOS that
    the decoder expects as its first token is added by the model's
    ``shift_tokens_right`` using ``decoder_start_token_id`` — so we strip the leading
    BOS from labels to avoid duplicating it.
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: PreTrainedTokenizerFast
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audio_features = [
            {"input_values": f["input_values"]} for f in features
        ]
        batch = self.feature_extractor.pad(
            audio_features,
            padding=True,
            return_tensors="pt",
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(
            label_features,
            padding=True,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # The tokenizer's post-processor prepends BOS. The Seq2Seq model prepends
        # decoder_start_token_id itself (via shift_tokens_right), so drop the
        # leading BOS token here to avoid emitting it twice.
        if (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
