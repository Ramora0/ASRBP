from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from transformers import PreTrainedTokenizerFast

from .config import DataConfig, ModelConfig
from .features import log_mel_spectrogram
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


def _preprocess_cache_key(
    mcfg: ModelConfig,
    tokenizer: PreTrainedTokenizerFast,
    cfg: DataConfig,
) -> str:
    """Stable hash of everything that affects preprocessing output.

    Covers subset/eval/test splits (dataset content), duration filter bounds,
    the log-Mel frontend params (n_mels/n_fft/hop_length/sampling_rate), and
    tokenizer state. Bump ``v`` whenever the prepare() logic itself changes.
    """
    try:
        tok_repr = tokenizer.backend_tokenizer.to_str()
    except Exception:
        tok_repr = json.dumps(sorted(tokenizer.get_vocab().items()))

    blob = json.dumps(
        {
            "v": 2,  # bumped when switching from raw-waveform FE to log-Mel
            "dataset_id": cfg.dataset_id,
            "subset": cfg.subset,
            "eval_split": cfg.eval_split,
            "test_split": cfg.test_split,
            "sampling_rate": cfg.sampling_rate,
            "max_audio_seconds": cfg.max_audio_seconds,
            "mel": {
                "n_mels": mcfg.n_mels,
                "n_fft": mcfg.n_fft,
                "hop_length": mcfg.hop_length,
            },
            "tok_sha": hashlib.sha1(tok_repr.encode("utf-8")).hexdigest(),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _preprocess_cache_dir(cfg: DataConfig, key: str) -> Path | None:
    if not cfg.cache_dir:
        return None
    return Path(cfg.cache_dir) / "preprocessed" / f"{cfg.subset}-{key}"


def preprocess_dataset(
    ds: DatasetDict,
    mcfg: ModelConfig,
    tokenizer: PreTrainedTokenizerFast,
    cfg: DataConfig,
) -> DatasetDict:
    """Filter by duration, compute log-Mel features, tokenize labels.

    The finished DatasetDict is saved to ``<cache_dir>/preprocessed/<key>``
    and reloaded on subsequent runs — HF's auto-fingerprint cache is unreliable
    here because the ``tokenizer`` closure doesn't hash deterministically
    across processes.

    Output columns per example:
      - ``input_features``: list[list[float]] of shape (T_mel, n_mels)
      - ``input_length``:   int = T_mel (used by ``group_by_length``)
      - ``labels``:         list[int] of token ids
    """

    key = _preprocess_cache_key(mcfg, tokenizer, cfg)
    save_dir = _preprocess_cache_dir(cfg, key)

    if save_dir is not None and save_dir.exists():
        print(f"[preprocess] loading cached dataset from {save_dir}")
        return load_from_disk(str(save_dir))

    sr = cfg.sampling_rate
    max_len = int(cfg.max_audio_seconds * sr)
    n_mels = mcfg.n_mels
    n_fft = mcfg.n_fft
    hop_length = mcfg.hop_length

    def is_valid_length(example):
        return len(example["audio"]["array"]) <= max_len

    # Only filter training clips by max length — we still want to score all eval audio.
    ds["train"] = ds["train"].filter(is_valid_length, num_proc=cfg.num_proc)

    def prepare(batch):
        audio = batch["audio"]
        wav = np.asarray(audio["array"], dtype=np.float32)
        mel = log_mel_spectrogram(
            wav,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            sampling_rate=sr,
        )  # (T_mel, n_mels) torch.float32
        batch["input_features"] = mel.numpy().tolist()
        batch["input_length"] = int(mel.shape[0])
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

    if save_dir is not None:
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir = save_dir.with_name(save_dir.name + ".tmp")
        print(f"[preprocess] saving preprocessed dataset to {save_dir}")
        ds.save_to_disk(str(tmp_dir))
        os.replace(tmp_dir, save_dir)
        ds = load_from_disk(str(save_dir))

    return ds


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pads log-Mel features and tokenized labels for ``SpeechEncoderDecoderModel``.

    Each input example carries ``input_features`` of shape ``(T_i, n_mels)``.
    We stack into a ``(B, T_max, n_mels)`` float tensor, zero-pad shorter
    sequences along the time axis, and build an ``attention_mask`` of shape
    ``(B, T_max)`` where 1 marks valid frames. Labels are right-padded with
    -100 so CE loss ignores pad positions.

    The BOS that the decoder expects as its first token is added by the model's
    ``shift_tokens_right`` using ``decoder_start_token_id`` — so we strip the
    leading BOS from labels to avoid duplicating it.

    We also precompute ``decoder_input_ids`` here. ``Seq2SeqTrainer.compute_loss``
    pops ``labels`` from inputs before calling the model when a label smoother
    is active (``label_smoothing_factor > 0``); without ``labels``,
    ``SpeechEncoderDecoderModel.forward`` won't derive ``decoder_input_ids``
    itself, and the decoder then gets called with both ids/embeds unset — which
    BartDecoder rejects. Providing ``decoder_input_ids`` in the batch avoids that.
    """

    tokenizer: PreTrainedTokenizerFast
    decoder_start_token_id: int
    n_mels: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch: dict[str, torch.Tensor] = {}

        # (T_i, n_mels) per example → stack into (B, T_max, n_mels) with
        # zero padding along the time axis + matching attention mask.
        feats = [torch.as_tensor(f["input_features"], dtype=torch.float32) for f in features]
        lengths = torch.tensor([t.shape[0] for t in feats], dtype=torch.long)
        t_max = int(lengths.max().item())
        bsz = len(feats)
        padded = torch.zeros((bsz, t_max, self.n_mels), dtype=torch.float32)
        attn = torch.zeros((bsz, t_max), dtype=torch.long)
        for i, t in enumerate(feats):
            padded[i, : t.shape[0]] = t
            attn[i, : t.shape[0]] = 1
        batch["input_features"] = padded
        batch["attention_mask"] = attn

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
        batch["decoder_input_ids"] = self._shift_right(labels)
        return batch

    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing inputs: prepend ``decoder_start_token_id`` and shift
        right, replacing -100 (ignore index) with ``pad_token_id`` so the
        embedding lookup is valid at those positions."""
        pad_id = self.tokenizer.pad_token_id
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = self.decoder_start_token_id
        return shifted.masked_fill(shifted == -100, pad_id)
