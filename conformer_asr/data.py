from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from scipy.signal import resample_poly
from transformers import PreTrainedTokenizerFast

from .config import DataConfig, ModelConfig
from .features import log_mel_spectrogram
from .tokenizer import normalize_text


_TRAIN_SUBSETS: dict[str, list[str]] = {
    "clean100": ["train.clean.100"],
    "clean460": ["train.clean.100", "train.clean.360"],
    "all960": ["train.clean.100", "train.clean.360", "train.other.500"],
}


def resolve_num_proc(n: int) -> int:
    """Resolve ``num_proc`` to an actual worker count.

    ``n <= 0`` means autodetect: prefer ``os.sched_getaffinity`` so we honor
    SLURM / cgroup pins (the host may have 128 cores but the job is allocated
    16), and fall back to ``os.cpu_count()`` on platforms without it (macOS).
    """
    if n and n > 0:
        return n
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 1


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


def _speed_ratio(speed: float) -> tuple[int, int]:
    """Return an ``(up, down)`` rational approximation of ``1 / speed``.

    ``scipy.signal.resample_poly`` takes integer up/down factors. For Kaldi-
    style speeds (0.9 / 1.1) the exact ratios are 10/9 and 10/11 — we cap the
    denominator at 100 for any other speed so we stay cheap for the common
    case but don't hard-code just the two canonical factors.
    """
    frac = Fraction(1.0 / float(speed)).limit_denominator(100)
    return frac.numerator, frac.denominator


def _speed_perturb(wav: np.ndarray, speed: float) -> np.ndarray:
    """Kaldi-style speed perturbation via polyphase resampling.

    A ``speed`` factor of 0.9 stretches the waveform to ~1.111× length (slower,
    lower-pitched); 1.1 compresses it to ~0.909× length (faster, higher-pitched).
    Pitch shifts with duration — this is the Ko et al. 2015 recipe, not a
    phase-vocoder time-stretch. ``speed == 1.0`` is a no-op and returns the
    input unchanged.
    """
    if abs(speed - 1.0) < 1e-9:
        return wav
    up, down = _speed_ratio(speed)
    return resample_poly(wav, up=up, down=down).astype(np.float32, copy=False)


class RandomSpeedVariantSampler(torch.utils.data.Sampler[int]):
    """Filter a base sampler so each epoch yields one of ``n_variants`` contiguous
    rows per original clip.

    ``preprocess_dataset`` lays speed variants out contiguously: row ``k * n + v``
    holds clip ``k`` at speed slot ``v`` for ``v`` in ``0..n-1``. We wrap the
    Trainer's normal sampler (e.g. ``LengthGroupedSampler``) so the length-sorted
    mega-batch shuffle still runs over all ``3N`` indices — we just drop 2 out of
    every 3 on the way out. Net effect: each epoch sees each clip exactly once
    at a uniformly-random speed (so epoch length = ``N``, not ``3N``), with
    bucket grouping preserved.

    Under DDP each rank filters its own shard independently, so a clip whose
    variants land on different ranks may appear 0, 1, or 2 times that epoch
    instead of exactly once — a minor statistical bias, not a correctness
    issue. Fine for single-GPU; revisit if you need strict DDP accounting.
    """

    def __init__(
        self,
        base: torch.utils.data.Sampler[int],
        n_variants: int,
        generator: torch.Generator | None = None,
    ) -> None:
        if n_variants < 1:
            raise ValueError(f"n_variants must be >= 1, got {n_variants}")
        base_len = len(base)
        if base_len % n_variants != 0:
            raise ValueError(
                f"base sampler length {base_len} is not a multiple of n_variants {n_variants}; "
                "the preprocessed cache layout assumes contiguous variants per clip."
            )
        self._base = base
        self._n_variants = int(n_variants)
        self._n_clips = base_len // n_variants
        self._generator = generator

    def __len__(self) -> int:
        return self._n_clips

    def __iter__(self):
        g = self._generator
        if g is None:
            g = torch.Generator()
            g.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        kept = torch.randint(0, self._n_variants, (self._n_clips,), generator=g).tolist()
        for idx in self._base:
            idx = int(idx)
            clip_id, variant = divmod(idx, self._n_variants)
            if variant == kept[clip_id]:
                yield idx


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
            "v": 3,  # bumped when adding speed perturbation to the train split
            "dataset_id": cfg.dataset_id,
            "subset": cfg.subset,
            "eval_split": cfg.eval_split,
            "test_split": cfg.test_split,
            "sampling_rate": cfg.sampling_rate,
            "max_audio_seconds": cfg.max_audio_seconds,
            # Sorted + rounded so [1.0, 0.9, 1.1] and [0.9, 1.0, 1.1] collide
            # on the same cache and tiny float noise can't fork the key.
            "speeds": sorted(round(float(s), 4) for s in cfg.speed_perturbations),
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
    num_proc = resolve_num_proc(cfg.num_proc)

    # Dedup + sort so cache key and map both see the same list even if the user
    # wrote the YAML as [1.1, 0.9, 1.0]. Eval/test always run at 1.0x regardless.
    train_speeds = sorted({round(float(s), 4) for s in cfg.speed_perturbations}) or [1.0]
    eval_speeds = [1.0]

    def is_valid_length(example):
        return len(example["audio"]["array"]) <= max_len

    # Only filter training clips by max length — we still want to score all eval audio.
    ds["train"] = ds["train"].filter(is_valid_length, num_proc=num_proc)

    def make_prepare(speeds: list[float]):
        def prepare_batched(batch):
            audios = batch["audio"]
            texts = batch["text"]
            mels: list[np.ndarray] = []
            lengths: list[int] = []
            labels: list[list[int]] = []
            for audio, text in zip(audios, texts):
                wav = np.asarray(audio["array"], dtype=np.float32)
                token_ids = tokenizer(normalize_text(text))["input_ids"]
                for speed in speeds:
                    wav_s = _speed_perturb(wav, speed) if speed != 1.0 else wav
                    mel = log_mel_spectrogram(
                        wav_s,
                        n_mels=n_mels,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        sampling_rate=sr,
                    )  # (T_mel, n_mels) torch.float32
                    # .numpy() is a zero-copy view; we skip .tolist() because materializing
                    # a nested Python list of ~160K floats per 20 s clip dominates walltime
                    # at 960 h scale. Arrow serializes the numpy array directly.
                    mels.append(mel.numpy())
                    lengths.append(int(mel.shape[0]))
                    # Transcript is unchanged across speeds — append the same token
                    # ids for each perturbed copy.
                    labels.append(token_ids)
            return {
                "input_features": mels,
                "input_length": lengths,
                "labels": labels,
            }

        return prepare_batched

    remove_cols = {
        split: [c for c in ds[split].column_names if c not in {"input_length"}]
        for split in ds
    }
    # Scale the map input batch by speed count so the per-worker in-flight
    # output (``batch_size * len(speeds)`` mel tensors at ~640 KB each) stays
    # roughly the same magnitude regardless of how many speed variants we emit.
    # Without this, num_proc=48 × 3-way perturb pushed ~15 GB of extra RAM into
    # worker memory just from in-flight output batches.
    OUTPUT_BATCH_TARGET = 256
    for split in ds:
        speeds = train_speeds if split == "train" else eval_speeds
        split_batch_size = max(1, OUTPUT_BATCH_TARGET // max(1, len(speeds)))
        ds[split] = ds[split].map(
            make_prepare(speeds),
            batched=True,
            batch_size=split_batch_size,
            remove_columns=remove_cols[split],
            num_proc=num_proc,
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
