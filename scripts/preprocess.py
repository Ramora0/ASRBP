"""Offline LibriSpeech preprocessing — run this on a fat CPU node.

``conformer_asr.data.preprocess_dataset`` already caches its output to
``<cache_dir>/preprocessed/<key>`` keyed on the subset + mel params +
tokenizer. This script is a thin CLI around it so the preprocessing job
can be submitted to a data-prep node (e.g. 48 cores, no GPU) separately
from the GPU training run. Once it finishes, ``scripts/train.py`` will
hit the cache path and skip straight to training.

Example::

    uv run python scripts/preprocess.py \\
        --subset all960 \\
        --num_proc 48 \\
        --cache_dir /fs/scratch/PAS2836/lees_stuff/hf_cache
"""
from __future__ import annotations

import os

# Each worker should be single-threaded — we already parallelize via num_proc.
# Without this, torch / numpy / MKL / OpenBLAS / tokenizers each default their
# own thread pool to the full core count, so with num_proc=48 on a 48-core box
# you get 48 × 48 ≈ 2300 threads fighting for 48 cores. That presents as
# single-digit CPU% per worker (everyone blocked in the scheduler) instead of
# near-100%. Must be set before the first import of torch / numpy / datasets /
# tokenizers — they read these at import time.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Cache env vars must be set before the first HF import — see bootstrap_cache.py.
from bootstrap_cache import bootstrap_cache_from_argv  # noqa: E402

_resolved_cache = bootstrap_cache_from_argv()
print(f"HF cache_dir (bootstrapped): {_resolved_cache}")

from conformer_asr.config import load_config  # noqa: E402
from conformer_asr.data import (  # noqa: E402
    _preprocess_cache_dir,
    _preprocess_cache_key,
    load_librispeech,
    preprocess_dataset,
    setup_cache_dir,
)
from conformer_asr.tokenizer import load_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/conformer_small.yaml")
    p.add_argument("--subset", choices=["clean100", "clean460", "all960"])
    p.add_argument("--num_proc", type=int, help="Override data.num_proc")
    p.add_argument("--cache_dir", help="Override data.cache_dir")
    p.add_argument("--tokenizer_dir")
    return p.parse_args()


def _flatten_overrides(args: argparse.Namespace) -> dict:
    return {k: v for k, v in vars(args).items() if k != "config" and v is not None}


def main() -> None:
    # Belt-and-braces: even if some library ignored OMP_NUM_THREADS, force
    # torch's intraop thread pool to 1. (set_num_interop_threads raises if the
    # pool is already initialized, so we try it but don't hard-fail.)
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    args = parse_args()
    cfg = load_config(args.config, overrides=_flatten_overrides(args))
    setup_cache_dir(cfg.data.cache_dir)

    key = _preprocess_cache_key(cfg.model, load_tokenizer(cfg.data.tokenizer_dir), cfg.data)
    save_dir = _preprocess_cache_dir(cfg.data, key)
    print(f"[preprocess] subset={cfg.data.subset}  num_proc={cfg.data.num_proc}")
    print(f"[preprocess] mel: n_mels={cfg.model.n_mels} n_fft={cfg.model.n_fft} hop={cfg.model.hop_length}")
    print(f"[preprocess] cache_dir:  {cfg.data.cache_dir}")
    print(f"[preprocess] output dir: {save_dir}")
    if save_dir is not None and save_dir.exists():
        print("[preprocess] cache already populated — nothing to do.")
        return

    tokenizer = load_tokenizer(cfg.data.tokenizer_dir)

    t0 = time.perf_counter()
    print("[preprocess] loading LibriSpeech …")
    ds = load_librispeech(cfg.data)
    for split, d in ds.items():
        print(f"  raw {split}: {len(d)} examples")

    print("[preprocess] extracting features + tokenizing …")
    ds = preprocess_dataset(ds, cfg.model, tokenizer, cfg.data)

    elapsed = time.perf_counter() - t0
    print(f"[preprocess] done in {elapsed / 60:.1f} min")
    for split, d in ds.items():
        print(f"  preprocessed {split}: {len(d)} examples")
    print(f"[preprocess] saved to {save_dir} — scripts/train.py will pick this up via cache key.")


if __name__ == "__main__":
    main()
