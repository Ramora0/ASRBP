"""Pre-fetch every LibriSpeech split the training + evaluation pipeline touches.

Run this once (on a login node with internet access) before scheduling a GPU job:

    python scripts/download_librispeech.py

It writes arrow shards + audio files into ``data.cache_dir`` (default scratch).
After this completes, ``train.py`` and ``evaluate.py`` will hit the cache instead
of re-downloading, and GPU nodes without outbound network can still train.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- HF cache bootstrap: MUST run before any HF / conformer_asr import. ---
from bootstrap_cache import bootstrap_cache_from_argv  # noqa: E402

_resolved_cache = bootstrap_cache_from_argv()
print(f"HF cache_dir (bootstrapped): {_resolved_cache}")
# -------------------------------------------------------------------------

from datasets import load_dataset  # noqa: E402

from conformer_asr.config import load_config  # noqa: E402
from conformer_asr.data import setup_cache_dir  # noqa: E402


# The full set of splits touched by train + eval, regardless of --subset choice.
ALL_SPLITS = [
    "train.clean.100",
    "train.clean.360",
    "train.other.500",
    "validation.clean",
    "validation.other",
    "test.clean",
    "test.other",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/conformer_c4x.yaml")
    p.add_argument("--cache_dir", default=None, help="overrides data.cache_dir")
    p.add_argument(
        "--splits",
        nargs="+",
        default=ALL_SPLITS,
        help=f"splits to fetch (default: all seven: {ALL_SPLITS})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cache_dir = args.cache_dir or cfg.data.cache_dir
    setup_cache_dir(cache_dir)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"Downloading LibriSpeech ({cfg.data.dataset_id}) → {cache_dir}")

    for split in args.splits:
        print(f"  • {split}")
        ds = load_dataset(
            cfg.data.dataset_id,
            split=split,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        print(f"    loaded {len(ds)} examples")

    print("Done. GPU-side runs will now hit the local cache.")


if __name__ == "__main__":
    main()
