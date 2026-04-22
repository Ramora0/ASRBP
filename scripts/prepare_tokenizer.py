"""Train the SentencePiece-BPE tokenizer from LibriSpeech transcripts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/prepare_tokenizer.py` without install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import concatenate_datasets, load_dataset  # noqa: E402

from conformer_asr.config import load_config  # noqa: E402
from conformer_asr.tokenizer import iter_transcripts, train_tokenizer  # noqa: E402


_TRAIN_SUBSETS = {
    "clean100": ["train.clean.100"],
    "clean460": ["train.clean.100", "train.clean.360"],
    "all960": ["train.clean.100", "train.clean.360", "train.other.500"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/conformer_small.yaml")
    p.add_argument("--subset", choices=list(_TRAIN_SUBSETS), default=None)
    p.add_argument("--vocab_size", type=int, default=1000)
    p.add_argument("--output_dir", default=None, help="overrides data.tokenizer_dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    subset = args.subset or cfg.data.subset
    out_dir = args.output_dir or cfg.data.tokenizer_dir

    splits = _TRAIN_SUBSETS[subset]
    print(f"Loading transcripts from {splits} …")
    parts = [
        load_dataset(cfg.data.dataset_id, split=s, trust_remote_code=True)
        for s in splits
    ]
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    ds = ds.remove_columns([c for c in ds.column_names if c != "text"])

    print(f"Training BPE (vocab_size={args.vocab_size}) from {len(ds)} transcripts → {out_dir}")
    tokenizer = train_tokenizer(iter_transcripts(ds), out_dir=out_dir, vocab_size=args.vocab_size)
    print(f"Done. Tokenizer has {len(tokenizer)} tokens. Example:")
    sample = "the quick brown fox jumps over the lazy dog"
    ids = tokenizer(sample).input_ids
    print(f"  '{sample}' → {ids}")
    print(f"  decoded: {tokenizer.decode(ids)!r}")


if __name__ == "__main__":
    main()
