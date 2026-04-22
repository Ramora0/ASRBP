from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
from transformers import PreTrainedTokenizerFast


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip().lower())


def train_tokenizer(
    corpus: Iterable[str],
    out_dir: str | Path,
    vocab_size: int = 1000,
) -> PreTrainedTokenizerFast:
    """Train a byte-level BPE on LibriSpeech transcripts and save to ``out_dir``.

    Saves files compatible with ``PreTrainedTokenizerFast.from_pretrained``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator((normalize_text(t) for t in corpus), trainer=trainer)

    bos_id = tokenizer.token_to_id(BOS_TOKEN)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
        special_tokens=[(BOS_TOKEN, bos_id), (EOS_TOKEN, eos_id)],
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token=PAD_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=UNK_TOKEN,
    )
    fast.save_pretrained(str(out_dir))
    return fast


def load_tokenizer(path: str | Path) -> PreTrainedTokenizerFast:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {path}. Run scripts/prepare_tokenizer.py first."
        )
    return PreTrainedTokenizerFast.from_pretrained(str(path))


def iter_transcripts(dataset) -> Iterable[str]:
    """Yield raw transcripts from a HF dataset that has a 'text' column."""
    for example in dataset:
        text = example.get("text")
        if text:
            yield text
