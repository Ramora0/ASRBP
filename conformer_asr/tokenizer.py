"""Tokenizer: SpeechBrain's pretrained SentencePiece (5K unigram) over LibriSpeech.

We consume SB's published ``tokenizer.ckpt`` from HF Hub rather than training
our own BPE, which (a) removes a pipeline step and hosted artifact, and (b)
keeps vocab-aligned with ``speechbrain/asr-transformer-transformerlm-librispeech``
so we can warm-start the decoder and do shallow fusion on future retrains.

The vocab was trained on uppercase LibriSpeech transcripts; ``normalize_text``
no longer lowercases.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Sequence

import torch

SB_TOKENIZER_REPO = "speechbrain/asr-transformer-transformerlm-librispeech"
SB_TOKENIZER_FILE = "tokenizer.ckpt"
SB_PAD_ID = 0
SB_BOS_ID = 1
SB_EOS_ID = 2
SB_UNK_ID = 3

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Whitespace-normalize a transcript. Preserves case — LibriSpeech is
    natively uppercase and so is SB's SentencePiece vocab."""
    return _WHITESPACE_RE.sub(" ", text.strip())


class SpeechBrainTokenizer:
    """Adapter around SB's pretrained SentencePiece over the subset of the
    ``PreTrainedTokenizerFast`` API this codebase consumes.

    Intentionally not a subclass of HF ``PreTrainedTokenizerFast`` — avoiding
    that inheritance spares us from having to satisfy its (considerable)
    internal contract around ``added_tokens_encoder``, ``slow_tokenizer_class``,
    ``save_vocabulary``, etc. Duck-typed API surface is enough here.
    """

    pad_token: str = "<pad>"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    unk_token: str = "<unk>"
    pad_token_id: int = SB_PAD_ID
    bos_token_id: int = SB_BOS_ID
    eos_token_id: int = SB_EOS_ID
    unk_token_id: int = SB_UNK_ID
    model_input_names: list[str] = ["input_ids", "attention_mask"]

    def __init__(self, sp_model_path: str | Path):
        import sentencepiece as spm

        self._sp_model_path = Path(sp_model_path)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self._sp_model_path))

    def __len__(self) -> int:
        return self.sp.get_piece_size()

    def __call__(self, text: str):
        # BOS/EOS framing mirrors the old TemplateProcessing post-processor so
        # the collator's leading-BOS strip (see DataCollatorSpeechSeq2SeqWithPadding)
        # keeps working unchanged. Return a BatchEncoding so callers can use
        # either `.input_ids` (HF attribute style) or `["input_ids"]`.
        from transformers import BatchEncoding

        ids = [self.bos_token_id] + self.sp.encode(text, out_type=int) + [self.eos_token_id]
        return BatchEncoding({"input_ids": ids})

    def _strip_special(self, ids: Sequence[int]) -> list[int]:
        specials = {self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id}
        out: list[int] = []
        for i in ids:
            i = int(i)
            if i < 0:
                continue
            if i in specials:
                continue
            out.append(i)
        return out

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        else:
            ids = list(ids)
        if skip_special_tokens:
            ids = self._strip_special(ids)
        return self.sp.decode(ids)

    def batch_decode(self, batch: Any, skip_special_tokens: bool = True) -> list[str]:
        if hasattr(batch, "tolist"):
            batch = batch.tolist()
        return [self.decode(row, skip_special_tokens=skip_special_tokens) for row in batch]

    def pad(self, features: list[dict[str, Any]], padding: bool | str = True, return_tensors: str | None = None):
        from transformers import BatchEncoding

        ids_list = [list(f["input_ids"]) for f in features]
        max_len = max(len(x) for x in ids_list)
        bsz = len(ids_list)
        if return_tensors == "pt":
            input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
            attn = torch.zeros((bsz, max_len), dtype=torch.long)
            for i, x in enumerate(ids_list):
                input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
                attn[i, : len(x)] = 1
        else:
            input_ids = [x + [self.pad_token_id] * (max_len - len(x)) for x in ids_list]
            attn = [[1] * len(x) + [0] * (max_len - len(x)) for x in ids_list]
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attn})

    def get_vocab(self) -> dict[str, int]:
        return {self.sp.id_to_piece(i): i for i in range(len(self))}

    def save_pretrained(self, path: str | Path) -> None:
        """Copy the SP model + a pointer manifest into ``path`` so a checkpoint
        directory is self-contained."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(self._sp_model_path), str(path / "sentencepiece.model"))
        with open(path / "tokenizer_info.json", "w") as f:
            json.dump(
                {
                    "source": "speechbrain",
                    "repo": SB_TOKENIZER_REPO,
                    "file": SB_TOKENIZER_FILE,
                    "vocab_size": len(self),
                    "pad_id": self.pad_token_id,
                    "bos_id": self.bos_token_id,
                    "eos_id": self.eos_token_id,
                    "unk_id": self.unk_token_id,
                },
                f,
                indent=2,
            )


def _download_sb_tokenizer(cache_dir: str | None, repo: str = SB_TOKENIZER_REPO) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=SB_TOKENIZER_FILE,
            cache_dir=cache_dir,
        )
    )


def load_tokenizer(
    path: str | Path | None = None,
    *,
    cache_dir: str | None = None,
    repo: str = SB_TOKENIZER_REPO,
) -> SpeechBrainTokenizer:
    """Load SB's pretrained SentencePiece tokenizer.

    - If ``path`` is given and contains ``sentencepiece.model`` (written by a
      prior training run's ``save_pretrained``), load from there.
    - Otherwise, download ``tokenizer.ckpt`` from ``repo`` (cached under
      ``cache_dir`` if provided, else HF's default).
    """
    if path is not None:
        local_sp = Path(path) / "sentencepiece.model"
        if local_sp.exists():
            return SpeechBrainTokenizer(local_sp)
        print(
            f"[tokenizer] no sentencepiece.model in {path}; falling back to HF repo {repo}"
        )
    sp_path = _download_sb_tokenizer(cache_dir=cache_dir, repo=repo)
    return SpeechBrainTokenizer(sp_path)
