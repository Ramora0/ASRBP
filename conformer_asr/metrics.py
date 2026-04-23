from __future__ import annotations

from typing import Callable

import jiwer
import numpy as np
import torch
from transformers import PreTrainedTokenizerFast


def compute_wer(predictions: list[str], references: list[str]) -> float:
    """WER via jiwer directly — avoids ``evaluate.load("wer")`` which tries to
    download a metric script from HF Hub at runtime (fails on offline cluster
    nodes).
    """
    pairs = [(p, r) for p, r in zip(predictions, references) if r.strip()]
    if not pairs:
        return 1.0
    preds, refs = zip(*pairs)
    return float(jiwer.wer(list(refs), list(preds)))


def build_compute_metrics(tokenizer: PreTrainedTokenizerFast) -> Callable:
    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # -100 appears on padded label positions, AND under DDP it also appears
        # inside ``pred_ids`` when ``pad_across_processes`` pads shorter
        # generations. Restore pad before decoding — the tokenizer's Rust
        # backend casts ids to unsigned int and overflows on -100.
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)
        pred_ids = np.where(pred_ids == -100, tokenizer.pad_token_id, pred_ids)

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        return {"wer": compute_wer(pred_str, label_str)}

    return compute_metrics


def ctc_greedy_decode(
    logits: torch.Tensor,
    input_lengths: torch.Tensor | None = None,
    blank: int = 0,
) -> list[list[int]]:
    """Greedy CTC decode: argmax per frame, collapse consecutive duplicates,
    strip blanks. Returns a list of variable-length token-id lists.

    ``logits`` shape ``(B, T, V)``. ``input_lengths`` is an optional per-sample
    valid-frame count; frames beyond it are ignored (SpecAugment / padding).
    The collapse-then-strip order matters: two separate emissions of the same
    token are represented in CTC as ``token … blank … token`` (or
    ``token … token`` in the trivial no-gap case), so we first merge repeats
    within runs and only then drop blanks.
    """
    argmax = logits.argmax(dim=-1)  # (B, T)
    argmax_list = argmax.detach().cpu().tolist()
    if input_lengths is not None:
        lengths = input_lengths.detach().cpu().tolist()
    else:
        lengths = [argmax.size(1)] * argmax.size(0)

    decoded: list[list[int]] = []
    for row, length in zip(argmax_list, lengths):
        seq = row[:length]
        collapsed: list[int] = []
        prev = -1
        for t in seq:
            if t != prev:
                if t != blank:
                    collapsed.append(int(t))
                prev = t
        decoded.append(collapsed)
    return decoded


def build_predictions_table(
    tokenizer: PreTrainedTokenizerFast,
    preds: list[list[int]],
    refs: list[list[int]],
    max_rows: int = 64,
):
    """Build a wandb.Table of ref/prediction pairs. Returns None if wandb is missing."""
    try:
        import wandb
    except ImportError:
        return None

    table = wandb.Table(columns=["reference", "prediction"])
    for p_ids, r_ids in list(zip(preds, refs))[:max_rows]:
        table.add_data(
            tokenizer.decode(r_ids, skip_special_tokens=True),
            tokenizer.decode(p_ids, skip_special_tokens=True),
        )
    return table
