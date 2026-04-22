from __future__ import annotations

from typing import Callable

import evaluate
import numpy as np
from transformers import PreTrainedTokenizerFast


def build_compute_metrics(tokenizer: PreTrainedTokenizerFast) -> Callable:
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # -100 was used on padded label positions; restore pad so decoding works.
        label_ids = np.where(label_ids == -100, tokenizer.pad_token_id, label_ids)

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # Guard against empty references (jiwer errors on empty strings).
        pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not pairs:
            return {"wer": 1.0}
        preds, refs = zip(*pairs)
        wer = wer_metric.compute(predictions=list(preds), references=list(refs))
        return {"wer": float(wer)}

    return compute_metrics


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
