"""Registry of autoregressive decoders with cross-attention.

Add a new decoder by:
  1. Dropping a module in this package that exposes a factory matching the
     signature below (takes ``mcfg`` + tokenizer ids, returns a
     ``*ForCausalLM``-style module).
  2. Registering it in ``DECODERS`` below under a string key.
  3. Referencing that key from ``model.decoder_type`` in the YAML config.

``build_decoder`` dispatches on ``mcfg.decoder_type``. Each builder owns its
own config-to-module translation so different decoder families can read
different subsets of ``ModelConfig``.
"""
from __future__ import annotations

from typing import Callable

import torch.nn as nn

from ..config import ModelConfig
from .bart import _CompatBartForCausalLM, build_bart_decoder


DECODERS: dict[str, Callable[..., nn.Module]] = {
    "bart": build_bart_decoder,
}


def build_decoder(
    mcfg: ModelConfig,
    *,
    vocab_size: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> nn.Module:
    """Instantiate the decoder named by ``mcfg.decoder_type``."""
    if mcfg.decoder_type not in DECODERS:
        known = ", ".join(sorted(DECODERS))
        raise ValueError(f"Unknown decoder_type {mcfg.decoder_type!r}; known: {known}")
    return DECODERS[mcfg.decoder_type](
        mcfg,
        vocab_size=vocab_size,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
    )


__all__ = ["DECODERS", "build_decoder", "_CompatBartForCausalLM"]
