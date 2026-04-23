"""Registry of spectrogram → transformer-input downsamplers.

Add a new architecture by:
  1. Dropping a module in this package that subclasses ``Downsampler`` from
     ``.base``.
  2. Registering it in ``DOWNSAMPLERS`` below under a string key.
  3. Referencing that key from ``model.downsampler.type`` in the YAML config.

``build_downsampler`` maps ``(DownsamplerConfig, n_mels, hidden, dropout)``
through the registry. Downsampler-specific config fields go into
``DownsamplerConfig.kwargs``.
"""
from __future__ import annotations

from typing import Any, Callable

from ..config import DownsamplerConfig
from .base import Downsampler
from .conv2d import Conv2dDownsampler


DOWNSAMPLERS: dict[str, Callable[..., Downsampler]] = {
    "conv2d": Conv2dDownsampler,
}


def build_downsampler(
    cfg: DownsamplerConfig,
    *,
    n_mels: int,
    hidden: int,
    dropout: float,
) -> Downsampler:
    """Instantiate the downsampler named by ``cfg.type``.

    ``cfg.kwargs`` is forwarded as extra keyword arguments to the constructor,
    so each downsampler can define its own knobs without the registry having
    to know about them.
    """
    if cfg.type not in DOWNSAMPLERS:
        known = ", ".join(sorted(DOWNSAMPLERS))
        raise ValueError(f"Unknown downsampler type {cfg.type!r}; known: {known}")
    cls = DOWNSAMPLERS[cfg.type]
    kwargs: dict[str, Any] = dict(cfg.kwargs or {})
    return cls(n_mels=n_mels, hidden=hidden, dropout=dropout, **kwargs)


__all__ = ["Downsampler", "Conv2dDownsampler", "DOWNSAMPLERS", "build_downsampler"]
