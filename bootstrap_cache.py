"""Cache-directory bootstrap — MUST run before any HuggingFace import.

``datasets`` and ``transformers`` read ``HF_HOME`` / ``HF_DATASETS_CACHE`` /
``TRANSFORMERS_CACHE`` at *import time* and cache the resulting paths as module
globals. Setting those env vars later (e.g. inside ``main()``) is silently
ignored by the in-process library state. To actually redirect caches we must
set the env vars *before* the first ``import datasets`` / ``import transformers``.

This module lives at the project root (not inside ``conformer_asr``) because
importing any ``conformer_asr`` submodule runs the package ``__init__``, which
eagerly imports ``datasets`` and ``transformers`` — exactly what we're trying
to avoid. Each entrypoint should do::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from bootstrap_cache import bootstrap_cache_from_argv
    bootstrap_cache_from_argv()                 # ← before any HF import

    # Now it is safe to import datasets / transformers / conformer_asr.*
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def set_hf_cache_env(cache_dir: str | os.PathLike) -> str:
    """Point every HF cache env var at ``cache_dir`` and create the directories.

    Uses unconditional assignment so a config-specified scratch path wins over
    any stale ``HF_HOME`` inherited from the environment. Users who want to
    override from the shell can set ``HF_HOME_OVERRIDE`` instead — it wins.
    """
    cache_dir = str(cache_dir)
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "hub").mkdir(parents=True, exist_ok=True)
    (root / "transformers").mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(root / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(root / "transformers")
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)

    # librosa uses numba; numba writes JIT cache next to its source files by
    # default — i.e. under site-packages in $HOME. On NFS with many parallel
    # workers (num_proc) that produces `OSError: [Errno 116] Stale file handle`
    # when workers race on the same .nbi file. Point numba at scratch too.
    numba_cache = root / "numba"
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ["NUMBA_CACHE_DIR"] = str(numba_cache)
    # matplotlib has the same NFS-contention problem on fresh nodes.
    mpl_cache = root / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    return cache_dir


def _read_cache_dir_from_yaml(config_path: str | os.PathLike) -> str | None:
    config_path = Path(config_path)
    if not config_path.exists():
        return None
    try:
        import yaml
    except ImportError:
        return None
    with open(config_path) as fh:
        raw = yaml.safe_load(fh) or {}
    return (raw.get("data") or {}).get("cache_dir")


def _peek_flag(argv: list[str], name: str) -> str | None:
    """Extract ``--flag value`` / ``--flag=value`` from argv without argparse.

    We can't use argparse here because the full parser lives in modules that
    import HuggingFace libraries — and pulling those in is exactly what we're
    trying to avoid until after env vars are set.
    """
    for i, tok in enumerate(argv):
        if tok == name and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(f"{name}="):
            return tok.split("=", 1)[1]
    return None


def bootstrap_cache_from_argv(
    argv: list[str] | None = None,
    default_config: str = "configs/cnns/c4x.yaml",
) -> str | None:
    """Resolve a cache dir from CLI / env / YAML and apply it.

    Priority: ``HF_HOME_OVERRIDE`` env var > ``--cache_dir`` flag > YAML ``data.cache_dir``.
    Returns the chosen cache_dir (or ``None`` if nothing was resolved).
    """
    argv = list(sys.argv if argv is None else argv)

    env_override = os.environ.get("HF_HOME_OVERRIDE")
    if env_override:
        return set_hf_cache_env(env_override)

    cli_override = _peek_flag(argv, "--cache_dir")
    if cli_override:
        return set_hf_cache_env(cli_override)

    config_path = _peek_flag(argv, "--config") or default_config
    cache_dir = _read_cache_dir_from_yaml(config_path)
    if cache_dir:
        return set_hf_cache_env(cache_dir)
    return None
