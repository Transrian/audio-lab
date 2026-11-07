"""Where data and run outputs live.

Datasets, audio and run outputs are large and are never committed (D-013), so
every path that holds them comes from an environment variable with a
repository-local default. The defaults are git-ignored, so a fresh clone stays
clean even if someone forgets to set them.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIR = "data"
DEFAULT_RUNS_DIR = "runs"
DEFAULT_CACHE_DIR = ".cache/audiolab"


def _from_env(variable: str, default: str) -> Path:
    return Path(os.environ.get(variable) or default).expanduser()


def data_dir() -> Path:
    """Root of the prepared datasets (``$AUDIOLAB_DATA``, default ``./data``)."""
    return _from_env("AUDIOLAB_DATA", DEFAULT_DATA_DIR)


def runs_dir() -> Path:
    """Root of the evaluation runs (``$AUDIOLAB_RUNS``, default ``./runs``)."""
    return _from_env("AUDIOLAB_RUNS", DEFAULT_RUNS_DIR)


def cache_dir() -> Path:
    """Where downloads are kept before they are prepared (``$AUDIOLAB_CACHE``)."""
    return _from_env("AUDIOLAB_CACHE", DEFAULT_CACHE_DIR)
