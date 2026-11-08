"""How a local checkpoint is served, as written in a model config (D-016).

A model config can carry a ``serve:`` block instead of an endpoint: ``audiolab
serve`` exposes the checkpoint through the same API a remote server speaks, and
``audiolab eval`` starts that server itself. The block is the only place a
checkpoint is named, and the path it names never leaves the machine that holds
the file: the repository keeps the shape of the block, not a path (see
[training.md](../../../docs/stt/training.md)).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

# Backends this version can serve. Supporting a model family means adding one
# name here and one module next to this file; that is the point of D-016.
BACKENDS = ("nemo",)
DEFAULT_GPU = "0"
# Greedy decoding stops after this many symbols without a blank. The value is
# recorded, because it is one of the things that decides an answer.
DEFAULT_MAX_SYMBOLS = 10
SERVE_FIELDS = (
    "backend",
    "checkpoint",
    "checkpoint_env",
    "hf_id",
    "gpu",
    "max_symbols",
)


class ServeConfigError(ValueError):
    """A ``serve:`` block that cannot be used as written."""


@dataclass(frozen=True)
class Checkpoint:
    """The weights a server loads: a local file, or a repository id on the Hub."""

    path: str = ""
    hf_id: str = ""

    def identity(self) -> str:
        """What these weights are, as text that never names a local path.

        A file is identified by its content, so two copies of it are the same
        checkpoint and a rewritten file is a different one. A Hub id names the
        repository only: NeMo resolves it when it loads, at whatever the
        repository's default branch holds then, so a run that must be
        reproducible from its identity serves a local file (see
        [training.md](../../../docs/stt/training.md)).
        """
        if self.path:
            if not os.path.isfile(self.path):
                raise ServeConfigError(
                    f"serve: checkpoint {self.path} does not exist; "
                    "fetch it, or set the variable named by checkpoint_env"
                )
            return f"sha256={file_sha256(self.path)}"
        return f"hf={self.hf_id}"


@dataclass(frozen=True)
class ServeSpec:
    """The ``serve:`` block: which checkpoint to expose, and how."""

    backend: str = "nemo"
    # A local ``.nemo`` file. A config meant to work on another machine names
    # ``checkpoint_env`` instead, and the machine holding the file sets it.
    checkpoint: str = ""
    checkpoint_env: str = ""
    # A Hugging Face repository id, downloaded on first use. The base model of a
    # public fine-tune is named this way.
    hf_id: str = ""
    gpu: str = DEFAULT_GPU
    max_symbols: int = DEFAULT_MAX_SYMBOLS

    @classmethod
    def from_mapping(cls, raw: dict, source: str | Path = "the model config") -> ServeSpec:
        """Build a spec, refusing unknown keys and impossible combinations."""
        if not isinstance(raw, dict):
            raise ServeConfigError(f"{source}: 'serve' must be a mapping")
        unknown = sorted(set(raw) - set(SERVE_FIELDS))
        if unknown:
            raise ServeConfigError(f"{source}: unknown serve keys: {', '.join(unknown)}")
        backend = str(raw.get("backend") or "nemo")
        if backend not in BACKENDS:
            raise ServeConfigError(
                f"{source}: unknown serve backend {backend!r}; known: {', '.join(BACKENDS)}"
            )
        spec = cls(
            backend=backend,
            checkpoint=str(raw.get("checkpoint") or ""),
            checkpoint_env=str(raw.get("checkpoint_env") or ""),
            hf_id=str(raw.get("hf_id") or ""),
            gpu=str(raw.get("gpu", DEFAULT_GPU)),
            max_symbols=int(raw.get("max_symbols", DEFAULT_MAX_SYMBOLS)),
        )
        if not (spec.checkpoint or spec.checkpoint_env or spec.hf_id):
            raise ServeConfigError(
                f"{source}: 'serve' needs one of 'checkpoint', 'checkpoint_env' or 'hf_id'"
            )
        if spec.max_symbols < 1:
            raise ServeConfigError(f"{source}: serve.max_symbols must be at least 1")
        return spec

    def with_gpu(self, gpu: str | None) -> ServeSpec:
        """The same checkpoint on another GPU; ``None`` keeps the config's choice."""
        return self if not gpu else replace(self, gpu=str(gpu))

    def resolved(self) -> Checkpoint:
        """The weights to load, after the override a committed config cannot hold.

        Order: the variable named by ``checkpoint_env`` when it is set, then the
        path written in the config, then the Hub id. That is what lets a
        committed file name a public model while this machine serves a local
        copy of it.
        """
        override = (os.environ.get(self.checkpoint_env, "") if self.checkpoint_env else "").strip()
        path = override or self.checkpoint
        if path:
            return Checkpoint(path=str(Path(path).expanduser()))
        return Checkpoint(hf_id=self.hf_id)

    def identity(self) -> dict:
        """What this checkpoint is, for a cache key or a run record.

        The *content* of the checkpoint is here, never its path: a run directory
        stays shareable, and two machines holding the same file agree on the
        identity of the model they scored.
        """
        return {
            "backend": self.backend,
            "checkpoint": self.resolved().identity(),
            "decoding": {"strategy": "greedy", "max_symbols": self.max_symbols},
        }


def file_sha256(path: str | Path) -> str:
    """Hash of a file's content, read once per (path, size, modification time).

    A checkpoint is gigabytes, and both the cache key and the run record ask for
    its hash, so the answer is kept for the process. The size and the
    modification time are part of the memo key, so a file rewritten under the
    same name is hashed again.
    """
    stat = os.stat(path)
    return _file_sha256(str(path), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=8)
def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
