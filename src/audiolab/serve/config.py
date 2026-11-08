"""How a local checkpoint is served, as written in a model config (D-016).

A model config can carry a ``serve:`` block instead of an endpoint: ``audiolab
serve`` exposes the checkpoint through the same API a remote server speaks, and
``audiolab eval`` starts that server itself. The block is the only place a
checkpoint is named, and the path it names never leaves the machine that holds
the file: the repository keeps the shape of the block, not a path (see
[training.md](../../../docs/stt/training.md)).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

from audiolab.core.checkpoint import Checkpoint, CheckpointError

# Backends this version can serve. Supporting a model family means adding one
# name here and one module next to this file; that is the point of D-016.
BACKENDS = ("nemo",)
DEFAULT_GPU = "0"
# Greedy decoding stops after this many symbols without a blank. The value is
# recorded, because it is one of the things that decides an answer.
DEFAULT_MAX_SYMBOLS = 10
SERVE_FIELDS = ("backend", "checkpoint", "checkpoint_env", "gpu", "max_symbols")


class ServeConfigError(ValueError):
    """A ``serve:`` block that cannot be used as written."""


@dataclass(frozen=True)
class ServeSpec:
    """The ``serve:`` block: which checkpoint to expose, and how."""

    backend: str = "nemo"
    # A local ``.nemo`` file, or a Hugging Face repository id. A config meant to
    # work on another machine names a public model here and sets
    # ``checkpoint_env``; the machine that holds a local copy sets that variable
    # and serves the copy instead.
    checkpoint: str = ""
    checkpoint_env: str = ""
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
            gpu=str(raw.get("gpu", DEFAULT_GPU)),
            max_symbols=int(raw.get("max_symbols", DEFAULT_MAX_SYMBOLS)),
        )
        if not (spec.checkpoint or spec.checkpoint_env):
            raise ServeConfigError(
                f"{source}: 'serve' needs a 'checkpoint' (a .nemo path or a Hub id), "
                "or a 'checkpoint_env' naming the variable that holds one"
            )
        if spec.max_symbols < 1:
            raise ServeConfigError(f"{source}: serve.max_symbols must be at least 1")
        return spec

    def with_gpu(self, gpu: str | None) -> ServeSpec:
        """The same checkpoint on another GPU; ``None`` keeps the config's choice."""
        return self if not gpu else replace(self, gpu=str(gpu))

    def resolved(self) -> Checkpoint:
        """The weights to load, after the override a committed config cannot hold.

        The variable named by ``checkpoint_env`` wins when it is set, which is
        what lets a committed file name a public model while this machine serves
        a local copy of it.
        """
        reference = (os.environ.get(self.checkpoint_env, "") if self.checkpoint_env else "").strip()
        try:
            return Checkpoint.from_reference(reference or self.checkpoint)
        except CheckpointError as failure:
            raise ServeConfigError(f"serve: {failure}") from failure

    def identity(self) -> dict:
        """What this checkpoint is, for a cache key or a run record.

        The *content* of the checkpoint is here, never its path: a run directory
        stays shareable, and two machines holding the same file agree on the
        identity of the model they scored.
        """
        try:
            checkpoint = self.resolved().identity()
        except CheckpointError as failure:
            raise ServeConfigError(f"serve: {failure}") from failure
        return {
            "backend": self.backend,
            "checkpoint": checkpoint,
            "decoding": {"strategy": "greedy", "max_symbols": self.max_symbols},
        }
