"""Where a model comes from, and what makes it that model.

A checkpoint is either a local ``.nemo`` file or a Hugging Face repository id,
because those are the two ways NeMo is handed a model. Everything that decides
*which* weights are used (a base model for training, a fine-tune to serve) is
identified by the content of those weights, never by the path they sit at: a
copy is the same checkpoint, a rewrite is another one, and a path is a fact
about one machine, which must not travel in a run record
([D-013](../../../docs/owner/directives.md)).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# A reference is a path when it names a file, and a repository id otherwise. The
# rule is here, in one place, because both a training config and a model config
# name a checkpoint the same way.
PATH_SUFFIXES = (".nemo",)
PATH_PREFIXES = ("/", "./", "~", "../")
HUB_SEPARATOR = "/"


class CheckpointError(ValueError):
    """A checkpoint that cannot be identified or loaded as named."""


@dataclass(frozen=True)
class Checkpoint:
    """The weights of a model: a local file, or a repository id on the Hub."""

    path: str = ""
    hf_id: str = ""

    @classmethod
    def from_reference(cls, reference: str) -> Checkpoint:
        """Read ``a/path/model.nemo`` or ``owner/repository`` as what it is."""
        reference = reference.strip()
        if not reference:
            raise CheckpointError("no checkpoint is named")
        is_path = reference.endswith(PATH_SUFFIXES) or reference.startswith(PATH_PREFIXES)
        if is_path:
            return cls(path=str(Path(reference).expanduser()))
        if HUB_SEPARATOR not in reference:
            raise CheckpointError(
                f"{reference!r} is neither a .nemo path nor a 'owner/repository' id"
            )
        return cls(hf_id=reference)

    def __str__(self) -> str:
        return self.path or self.hf_id

    def identity(self) -> str:
        """What these weights are, as text that never names a local path.

        A file is identified by its content, so two copies of it are the same
        checkpoint and a rewritten file is a different one. A Hub id names the
        repository only: NeMo resolves it when it loads, at whatever the
        repository holds then, so a run that must be reproducible from its
        identity uses a local file.
        """
        if self.path:
            if not os.path.isfile(self.path):
                raise CheckpointError(
                    f"checkpoint {self.path} does not exist; fetch it, or set the "
                    "environment variable the config names for a local copy"
                )
            return f"sha256={file_sha256(self.path)}"
        return f"hf={self.hf_id}"


def file_sha256(path: str | Path) -> str:
    """Hash of a file's content, read once per (path, size, modification time).

    A checkpoint is gigabytes, and both a cache key and a run record ask for its
    hash, so the answer is kept for the process. Size and modification time are
    part of the memo key, so a file rewritten under the same name is hashed
    again.
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
