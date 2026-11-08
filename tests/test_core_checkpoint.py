"""Naming a checkpoint, and identifying it by what it holds, not where it sits."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402

from audiolab.core.checkpoint import Checkpoint, CheckpointError, file_sha256  # noqa: E402


def write(path: Path, content: bytes = b"weights") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_a_nemo_path_and_a_repository_id_are_told_apart(tmp_path):
    assert Checkpoint.from_reference("nvidia/parakeet-tdt-0.6b-v3").hf_id
    assert Checkpoint.from_reference("nvidia/parakeet-tdt-0.6b-v3").path == ""
    assert Checkpoint.from_reference("/data/models/run.nemo").path == "/data/models/run.nemo"
    assert Checkpoint.from_reference("./run.nemo").path == "run.nemo"
    assert Checkpoint.from_reference("~/runs/train/a/model.nemo").path.endswith(
        "/runs/train/a/model.nemo"
    )
    with pytest.raises(CheckpointError, match="neither a .nemo path nor"):
        Checkpoint.from_reference("parakeet")
    with pytest.raises(CheckpointError, match="no checkpoint is named"):
        Checkpoint.from_reference("  ")


def test_the_identity_of_a_file_is_its_content(tmp_path):
    original = write(tmp_path / "a.nemo")
    copy = write(tmp_path / "sub" / "b.nemo", original.read_bytes())
    assert Checkpoint(path=str(original)).identity() == Checkpoint(path=str(copy)).identity()
    assert Checkpoint(path=str(original)).identity() == f"sha256={file_sha256(original)}"

    write(original, b"other weights")
    assert Checkpoint(path=str(original)).identity() != Checkpoint(path=str(copy)).identity()


def test_a_path_never_appears_in_an_identity(tmp_path):
    """ "A run record must be shareable, so a machine's paths stay out of it."""
    path = write(tmp_path / "model.nemo")
    identity = Checkpoint(path=str(path)).identity()
    assert str(tmp_path) not in identity
    assert json.dumps(identity).find("nemo") == -1


def test_a_repository_id_is_identified_as_it_is(tmp_path):
    checkpoint = Checkpoint(hf_id="nvidia/parakeet-tdt-0.6b-v3")
    assert checkpoint.identity() == "hf=nvidia/parakeet-tdt-0.6b-v3"
    assert str(checkpoint) == "nvidia/parakeet-tdt-0.6b-v3"


def test_a_missing_checkpoint_is_reported_before_anything_is_loaded():
    with pytest.raises(CheckpointError, match="does not exist"):
        Checkpoint(path="/nope/missing.nemo").identity()


def test_a_rewritten_file_is_hashed_again(tmp_path):
    """A resume slot is overwritten under the same name, at the same size.

    The hash is remembered per process to avoid reading gigabytes twice, so what
    it is remembered *by* has to include the file's own timestamps.
    """
    path = write(tmp_path / "slot.nemo", b"first")
    before = Checkpoint(path=str(path)).identity()
    stamp = path.stat().st_mtime_ns
    write(path, b"other")
    os.utime(path, ns=(stamp + 1_000_000_000, stamp + 1_000_000_000))
    assert Checkpoint(path=str(path)).identity() != before
