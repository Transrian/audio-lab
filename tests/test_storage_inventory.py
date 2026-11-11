"""Storage inventory (plan 09): synthetic roots, read-only behavior, classes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "experiments/storage-inventory/inventory.py"
spec = importlib.util.spec_from_file_location("storage_inventory", SCRIPT)
inventory = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = inventory
spec.loader.exec_module(inventory)

ENV = ("AUDIOLAB_DATA", "AUDIOLAB_RUNS", "AUDIOLAB_CACHE")


def write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def row(text: str, label: str) -> str:
    """The table row of one entry, as a single string."""
    for line in text.splitlines():
        if line.startswith(f"| {label} |"):
            return line
    raise AssertionError(f"no row for {label}")


def snapshot(roots: list[Path]) -> dict[str, tuple[str, int]]:
    """(content sha, mtime ns) of every file under the roots: proof of read-only."""
    state: dict[str, tuple[str, int]] = {}
    for root in roots:
        for current, _dirs, files in os.walk(root):
            for name in files:
                path = Path(current) / name
                state[str(path)] = (
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    path.stat().st_mtime_ns,
                )
    return state


@pytest.fixture
def roots(tmp_path, monkeypatch):
    """A synthetic repository and the three storage roots it points at.

    The teardown asserts the snapshot: the inventory must not write, create,
    move or rename anything under the storage roots.
    """
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    repo = tmp_path / "repo"
    # The keep rule (plan 09, lines 16-28) computed from the repository.
    write(
        repo / "configs/panels/ds-fr-fast.txt",
        "# panel: ds-fr-fast\n# dataset: ds-fr (10 clips in 1 groups; test 10)\n",
    )
    write(repo / "configs/train/screen-x.yaml", "dataset: x-train\nclips: 10\n")
    write(repo / "configs/train/example.yaml", "dataset: ds-fr\nclips: 10\n")
    write(repo / "configs/models/base-model.yaml", "name: base-model\n")
    # The evaluation run the docs cite (keep rule) and the transcript cache.
    write(
        repo / "docs/stt/screening.md",
        "The uncleaned run is stored as `runs/eval/old-screen-2026-01-01/`.\n"
        "The ghost checkpoint's evaluation is kept as `runs/eval/screen-ghost/`.\n",
    )
    write(
        repo / "docs/stt/datasets.md",
        "Two archives hold what a screening trained on: `docs-only-train` is one\n"
        "of them, kept as the frozen input of its run.\n",
    )
    write(
        repo / "docs/stt/evaluation.md",
        "The cache: `$AUDIOLAB_RUNS/transcripts/<key>.jsonl`. A per-machine key in "
        "`<runs dir>/.identity-key` opens it.\n",
    )
    (repo / "experiments").mkdir()

    data = tmp_path / "data"
    write(data / "ds-fr/manifest.jsonl")  # panel + train dataset
    write(data / "x-train/manifest.jsonl")  # train dataset of screen-x.yaml
    write(data / "mls-fr-train/manifest.jsonl")  # babble rule
    write(data / "nasa-en-train.curly-2026-11-09/manifest.jsonl")  # plan 05 rule
    write(data / "lonely-train/manifest.jsonl")  # prepared, referenced nowhere
    write(data / "docs-only-train/manifest.jsonl")  # prepared, named in docs only
    write(
        data / "icsi-en-train.pre-plain-label-2026-11-11/manifest.jsonl"
    )  # archived preparation, kept by name
    (data / "mystery-dir").mkdir(parents=True)  # no manifest, unreferenced
    write(data / "fleurs-fr/manifest.jsonl")  # prepared, like its cache
    write(data / "granary-fr-train/manifest.jsonl")

    cache = tmp_path / "cache"
    write(cache / "peertube-fr-train/a.wav", "a" * 10)
    write(cache / "peertube-fr-train/a.stt-v-whisper-large-v3.300s.jsonl", "b" * 5)
    write(cache / "peertube-fr-train/b.stt-c-parakeet-tdt-0.6B-v3.clips.json", "c" * 2)
    write(cache / "peertube-fr-train/pages/ep.html", "d" * 3)
    write(cache / "old-talk-cache/a.stt-c-whisper-large-v3.300s.json", "e" * 4)
    write(cache / "old-talk-cache/b.stt-a-qwen3-asr-1.7B.jsonl", "f" * 6)
    write(cache / "fleurs-fr/archive.tar.gz")  # prepared + pinned: re-creatable
    write(cache / "granary-fr-train/shard.parquet")  # prepared, pinned via Hub map
    write(cache / "notsofar-en-train/archive.bin")  # pinned, not prepared here
    # Unreferenced cache holding one pre-rename label-model answer (old id).
    write(cache / "old-cache/a1.stt-a-qwen3-asr-1.7B.jsonl", "o" * 4)
    write(cache / "old-cache/a2.stt-c-whisper-large-v3.300s.json", "o" * 5)

    runs = tmp_path / "runs"
    write(runs / ".identity-key", "k" * 65)
    write(runs / "transcripts/a.jsonl", "g" * 8)
    write(runs / "label-logs/x.log", "h" * 4)
    write(runs / "train/screen-peertube-fr/model.nemo", "n" * 100)
    write(runs / "train/screen-nasa-en/model.nemo", "n" * 100)
    write(runs / "train/screen-x/model.nemo", "n" * 100)
    write(runs / "train/screen-ghost/model.nemo", "n" * 100)  # no recipe
    write(runs / "train/mystery-run/foo.txt", "m" * 2)  # not a screen-* run
    write(runs / "eval/old-screen-2026-01-01/full/records.jsonl", "r" * 9)
    write(runs / "eval/base-model/full/records.jsonl", "r" * 9)
    write(runs / "eval/screen-x/full/records.jsonl", "r" * 9)
    write(runs / "eval/screen-peertube-fr/full/records.jsonl", "r" * 9)
    # Quoted by the docs in prose without naming the directory (DOC_CITED_EVAL_RUNS).
    write(runs / "eval/diag-african-apostrophe/full/records.jsonl", "r" * 9)
    write(runs / "eval/ghost-eval/full/run.json", json.dumps({"model": {"name": "some-model"}}))
    write(runs / "eval/ghost-eval/full/records.jsonl", "r" * 9)

    argv = [
        "--data-dir",
        str(data),
        "--runs-dir",
        str(runs),
        "--cache-dir",
        str(cache),
        "--repo-root",
        str(repo),
    ]
    before = snapshot([data, runs, cache])
    yield argv, [data, runs, cache]
    assert snapshot([data, runs, cache]) == before, "the inventory wrote under the roots"


def test_help_needs_no_environment(monkeypatch, capsys):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as error:
        inventory.main(["--help"])
    assert error.value.code == 0
    assert "read-only" in capsys.readouterr().out


def test_missing_roots_fail_with_exit_2(monkeypatch, capsys):
    for name in ENV:
        monkeypatch.delenv(name, raising=False)
    assert inventory.main([]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert "error:" in err


def test_classifications(roots, capsys):
    argv, _paths = roots
    assert inventory.main(argv) == 0
    text = capsys.readouterr().out

    # Plan 09 keep rule, computed from the repository.
    assert "keep (rule above)" in row(text, "$AUDIOLAB_DATA/ds-fr")
    assert "frozen panel" in row(text, "$AUDIOLAB_DATA/ds-fr")
    assert "keep (rule above)" in row(text, "$AUDIOLAB_DATA/x-train")
    assert "configs/train dataset" in row(text, "$AUDIOLAB_DATA/x-train")
    assert "keep (rule above)" in row(text, "$AUDIOLAB_DATA/mls-fr-train")
    assert "babble" in row(text, "$AUDIOLAB_DATA/mls-fr-train")
    assert "keep (rule above)" in row(text, "$AUDIOLAB_DATA/nasa-en-train.curly-2026-11-09")
    assert "plan 05 Part C" in row(text, "$AUDIOLAB_DATA/nasa-en-train.curly-2026-11-09")
    assert "candidate: unreferenced" in row(text, "$AUDIOLAB_DATA/lonely-train")
    assert "unknown" in row(text, "$AUDIOLAB_DATA/mystery-dir")
    # A prepared dataset the docs alone name is referenced, so it is kept: the
    # data branch must check the references like every other entry.
    docs_only = row(text, "$AUDIOLAB_DATA/docs-only-train")
    assert "keep (referenced)" in docs_only
    assert "docs/stt/datasets.md" in docs_only
    # An archived preparation of the keep table is kept whatever the
    # references say, and the row says which screening it is the input of.
    archived = row(text, "$AUDIOLAB_DATA/icsi-en-train.pre-plain-label-2026-11-11")
    assert "keep (rule above)" in archived
    assert "frozen screening input of configs/train/screen-icsi-en-train.yaml" in archived
    assert "configs/train dataset:" not in archived

    # Talk cache: answers kept, audio a candidate, both sizes reported.
    talk = row(text, "$AUDIOLAB_CACHE/peertube-fr-train")
    assert "keep (rule above)" in talk
    # 10 bytes audio, 5 + 2 bytes answers, 3 bytes of cached pages.
    assert "10 B (candidate)" in talk and "7 B (keep)" in talk and "3 B" in talk
    assert "candidate: unreferenced" in row(text, "$AUDIOLAB_CACHE/old-talk-cache")
    assert "2 model-answer files" in row(text, "$AUDIOLAB_CACHE/old-talk-cache")
    assert "candidate: re-creatable" in row(text, "$AUDIOLAB_CACHE/fleurs-fr")
    assert "pinned in audiolab.data.fleurs" in row(text, "$AUDIOLAB_CACHE/fleurs-fr")
    assert "candidate: re-creatable" in row(text, "$AUDIOLAB_CACHE/granary-fr-train")
    assert "not prepared here" in row(text, "$AUDIOLAB_CACHE/notsofar-en-train")
    # An unreferenced cache holding a pre-rename label-model answer: the note
    # counts it and checks the siblings, so the owner sees what would be lost.
    old = row(text, "$AUDIOLAB_CACHE/old-cache")
    assert "candidate: unreferenced" in old
    assert "1 of its 2 files are stt-a-qwen3-asr-1.7B.*" in old
    assert "exist in no other cache directory" in old

    assert "keep (referenced)" in row(text, "$AUDIOLAB_RUNS/.identity-key")
    assert "docs/stt/evaluation.md" in row(text, "$AUDIOLAB_RUNS/.identity-key")
    assert "keep (rule above)" in row(text, "$AUDIOLAB_RUNS/transcripts")
    assert "candidate: unreferenced" in row(text, "$AUDIOLAB_RUNS/label-logs")
    assert "summary of the rows below" in row(text, "$AUDIOLAB_RUNS/train")

    # Plan 09 step 5: the checkpoints.
    control = row(text, "$AUDIOLAB_RUNS/train/screen-peertube-fr")
    assert "keep (rule above)" in control
    assert "control run of plan 02" in control
    assert "model.nemo 100 B" in control
    assert "must stay" in control
    assert "keep (rule above)" in row(text, "$AUDIOLAB_RUNS/train/screen-nasa-en")
    assert "candidate: re-creatable" in row(text, "$AUDIOLAB_RUNS/train/screen-x")
    assert "configs/train/screen-x.yaml" in row(text, "$AUDIOLAB_RUNS/train/screen-x")
    assert "candidate: unreferenced" in row(text, "$AUDIOLAB_RUNS/train/screen-ghost")
    assert "not retrainable" in row(text, "$AUDIOLAB_RUNS/train/screen-ghost")
    # The checkpoint's basename is referenced — as its evaluation run. The
    # row shows that reference and still follows plan 09 step 5.
    ghost = row(text, "$AUDIOLAB_RUNS/train/screen-ghost")
    assert "the same basename is also referenced by docs/stt/screening.md" in ghost
    assert "runs/eval/screen-ghost" in ghost
    assert "unknown" in row(text, "$AUDIOLAB_RUNS/train/mystery-run")

    # The evaluation runs.
    cited = row(text, "$AUDIOLAB_RUNS/eval/old-screen-2026-01-01")
    assert "keep (rule above)" in cited
    assert "docs/stt/screening.md" in cited
    assert "keep (referenced)" in row(text, "$AUDIOLAB_RUNS/eval/base-model")
    assert "configs/models/base-model.yaml" in row(text, "$AUDIOLAB_RUNS/eval/base-model")
    ghost = row(text, "$AUDIOLAB_RUNS/eval/ghost-eval")
    assert "candidate: unreferenced" in ghost
    assert "evaluates the some-model checkpoint (run.json)" in ghost
    # Quoted by the docs in prose without naming the directory: the keep rule
    # still applies (DOC_CITED_EVAL_RUNS). Remove the constant entry and this
    # row falls back to candidate: unreferenced, which these asserts catch.
    diag = row(text, "$AUDIOLAB_RUNS/eval/diag-african-apostrophe")
    assert "keep (rule above)" in diag
    assert "docs/stt/screening.md (line 164)" in diag
    assert "reproducible from this run only" in diag


def test_unreadable_path_is_reported_and_exits_3(roots, monkeypatch, capsys):
    argv, paths = roots
    blocked = paths[1] / "transcripts"
    real_scandir = os.scandir
    state = {"blocked": True}

    def flaky_scandir(path, *args, **kwargs):
        if state["blocked"] and Path(path) == blocked:
            raise PermissionError(13, "Permission denied", str(blocked))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", flaky_scandir)
    assert inventory.main(argv) == 3
    out = capsys.readouterr().out
    assert "## Warnings: paths that could not be read" in out
    assert str(blocked) in out
    # The affected row is measured as 0 B, not silently.
    assert "0 B" in row(out, "$AUDIOLAB_RUNS/transcripts")
    # Unblock before the fixture teardown snapshots the tree: os.walk swallows
    # a directory whose scandir raises, and the snapshot would miss its files.
    state["blocked"] = False


def test_totals_and_talk_table(roots, capsys):
    argv, _paths = roots
    assert inventory.main(argv) == 0
    text = capsys.readouterr().out

    # Every leaf counted once: 6 cache + 10 data + 3 runs leaves + 5 train + 6 eval.
    assert "| **all** | 30 |" in text
    assert "candidate: re-creatable" in text
    assert "## Talk caches: downloaded audio vs model answers" in text
    talk_row = next(
        line
        for line in text.splitlines()
        if line.startswith("| $AUDIOLAB_CACHE/peertube-fr-train |")
        and " | 10 B | 7 B | 3 B |" in line
    )
    # The talk source's total: 10 + 5 + 2 + 3 bytes.
    assert "20 B" in talk_row
    # The free space line names its units and, with df available, df's figure.
    free_line = next(line for line in text.splitlines() if line.startswith("Free space:"))
    assert "free of" in free_line
    if "df -h:" in free_line:
        assert "1024-base" in free_line
