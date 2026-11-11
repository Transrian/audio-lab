"""Label-text check (plan 03 A4, plan 08): synthetic datasets, table format, read-only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "experiments/label-text/check_text.py"
spec = importlib.util.spec_from_file_location("check_text", SCRIPT)
check_text = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = check_text
spec.loader.exec_module(check_text)


def write_manifest(root: Path, dataset: str, texts: list[str]) -> None:
    """A manifest of one clip per text, with every required field set."""
    lines = []
    for index, text in enumerate(texts):
        clip = {
            "id": f"{dataset}:clip-{index}",
            "audio": f"{index:04d}.wav",
            "sha256": "0" * 64,
            "duration_s": 1.0,
            "language": "fr",
            "text": text,
            "text_has_casing_punct": True,
            "group": f"{dataset}:group-0",
            "group_kind": "recording",
            "source_split": "train",
            "license": "CC0-1.0",
        }
        lines.append(json.dumps(clip, ensure_ascii=False))
    (root / dataset).mkdir(parents=True)
    (root / dataset / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def snapshot(root: Path) -> dict[str, tuple[str, int]]:
    """(content sha, mtime ns) of every file under the root: proof of read-only."""
    state = {}
    for path in root.rglob("*"):
        if path.is_file():
            state[str(path)] = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                path.stat().st_mtime_ns,
            )
    return state


def test_help_needs_no_environment(monkeypatch, capsys):
    monkeypatch.delenv("AUDIOLAB_DATA", raising=False)
    with pytest.raises(SystemExit) as error:
        check_text.main(["--help"])
    assert error.value.code == 0
    assert "prints nothing" in capsys.readouterr().out


def test_clean_dataset_prints_nothing(tmp_path, capsys):
    root = tmp_path / "data"
    write_manifest(root, "clean-train", ["hello world", "café crème, 2020-2021."])
    assert check_text.main(["--data-dir", str(root), "clean-train"]) == 0
    assert capsys.readouterr().out == ""


def test_table_per_dataset_and_character(tmp_path, capsys):
    root = tmp_path / "data"
    write_manifest(
        root,
        "dirty-train",
        [
            "« bonjour »",  # « and »
            "« au revoir »",  # both characters again, in another clip: each count is 2
            "wait — really?",  # em dash
            "wait … really",  # ellipsis
            "clean line",
        ],
    )
    write_manifest(root, "clean-train", ["hello"])
    assert check_text.main(["--data-dir", str(root), "dirty-train", "clean-train"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "| dataset | character | clips |",
        "| --- | --- | --- |",
        "| dirty-train | `«` | 2 |",
        "| dirty-train | `»` | 2 |",
        "| dirty-train | `—` | 1 |",
        "| dirty-train | `…` | 1 |",
    ]


def test_a_character_twice_in_one_clip_counts_once(tmp_path, capsys):
    """One clip holding « twice and » twice gives one clip per character."""
    root = tmp_path / "data"
    write_manifest(root, "x-train", ["« a » « b »"])
    assert check_text.main(["--data-dir", str(root), "x-train"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "| dataset | character | clips |",
        "| --- | --- | --- |",
        "| x-train | `«` | 1 |",
        "| x-train | `»` | 1 |",
    ]


def test_default_is_every_train_directory(tmp_path, monkeypatch, capsys):
    root = tmp_path / "data"
    write_manifest(root, "dirty-train", ["« x »"])
    write_manifest(root, "clean-train", ["hello"])
    write_manifest(root, "fleurs-fr", ["« evaluation »"])  # not a training set
    monkeypatch.setenv("AUDIOLAB_DATA", str(root))
    assert check_text.main([]) == 0
    out = capsys.readouterr().out
    assert "dirty-train" in out
    assert "clean-train" not in out
    assert "fleurs-fr" not in out


def test_missing_dataset_fails_with_exit_2(tmp_path, capsys):
    root = tmp_path / "data"
    root.mkdir()
    assert check_text.main(["--data-dir", str(root), "nope-train"]) == 2
    assert "no manifest.jsonl" in capsys.readouterr().err


def test_malformed_manifest_fails_with_exit_2(tmp_path, capsys):
    root = tmp_path / "data"
    (root / "bad-train").mkdir(parents=True)
    (root / "bad-train" / "manifest.jsonl").write_text(
        json.dumps({"id": "bad-train:x", "text": "« y »"}) + "\n", encoding="utf-8"
    )
    assert check_text.main(["--data-dir", str(root), "bad-train"]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert "missing fields" in err


def test_never_writes_under_the_data_root(tmp_path):
    root = tmp_path / "data"
    write_manifest(root, "dirty-train", ["« x »", "plain y"])
    before = snapshot(root)
    assert check_text.main(["--data-dir", str(root)]) == 0
    assert snapshot(root) == before, "the checker wrote under the data root"
