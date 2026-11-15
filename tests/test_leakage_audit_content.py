"""The content-level leakage audit, on synthetic manifests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "experiments/leakage-audit/content.py"
spec = importlib.util.spec_from_file_location("leakage_content", SCRIPT)
content = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = content
spec.loader.exec_module(content)


def text(n_words: int, prefix: str) -> str:
    return " ".join(f"{prefix}{i}" for i in range(n_words))


TEXT_A = text(60, "alpha")
TEXT_B = text(60, "beta")
TEXT_C = text(60, "gamma")
TEXT_D = text(60, "delta")


def row(
    dataset: str,
    group: str,
    kind: str,
    n: int,
    start_cs: int,
    text: str,
    sha: str | None = None,
) -> dict:
    """One manifest line, with every required field of a prepared clip."""
    return {
        "id": f"{dataset}:{group}-{start_cs:07d}",
        "audio": f"audio/{dataset}_{group}_{n}.wav",
        "sha256": sha or hashlib.sha256(f"{dataset}:{group}:{n}".encode()).hexdigest(),
        "duration_s": 10.0,
        "language": "en",
        "text": text,
        "text_has_casing_punct": True,
        "group": f"{dataset}:{group}",
        "group_kind": kind,
        "source_split": "test",
        "license": "CC0-1.0",
    }


def write_dataset(root: Path, name: str, rows: list[dict]) -> None:
    (root / name).mkdir(parents=True)
    (root / name / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def synthetic_data(root: Path) -> None:
    """One evaluation set and two training sets:

    - ``talkA`` is re-uploaded in both training sets under other ids
      (``talkA-copy``, ``talkA-copy2``);
    - ``talkC`` is in the evaluation set and in both training sets under the
      same id (a SAME GROUP pair);
    - ``talkB`` (evaluation) and ``talkD`` (training) are unrelated texts.
    """
    talk = "recording"
    # The second clip of a talk is the first half of the text (word-aligned),
    # so the group's shingle set is the 60 words' 55 6-grams.
    HALF_A = " ".join(TEXT_A.split()[:30])
    write_dataset(
        root,
        "syn-eval",
        [
            row("syn-eval", "talkA", talk, 1, 0, TEXT_A, sha="a" * 64),
            row("syn-eval", "talkA", talk, 2, 1000, HALF_A, sha="b" * 64),
            row("syn-eval", "talkB", talk, 3, 0, TEXT_B),
            row("syn-eval", "talkC", talk, 4, 0, TEXT_C),
        ],
    )
    write_dataset(
        root,
        "syn-train",
        [
            row("syn-train", "talkA-copy", talk, 1, 0, TEXT_A, sha="a" * 64),
            row("syn-train", "talkA-copy", talk, 2, 1000, HALF_A, sha="b" * 64),
            row("syn-train", "talkC", talk, 3, 0, TEXT_C),
            row("syn-train", "talkD", talk, 4, 0, TEXT_D),
        ],
    )
    write_dataset(
        root,
        "syn-train-40k",
        [
            row("syn-train-40k", "talkA-copy2", talk, 1, 0, TEXT_A, sha="a" * 64),
            row("syn-train-40k", "talkC", talk, 2, 0, TEXT_C),
        ],
    )


def synthetic_panels(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for panel in ("fast", "full", "sealed"):
        (root / f"syn-eval-{panel}.txt").write_text("")
        (root / f"nonspeech-{panel}.txt").write_text("")


def test_the_planted_reupload_is_found_and_unrelated_text_is_not(tmp_path):
    write_panels = tmp_path / "panels"
    data = tmp_path / "data"
    synthetic_panels(write_panels)
    synthetic_data(data)
    eval_sets, train_sets = content.discover(data, write_panels)
    assert eval_sets == ["syn-eval"]
    assert train_sets == ["syn-train", "syn-train-40k"]

    result = content.audit(eval_sets, train_sets, data)
    pairs = {(f.pair.left[1], f.pair.right[0], f.pair.right[1]) for f in result.findings}
    # The re-upload (different ids) and the SAME GROUP pair are found.
    assert ("talkA", "syn-train", "talkA-copy") in pairs
    assert ("talkA", "syn-train-40k", "talkA-copy2") in pairs
    assert ("talkC", "syn-train", "talkC") in pairs
    assert ("talkC", "syn-train-40k", "talkC") in pairs
    # The unrelated texts never appear.
    assert not any(name in pairs for name in ("talkB", "talkD"))

    by_key = {(f.pair.left, f.pair.right): f for f in result.findings}
    reupload = by_key[(("syn-eval", "talkA"), ("syn-train", "talkA-copy"))]
    # full shared shingle set of talkA: 60 words -> 55 6-grams
    assert reupload.pair.shared == 55
    assert reupload.ratio == 1.0
    assert not reupload.pair.same_group
    same = by_key[(("syn-eval", "talkC"), ("syn-train", "talkC"))]
    assert same.pair.same_group


def test_word_recall_counts_only_time_overlapping_clips():
    eval_clips = [(0.0, 10.0, TEXT_A, "a" * 64), (10.0, 20.0, TEXT_B, "b" * 64)]
    train_clips = [(0.0, 10.0, TEXT_A, "a" * 64)]
    mean, counted, ge07 = content.word_recall(eval_clips, train_clips)
    # Only the first evaluation clip overlaps a training clip, and it is
    # word-identical to it.
    assert counted == 1
    assert mean == 1.0
    assert ge07 == 1
    # No time overlap at all: no recall, not a number.
    mean, counted, ge07 = content.word_recall(eval_clips, [(50.0, 60.0, TEXT_A, "a" * 64)])
    assert (mean, counted, ge07) == (None, 0, 0)
    # Clips without a start time (no talk) give no recall.
    mean, counted, ge07 = content.word_recall([(None, 10.0, TEXT_A, "a" * 64)], train_clips)
    assert (mean, counted, ge07) == (None, 0, 0)


def test_the_reupload_shortlist_uses_ratio_and_distinct_ids(tmp_path):
    write_panels = tmp_path / "panels"
    data = tmp_path / "data"
    synthetic_panels(write_panels)
    synthetic_data(data)
    result = content.audit(["syn-eval"], ["syn-train", "syn-train-40k"], data)
    reuploads = {(p.left, p.right) for p in result.reuploads}
    # The two re-uploaded ids are a re-upload pair (ratio 100 % >= 75 %).
    assert reuploads == {(("syn-train", "talkA-copy"), ("syn-train-40k", "talkA-copy2"))}
    # The same id across the two sets is reported, but is not a re-upload.
    across = {(p.left, p.right) for p in result.train_pairs}
    assert (("syn-train", "talkC"), ("syn-train-40k", "talkC")) in across
    assert result.within_pairs == []


def test_skip_markers_and_nontrain_names_are_excluded(tmp_path):
    write_panels = tmp_path / "panels"
    data = tmp_path / "data"
    synthetic_panels(write_panels)
    synthetic_data(data)
    for name in ("syn-train.pre-2026-01-01", "syn-train.curly-2026-01-01", "noise-bank"):
        (data / name).mkdir()
        (data / name / "manifest.jsonl").write_text("")
    eval_sets, train_sets = content.discover(data, write_panels)
    assert eval_sets == ["syn-eval"]
    assert train_sets == ["syn-train", "syn-train-40k"]


def test_the_cli_reports_the_tables(tmp_path, capsys):
    write_panels = tmp_path / "panels"
    data = tmp_path / "data"
    synthetic_panels(write_panels)
    synthetic_data(data)
    assert content.main(["--data-dir", str(data), "--panels-dir", str(write_panels)]) == 0
    out = capsys.readouterr().out
    assert "syn-eval:talkA" in out
    assert "syn-train:talkA-copy" in out
    assert "SAME GROUP" in out
    # The re-upload shortlist holds the cross-set re-upload only.
    assert out.count("talkA-copy2") >= 2
    assert "syn-train-40k:talkC" in out


def test_main_refuses_without_prepared_data(tmp_path, capsys):
    assert content.main(["--data-dir", str(tmp_path / "missing")]) == 2
    assert "no prepared data" in capsys.readouterr().err
