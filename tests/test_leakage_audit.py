"""The VoxPopuli session-overlap audit, on synthetic manifests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "experiments/leakage-audit/sessions.py"
spec = importlib.util.spec_from_file_location("leakage_sessions", SCRIPT)
sessions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sessions)


def test_session_is_the_prefix_before_the_language_tag():
    assert (
        sessions.session_of("vp-en:20090310-0900-PLENARY-7-en_20090310-11:10:07_2")
        == "20090310-0900-PLENARY-7"
    )
    # No sitting number: the tag still delimits the session.
    assert (
        sessions.session_of("vp-en:20190415-0900-PLENARY-fr_20190415-09:01:00_0")
        == "20190415-0900-PLENARY"
    )
    # The sitting name may carry a lowercase qualifier (two of the prepared
    # sittings do: PLENARY-widetrim and PLENARY-witholdRO).
    assert (
        sessions.session_of("vp-fr:20180911-0900-PLENARY-witholdRO-fr_20180911-10:53:30_12")
        == "20180911-0900-PLENARY-witholdRO"
    )


def test_a_malformed_id_fails_loudly():
    with pytest.raises(ValueError, match="language tag"):
        sessions.session_of("vp-fr:not-a-voxpopuli-id")


def row(clip_id: str, duration: float) -> dict:
    return {"id": clip_id, "duration_s": duration, "group": clip_id.split(":", 1)[0]}


def test_overlap_counts_shared_sessions_clips_and_hours():
    eval_rows = [
        row("vp:20090310-0900-PLENARY-7-en_20090310-11:10:07_2", 3600.0),
        row("vp:20090310-0900-PLENARY-7-en_20090310-11:10:07_4", 1800.0),
        row("vp:20090311-0900-PLENARY-6-en_20090311-11:11:51_0", 7200.0),
    ]
    train_rows = [
        row("vp:20090310-0900-PLENARY-7-fr_20090310-11:00:00_0", 900.0),
        row("vp:20090311-0900-PLENARY-6-fr_20090311-10:00:00_0", 900.0),
        row("vp:20090312-0900-PLENARY-5-fr_20090312-10:00:00_0", 900.0),
    ]
    table = sessions.overlap(eval_rows, train_rows)
    assert table["eval_sessions"] == 2
    assert table["train_sessions"] == 3
    assert table["shared_sessions"] == 2
    assert table["eval_clips"] == (3, 3)
    assert table["train_clips"] == (3, 2)
    assert table["eval_hours"] == pytest.approx(3.5)
    assert table["train_hours"] == pytest.approx(0.5)


def test_the_cli_prints_the_table(tmp_path, monkeypatch, capsys):
    for name, rows in (
        ("voxpopuli-en-nonnative", [row("vp:20090310-0900-PLENARY-7-en_a", 3600.0)]),
        ("voxpopuli-fr-train", [row("vp:20090310-0900-PLENARY-7-fr_a", 1800.0)]),
    ):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert sessions.main(["--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "shared sessions: 1" in out
    assert "evaluation clips in shared sessions: 1 of 1 (1.00 h)" in out
    assert "train clips in shared sessions: 1 of 1 (0.50 h)" in out
