"""NASA filters analysis: synthetic sets, cache and run, table format, read-only."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "experiments/nasa-filters/analyze.py"
spec = importlib.util.spec_from_file_location("nasa_filters", SCRIPT)
nasa_filters = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nasa_filters
spec.loader.exec_module(nasa_filters)

SUFFIX = ".stt-c-parakeet-tdt-0.6B-v3.clips.jsonl"


def clip_line(
    slug: str,
    start_cs: int,
    show: str,
    text: str = "a line of speech",
    base_wer: float = 0.0,
) -> str:
    """One manifest line: its id carries the start in centiseconds."""
    return json.dumps(
        {
            "id": f"nasa-en-train:{slug}-{start_cs:07d}",
            "audio": f"audio/{slug}-{start_cs:07d}.wav",
            "sha256": "0" * 64,
            "duration_s": 5.0,
            "language": "en",
            "text": text,
            "text_has_casing_punct": True,
            "group": f"nasa-en-train:{slug}",
            "group_kind": "recording",
            "source_split": "listing",
            "license": "public domain (US government work)",
            "source_author": f"NASA, {show}",
            "base_wer": base_wer,
        }
    )


def write_manifest(root: Path, dataset: str, lines: list[str]) -> None:
    (root / dataset).mkdir(parents=True)
    (root / dataset / "manifest.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_nonspeech(root: Path, ids: dict[str, str]) -> None:
    """The non-speech manifest: one clip per id, its category the value."""
    lines = []
    for number, (clip_id, category) in enumerate(sorted(ids.items())):
        lines.append(
            json.dumps(
                {
                    "id": clip_id,
                    "audio": f"audio/{number:04d}.wav",
                    "sha256": "1" * 64,
                    "duration_s": 5.0,
                    "language": "none",
                    "text": "",
                    "text_has_casing_punct": False,
                    "group": clip_id,
                    "group_kind": "clip",
                    "source_split": "test",
                    "license": "CC0-1.0",
                    "category": category,
                }
            )
        )
    write_manifest(root, "nonspeech", lines)


def write_cache(
    cache_root: Path, slug: str, starts_cs: list[int], dataset: str = "nasa-en-train"
) -> None:
    """The per-clip answer cache of one episode, one line per asked clip."""
    (cache_root / dataset).mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {"key": f"{start / 100:.2f}-{(start + 5000) / 100:.2f}", "answer": {"text": "heard"}}
        )
        for start in starts_cs
    ]
    (cache_root / dataset / f"{slug}{SUFFIX}").write_text("\n".join(lines) + "\n", encoding="utf-8")


def record(clip_id: str, hypothesis: str, panel: str = "nonspeech-full") -> str:
    return json.dumps({"id": clip_id, "panel": panel, "hypothesis": hypothesis, "error": None})


def write_run(runs_root: Path, records: list[str]) -> None:
    (runs_root / "eval" / "screen-nasa-en" / "full").mkdir(parents=True)
    (runs_root / "eval" / "screen-nasa-en" / "full" / "records.jsonl").write_text(
        "\n".join(records) + "\n", encoding="utf-8"
    )


def world(tmp_path: Path) -> tuple[Path, Path, Path]:
    """The synthetic NASA world: two shows, one rewrite loss, three gate drops.

    ep-one (Show A): kept 10.00 and 20.00, rewrite-dropped 30.00, gate-dropped
    40.00 and 50.00. ep-two (Show B): kept 1.00, gate-dropped 2.00.
    """
    data, runs, cache = (tmp_path / name for name in ("data", "runs", "cache"))
    write_manifest(
        data,
        "nasa-en-train.curly-2026-11-09",
        [
            clip_line("ep-one", 1000, "Show A"),
            clip_line("ep-one", 2000, "Show A"),
            clip_line("ep-one", 3000, "Show A"),
            clip_line("ep-two", 100, "Show B"),
        ],
    )
    write_manifest(
        data,
        "nasa-en-train",
        [
            clip_line("ep-one", 1000, "Show A", base_wer=0.5),
            clip_line("ep-one", 2000, "Show A"),
            clip_line("ep-two", 100, "Show B"),
        ],
    )
    write_nonspeech(
        data,
        {
            "nonspeech:clip-0": "silence",
            "nonspeech:clip-1": "noise",
            "nonspeech:clip-2": "animals/dog",
            "nonspeech:clip-3": "human/laughing",
        },
    )
    write_cache(cache, "ep-one", [1000, 2000, 3000, 4000, 5000])
    write_cache(cache, "ep-two", [100, 200])
    write_run(
        runs,
        [
            record("nonspeech:clip-0", ""),
            record("nonspeech:clip-1", "Oh"),  # an older record, superseded below
            record("nonspeech:clip-1", "Yeah."),
            record("nonspeech:clip-2", "So they use the two microphones."),
            record("nonspeech:clip-3", "It \u2047 s new."),
            record("other:clip", "an answer on another panel", panel="fleurs-fr-full"),
        ],
    )
    return data, runs, cache


def main_in(tmp_path: Path, argv: list[str] | None = None) -> int:
    data, runs, cache = world(tmp_path)
    return run(data, runs, cache, argv)


def run(data: Path, runs: Path, cache: Path, argv: list[str] | None = None) -> int:
    args = ["--data-dir", str(data), "--runs-dir", str(runs), "--cache-dir", str(cache)]
    return nasa_filters.main(args + (argv or []))


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
    for variable in ("AUDIOLAB_DATA", "AUDIOLAB_RUNS", "AUDIOLAB_CACHE"):
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(SystemExit) as error:
        nasa_filters.main(["--help"])
    assert error.value.code == 0
    assert "read-only" in capsys.readouterr().out


def test_missing_environment_is_exit_2(monkeypatch, capsys):
    for variable in ("AUDIOLAB_DATA", "AUDIOLAB_RUNS", "AUDIOLAB_CACHE"):
        monkeypatch.delenv(variable, raising=False)
    with pytest.raises(SystemExit) as error:
        nasa_filters.main([])
    assert error.value.code == 2
    assert "AUDIOLAB_DATA" in capsys.readouterr().err


def test_table_by_show(tmp_path, capsys):
    assert main_in(tmp_path) == 0
    out = capsys.readouterr().out
    # One row per show: its episodes, clips, hours and mean base_wer.
    assert "| Show A | 1 | 2 | 0.0 | 25.0 % |" in out
    assert "| Show B | 1 | 1 | 0.0 | 0.0 % |" in out
    assert "Houston" not in out


def test_table_curious(tmp_path, capsys):
    # The synthetic world has no Curious Universe episode: zero episodes, zero
    # clips, and the share still divides by the whole set.
    assert main_in(tmp_path) == 0
    out = capsys.readouterr().out
    assert "| 0 | 0 | 0.00 % (0 of 3) |" in out


def test_table_hallucinations(tmp_path, capsys):
    assert main_in(tmp_path) == 0
    out = capsys.readouterr().out
    # The empty answer and the other panel stay out; the newest record wins.
    assert "| noise | 'Yeah.' |  |" in out
    assert "| animals/dog | 'So they use the two microphones.' | yes |" in out
    assert "| human/laughing | 'It \u2047 s new.' | yes |" in out
    assert "'Oh'" not in out
    assert "an answer on another panel" not in out
    assert "3 non-empty answers" in out
    assert "2 of the 3 answers look like podcast speech" in out
    assert "1 hold the tokenizer's unknown token" in out


def test_table_edge_gate(tmp_path, capsys):
    assert main_in(tmp_path) == 0
    out = capsys.readouterr().out
    # Asked: every cached answer. Gate-dropped: in the cache, not in the first
    # set. Rewrite-dropped: in the first set, not in the current one.
    assert "| Show A | 1 | 5 | 3 | 2 | 1 |" in out
    assert "| Show B | 1 | 2 | 1 | 1 | 0 |" in out
    assert "3 of the 7 asked clips were dropped by the gate (42.9 %)" in out
    assert "1 clips were lost in the label rewrite" in out
    assert "base_wer" in out and "not recorded" in out
    assert "NOTE:" not in out


def test_kept_clip_without_cached_answer_fails(tmp_path, capsys):
    data, runs, cache = world(tmp_path)
    # A kept clip whose start the cache does not hold: the arithmetic would
    # go negative, so the script must say so rather than print it.
    (cache / "nasa-en-train" / f"ep-one{SUFFIX}").write_text(
        json.dumps({"key": "10.00-15.00", "answer": {"text": "heard"}}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ep-one"):
        nasa_filters.table_edge_gate(
            cache / "nasa-en-train",
            SUFFIX,
            list(nasa_filters.read_manifest(data / "nasa-en-train")),
            list(nasa_filters.read_manifest(data / "nasa-en-train.curly-2026-11-09")),
        )


def test_table_characters_clean(tmp_path, capsys):
    assert main_in(tmp_path) == 0
    out = capsys.readouterr().out
    assert "0 clips hold a character the base model cannot write." in out


def test_table_characters_dirty(tmp_path, capsys):
    data, runs, cache = world(tmp_path)
    write_manifest(data, "dirty", [clip_line("ep-one", 1000, "Show A", text="« a »")])
    write_cache(cache, "ep-one", [1000], dataset="dirty")
    assert run(data, runs, cache, ["--dataset", "dirty", "--old-dataset", "dirty"]) == 0
    out = capsys.readouterr().out
    assert "## 5. Characters of dirty's labels" in out
    assert "| `«` | 1 |" in out and "| `»` | 1 |" in out


def test_missing_run_is_exit_2(tmp_path, capsys):
    data, runs, cache = world(tmp_path)
    (runs / "eval" / "screen-nasa-en" / "full" / "records.jsonl").unlink()
    assert run(data, runs, cache) == 2
    assert "records.jsonl" in capsys.readouterr().err


def test_never_writes_under_the_input_roots(tmp_path):
    data, runs, cache = world(tmp_path)
    before = {root: snapshot(root) for root in (data, runs, cache)}
    assert run(data, runs, cache) == 0
    for root, state in before.items():
        assert snapshot(root) == state, f"the analysis wrote under {root}"


def test_speech_like():
    # The shape rule: three or more words with two of them different.
    assert nasa_filters.speech_like("So they use the two microphones.")
    assert nasa_filters.speech_like("This is mist! This is mist!")
    assert nasa_filters.speech_like("It \u2047 s new.")
    assert nasa_filters.speech_like("Oh, no, no, no.")
    assert not nasa_filters.speech_like("Oh")
    assert not nasa_filters.speech_like("Yeah.")
    assert not nasa_filters.speech_like("No, no,")
    assert not nasa_filters.speech_like("Come there!")
    assert not nasa_filters.speech_like("Yeah, yeah, yeah.")
    assert not nasa_filters.speech_like("Right, right, right, right, right.")
    assert not nasa_filters.speech_like("BIT! BIT! BIT! BIT!")
    assert not nasa_filters.speech_like("Oh! Oh! Oh, oh!")
    # The unknown token is not a word.
    assert not nasa_filters.speech_like("\u2047 \u2047 \u2047")
