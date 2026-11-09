"""Standalone screening reports, on synthetic CPU-only runs and real comparison code."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import make_clip  # noqa: E402

from audiolab.data.panels import Panel  # noqa: E402
from audiolab.data.store import write_manifest  # noqa: E402
from audiolab.eval.records import load_panel_items, panel_hashes, with_records  # noqa: E402
from audiolab.eval.report import compare_runs, panel_summary  # noqa: E402

SCRIPT = Path(__file__).resolve().parent.parent / "experiments/source-screening/table.py"
spec = importlib.util.spec_from_file_location("screen_table", SCRIPT)
table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table)

SPEECH = "fake-fr-full"
NONSPEECH = "fake-ns-full"
REFERENCE = "Bonjour tout le monde."


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture(autouse=True)
def small_draws(monkeypatch):
    """Only tests reduce bootstrap draws; the script exposes no protocol overrides."""
    monkeypatch.setattr(table, "compare_runs", lambda *args: compare_runs(*args, draws=200))


@pytest.fixture
def runs(tmp_path):
    panels_dir, data_dir = tmp_path / "panels", tmp_path / "data"
    panels_dir.mkdir()
    ids = {}
    for title, language in ((SPEECH, "fr"), (NONSPEECH, "none")):
        dataset = title.rsplit("-", 1)[0]
        clips = [
            make_clip(
                f"{dataset}:{index:03d}",
                language=language,
                group=f"{dataset}:group-{index // 3}",
                text=REFERENCE if language == "fr" else "",
            )
            for index in range(12)
        ]
        write_manifest(data_dir / dataset, clips)
        ids[title] = [clip.id for clip in clips]
        panel = Panel(
            name="full",
            dataset=dataset,
            ids=ids[title],
            items=len(clips),
            seed=20261107,
            manifest_sha256="synthetic",
            clips_sha256="synthetic",
            dataset_line=dataset,
            built="2026-11-09",
        )
        (panels_dir / f"{title}.txt").write_text(panel.text(), encoding="utf-8")
    paths = {}
    for model in ("base", "same", "worse"):
        path = tmp_path / model
        path.mkdir()
        write_json(
            path / "run.json",
            {
                "model": {"name": model},
                "panels": panel_hashes(list(ids), panels_dir),
            },
        )
        records = []
        for title, panel_ids in ids.items():
            for index, clip_id in enumerate(panel_ids):
                if title == SPEECH:
                    hypothesis = "cuisine" if model == "worse" else REFERENCE
                else:
                    hypothesis = "So, yeah." if model == "worse" or index % 3 == 0 else ""
                records.append(
                    {"id": clip_id, "panel": title, "hypothesis": hypothesis, "error": ""}
                )
        (path / "records.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        # Real summaries, not a second approximate WER implementation in the fixture.
        write_json(
            path / "summary.json",
            {
                "panels": {
                    title: panel_summary(
                        title,
                        with_records(load_panel_items(title, panels_dir, data_dir), records),
                        draws=200,
                        seed=20261107,
                    )
                    for title in ids
                }
            },
        )
        paths[model] = path
    return {**paths, "panels_dir": panels_dir, "data_dir": data_dir, "ids": ids}


def command(runs, *candidates):
    argv = [
        "--baseline",
        str(runs["base"]),
        "--panels-dir",
        str(runs["panels_dir"]),
        "--data-dir",
        str(runs["data_dir"]),
    ]
    for label in candidates or ("same",):
        argv += ["--candidate", f"{label}={runs[label]}"]
    return argv


def test_help_needs_no_data_or_environment(capsys, monkeypatch):
    monkeypatch.delenv("AUDIOLAB_DATA", raising=False)
    monkeypatch.delenv("AUDIOLAB_RUNS", raising=False)
    with pytest.raises(SystemExit) as result:
        table.main(["--help"])
    assert result.value.code == 0
    text = capsys.readouterr().out
    assert "NAME=PATH" in text and "example:" in text
    assert "source-screening/table.py" in text


@pytest.mark.parametrize("candidate", [None, "/nope", "=path", "name="])
def test_candidates_are_explicit_and_labelled(candidate, capsys):
    argv = ["--baseline", "/nope"]
    if candidate is not None:
        argv += ["--candidate", candidate]
    with pytest.raises(SystemExit) as result:
        table.main(argv)
    assert result.value.code == 2
    assert not capsys.readouterr().out


def test_duplicate_labels_are_refused_before_loading_inputs(capsys):
    assert (
        table.main(
            [
                "--baseline",
                "/nope",
                "--candidate",
                "same=/one",
                "--candidate",
                "same=/two",
            ]
        )
        == 2
    )
    assert "labels must be unique" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["run.json", "records.jsonl", "summary.json"])
def test_missing_run_files_are_refused(runs, capsys, name):
    target = (runs["same"] / name).resolve()
    assert target.parent == runs["same"].resolve()
    target.unlink()
    assert table.main(command(runs)) == 2
    output = capsys.readouterr()
    assert name in output.err and not output.out


def test_missing_run_is_refused(tmp_path, capsys):
    assert table.main(["--baseline", str(tmp_path / "absent"), "--candidate", "a=/missing"]) == 2
    assert "run.json" in capsys.readouterr().err


@pytest.mark.parametrize("contents", ["{broken json\n", "null\n", "{}\n"])
def test_invalid_records_are_reported_without_a_traceback(runs, capsys, contents):
    (runs["same"] / "records.jsonl").write_text(contents, encoding="utf-8")
    assert table.main(command(runs)) == 2
    output = capsys.readouterr()
    assert "records.jsonl" in output.err and not output.out


def test_missing_clip_is_not_silently_dropped(runs, capsys):
    path = runs["same"] / "records.jsonl"
    path.write_text(
        "\n".join(path.read_text(encoding="utf-8").splitlines()[1:]) + "\n", encoding="utf-8"
    )
    assert table.main(command(runs)) == 2
    assert "1 not recorded" in capsys.readouterr().err


def test_record_from_an_undeclared_panel_is_refused(runs, capsys):
    path = runs["same"] / "records.jsonl"
    extra = {"id": runs["ids"][SPEECH][0], "panel": "unknown-full", "hypothesis": ""}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(extra) + "\n")
    assert table.main(command(runs)) == 2
    assert "1 unknown" in capsys.readouterr().err


def test_changed_panel_files_are_refused(runs, capsys):
    path = runs["panels_dir"] / f"{SPEECH}.txt"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("# changed\n")
    assert table.main(command(runs)) == 2
    assert "panel has changed" in capsys.readouterr().err


def test_different_panel_sets_are_refused(runs, capsys):
    path = runs["same"]
    identity = read_json(path / "run.json")
    del identity["panels"][SPEECH]
    write_json(path / "run.json", identity)
    records = [json.loads(line) for line in (path / "records.jsonl").read_text().splitlines()]
    (path / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records if record["panel"] == NONSPEECH),
        encoding="utf-8",
    )
    summary = read_json(path / "summary.json")
    del summary["panels"][SPEECH]
    write_json(path / "summary.json", summary)
    assert table.main(command(runs)) == 2
    assert "different panels" in capsys.readouterr().err


@pytest.mark.parametrize("panels", [{}, 42, None])
def test_incomplete_or_malformed_summaries_are_refused(runs, capsys, panels):
    write_json(runs["same"] / "summary.json", {"panels": panels})
    assert table.main(command(runs)) == 2
    assert "audiolab score" in capsys.readouterr().err


@pytest.mark.parametrize("section", [None, {"wer": [1]}, {"wer": {"formatted": 42}}])
def test_invalid_summary_sections_are_input_errors(runs, capsys, section):
    path = runs["same"] / "summary.json"
    summary = read_json(path)
    summary["panels"][SPEECH] = section
    write_json(path, summary)
    assert table.main(command(runs)) == 2
    output = capsys.readouterr()
    assert "summary.json" in output.err and not output.out


def test_missing_dataset_manifest_is_an_input_error(runs, capsys):
    argv = command(runs)
    argv[argv.index("--data-dir") + 1] = str(runs["data_dir"] / "absent")
    assert table.main(argv) == 2
    output = capsys.readouterr()
    assert "manifest" in output.err and not output.out


def test_sealed_panels_are_not_used_for_screening(runs, capsys):
    path = runs["same"] / "run.json"
    identity = read_json(path)
    identity["panels"] = {"fake-fr-sealed": "not-accessed"}
    write_json(path, identity)
    assert table.main(command(runs)) == 2
    assert "sealed final panels" in capsys.readouterr().err


def test_rendering_uses_the_official_comparison_and_cautions(runs, capsys):
    before = {
        path: path.read_bytes()
        for name in ("base", "same", "worse")
        for path in runs[name].iterdir()
    }
    assert table.main(command(runs, "worse", "same")) == 0
    text = capsys.readouterr().out
    assert text.count("### Comparison: base (A) against") == 2
    assert text.count("| base better |") == 2
    assert text.count("| not separable |") == 2
    assert "interval approximate" in text
    assert "No id was left out" in text
    assert "12 ids compared in 4 groups" in text
    official = compare_runs(
        runs["base"], runs["worse"], runs["panels_dir"], runs["data_dir"], draws=200
    )
    for panel, section in official["panels"].items():
        assert f"| {panel} | {section['metric']} |" in text
        assert f"{section['a'] * 100:.2f}" in text
        assert f"{section['difference'] * 100:.2f}" in text
        assert f"{section['ci']['low'] * 100:.2f}" in text
        assert f"{section['ci']['high'] * 100:.2f}" in text
    assert before == {path: path.read_bytes() for path in before}


def test_formatted_zero_is_preserved_and_missing_values_are_not_zero(runs, capsys):
    assert table.main(command(runs)) == 0
    assert f"| {SPEECH} | 0.00 % | 0.00 % |" in capsys.readouterr().out
    path = runs["same"] / "summary.json"
    summary = read_json(path)
    summary["panels"][SPEECH]["wer"]["formatted"] = None
    write_json(path, summary)
    assert table.main(command(runs)) == 0
    assert "Not available:" in capsys.readouterr().out


def test_failures_keep_exclusions_visible_and_baselines_candidate_specific(runs, capsys):
    path = runs["same"] / "records.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    failed_id = runs["ids"][NONSPEECH][0]  # One of the baseline's hallucinations.
    for record in records:
        if record["id"] == failed_id:
            record["error"] = "HTTP 500"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    assert table.main(command(runs, "same", "worse")) == 0
    text = capsys.readouterr().out
    assert "1 failed in b" in text and failed_id in text
    assert "11 ids compared" in text
    # Baseline: 3/11 for the failed candidate, 4/12 for the complete one.
    assert "27.27" in text and "33.33" in text


def test_scoring_errors_use_the_common_reporter(runs, capsys, monkeypatch):
    def with_error(*args):
        comparison = compare_runs(*args, draws=200)
        comparison["panels"][SPEECH]["scoring_errors"] = {
            "count": 1,
            "items": [{"id": "unreadable-item", "error": "test diagnostic"}],
        }
        return comparison

    monkeypatch.setattr(table, "compare_runs", with_error)
    assert table.main(command(runs)) == 0
    text = capsys.readouterr().out
    assert "1 hypothesis could not be scored" in text
    assert "unreadable-item" in text and "test diagnostic" in text
