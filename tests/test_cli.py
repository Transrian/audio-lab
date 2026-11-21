"""The command line: help, exit codes and the data commands, all offline."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402
from helpers import make_clip, start_transcription_server, write_tone  # noqa: E402

from audiolab import cli  # noqa: E402
from audiolab.core.audio import pcm_sha256  # noqa: E402
from audiolab.core.gpu import parse_nvidia_smi  # noqa: E402
from audiolab.data.panels import build_panels, read_panel  # noqa: E402
from audiolab.data.store import MANIFEST_NAME, read_manifest, write_manifest  # noqa: E402
from audiolab.eval.client import NotAnswering  # noqa: E402
from audiolab.eval.records import panel_hashes  # noqa: E402

PANEL = "fake-fr-fast"


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "gpus" in out and "eval" in out and "compare" in out


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "usage: audiolab" in capsys.readouterr().out


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--version"])
    assert exit_info.value.code == 0
    assert "audiolab" in capsys.readouterr().out


def test_every_command_help_names_the_command_and_shows_an_example(capsys):
    """A --help that only lists flags leaves the reader to guess the invocation."""
    for argv, word in (
        (["data", "--help"], "data"),
        (["data", "list", "--help"], "list"),
        (["data", "fetch", "--help"], "fetch"),
        (["data", "panels", "--help"], "panels"),
        (["eval", "--help"], "transcribe"),
        (["score", "--help"], "score a finished run again"),
        (["compare", "--help"], "compare two runs"),
        (["gpus", "--help"], "gpus"),
    ):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(argv)
        assert exit_info.value.code == 0, argv
        out = capsys.readouterr().out
        assert word in out, argv
        assert "example:" in out, argv
        assert "audiolab" in out, argv


def test_the_normalizer_flag_is_documented(capsys):
    """The v1 reproduction is a documented command, not a hidden one."""
    for argv in (["score", "--help"], ["compare", "--help"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        out = capsys.readouterr().out
        assert "--normalizer" in out, argv
        assert "{1,2}" in out, argv
    # `eval` scores a new run with the current version; only re-scoring picks one.
    with pytest.raises(SystemExit):
        cli.main(["eval", "--help"])
    assert "--normalizer" not in capsys.readouterr().out


def test_runs_dir_is_only_where_it_is_used(capsys):
    """--runs-dir decides where a run is written, so score and compare must not take it."""
    for argv in (["score", "--help"], ["compare", "--help"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        assert "--runs-dir" not in capsys.readouterr().out, argv
    with pytest.raises(SystemExit):
        cli.main(["eval", "--help"])
    assert "--runs-dir" in capsys.readouterr().out


def test_every_command_has_help(capsys):
    for argv in (
        ["data", "--help"],
        ["data", "list", "--help"],
        ["data", "fetch", "--help"],
        ["data", "panels", "--help"],
        ["eval", "--help"],
        ["score", "--help"],
        ["compare", "--help"],
        ["gpus", "--help"],
    ):
        with pytest.raises(SystemExit) as exit_info:
            cli.main(argv)
        assert exit_info.value.code == 0, argv
        assert "usage: audiolab" in capsys.readouterr().out


def test_gpus_command(monkeypatch, capsys):
    gpus = parse_nvidia_smi("0, GPU-aaaa, NVIDIA GeForce RTX 5090, 32607\n")
    monkeypatch.setattr(cli, "list_gpus", lambda: gpus)
    assert cli.main(["gpus"]) == 0
    out = capsys.readouterr().out
    assert "NVIDIA GeForce RTX 5090" in out
    assert "GPU-aaaa" in out
    assert "32 GiB" in out


def test_gpus_command_without_gpu(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_gpus", lambda: [])
    assert cli.main(["gpus"]) == 1
    assert "No NVIDIA GPU found." in capsys.readouterr().out


def test_data_list_names_the_spoken_languages(capsys, tmp_path):
    assert cli.main(["--data-dir", str(tmp_path), "data", "list"]) == 0
    out = capsys.readouterr().out
    for name in ("fleurs-fr", "fleurs-en", "mls-fr", "librispeech-en", "nonspeech"):
        assert name in out
    assert str(tmp_path) in out


def test_data_fetch_refuses_an_unknown_name(capsys, tmp_path):
    assert cli.main(["--data-dir", str(tmp_path), "data", "fetch", "nope"]) == 2
    assert "nope" in capsys.readouterr().err


def test_data_panels_requires_a_dataset(capsys, tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--data-dir", str(tmp_path), "data", "panels"])
    assert exit_info.value.code == 2
    assert "dataset" in capsys.readouterr().err


@pytest.fixture
def prepared(tmp_path) -> dict:
    """A tiny prepared dataset with its panels, plus the paths the CLI needs."""
    data_dir = tmp_path / "datasets"
    # The audio hash is the real one: ``data panels --check`` hashes the clips.
    clips = []
    for speaker in range(6):
        for index in range(3):
            clip = make_clip(f"fake:{speaker:03d}-{index:03d}", group=f"fake:spk{speaker:03d}")
            path = write_tone(data_dir / "fake-fr" / clip.audio, seconds=0.5)
            clips.append(replace(clip, sha256=pcm_sha256(path)))
    write_manifest(data_dir / "fake-fr", clips)
    panels_dir = tmp_path / "panels"
    build_panels("fake-fr", clips, panels_dir, built="2026-11-07")
    model = tmp_path / "model.yaml"
    model.write_text(
        "name: fake-model\nmodel: served-id\nbase_url_env: FAKE_URL\n", encoding="utf-8"
    )
    suite = tmp_path / "fast.yaml"
    suite.write_text(f"name: fast\npanels: [{PANEL}]\n", encoding="utf-8")
    return {
        "data_dir": data_dir,
        "panels_dir": panels_dir,
        "runs_dir": tmp_path / "runs",
        "model": model,
        "suite": suite,
    }


def answer(text: str = "bonjour tout le monde"):
    def respond(headers):
        return (200, json.dumps({"text": text}), "application/json")

    return start_transcription_server(respond)


def test_a_run_through_the_command_line(prepared, capsys):
    server, url = answer()
    try:
        code = cli.main(
            [
                "--data-dir",
                str(prepared["data_dir"]),
                "eval",
                "--model",
                str(prepared["model"]),
                "--suite",
                str(prepared["suite"]),
                "--panels-dir",
                str(prepared["panels_dir"]),
                "--runs-dir",
                str(prepared["runs_dir"]),
                "--base-url",
                url,
            ]
        )
    finally:
        server.shutdown()
    assert code == 0
    out = capsys.readouterr().out
    run_dir = prepared["runs_dir"] / "eval" / "fake-model" / "fast"
    assert str(run_dir / "summary.json") in out
    assert (run_dir / "report.md").is_file()

    # score rebuilds the same two files from the records alone.
    assert (
        cli.main(
            [
                "--data-dir",
                str(prepared["data_dir"]),
                "score",
                str(run_dir),
                "--panels-dir",
                str(prepared["panels_dir"]),
            ]
        )
        == 0
    )
    assert "scored" in capsys.readouterr().out

    assert (
        cli.main(
            [
                "--data-dir",
                str(prepared["data_dir"]),
                "compare",
                str(run_dir),
                str(run_dir),
                "--panels-dir",
                str(prepared["panels_dir"]),
                "--json",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "not separable" in out
    assert '"verdict": "not separable"' in out


def test_eval_sends_as_many_requests_at_once_as_the_flag_or_the_environment_says(
    prepared, monkeypatch, capsys
):
    asked: list[int] = []

    def run_suite(context):
        asked.append(context.concurrency)
        return prepared["runs_dir"]

    monkeypatch.setattr(cli, "run_suite", run_suite)
    command = ["eval", "--model", str(prepared["model"]), "--suite", str(prepared["suite"])]
    monkeypatch.delenv("AUDIOLAB_CONCURRENCY", raising=False)
    assert cli.main(command) == 0
    monkeypatch.setenv("AUDIOLAB_CONCURRENCY", "served-id=4")
    assert cli.main(command) == 0
    assert cli.main([*command, "--concurrency", "2"]) == 0
    assert asked == [1, 4, 2]

    for wrong in ("0", "²"):  # a superscript is a digit, not a number int() reads
        with pytest.raises(SystemExit):
            cli.main([*command, "--concurrency", wrong])
        assert "at least 1" in capsys.readouterr().err


def test_a_model_that_does_not_answer_its_warmup_exits_with_two(prepared, monkeypatch, capsys):
    def run_suite(context):
        raise NotAnswering("served-id did not answer within 15 min (HTTP 503)")

    monkeypatch.setattr(cli, "run_suite", run_suite)
    command = ["eval", "--model", str(prepared["model"]), "--suite", str(prepared["suite"])]

    assert cli.main(command) == 2
    assert "audiolab: served-id did not answer within 15 min" in capsys.readouterr().err


def test_a_missing_endpoint_variable_exits_with_two(prepared, capsys, monkeypatch):
    monkeypatch.delenv("FAKE_URL", raising=False)
    code = cli.main(
        [
            "--data-dir",
            str(prepared["data_dir"]),
            "eval",
            "--model",
            str(prepared["model"]),
            "--suite",
            str(prepared["suite"]),
            "--panels-dir",
            str(prepared["panels_dir"]),
            "--runs-dir",
            str(prepared["runs_dir"]),
        ]
    )
    assert code == 2
    assert "FAKE_URL" in capsys.readouterr().err


def test_a_missing_manifest_says_which_dataset_to_fetch(prepared, capsys):
    (prepared["data_dir"] / "fake-fr" / MANIFEST_NAME).unlink()
    code = cli.main(
        [
            "--data-dir",
            str(prepared["data_dir"]),
            "data",
            "panels",
            "fake-fr",
            "--panels-dir",
            str(prepared["panels_dir"]),
        ]
    )
    assert code == 2
    assert "fake-fr" in capsys.readouterr().err


def test_data_panels_check_reports_what_it_found(prepared, capsys):
    code = cli.main(
        [
            "--data-dir",
            str(prepared["data_dir"]),
            "data",
            "panels",
            "fake-fr",
            "--check",
            "--panels-dir",
            str(prepared["panels_dir"]),
        ]
    )
    assert code == 0
    assert "ok" in capsys.readouterr().out

    changed = prepared["data_dir"] / "fake-fr" / MANIFEST_NAME
    clips = read_manifest(prepared["data_dir"] / "fake-fr")
    write_manifest(prepared["data_dir"] / "fake-fr", clips[:-1])
    assert changed.is_file()
    code = cli.main(
        [
            "--data-dir",
            str(prepared["data_dir"]),
            "data",
            "panels",
            "fake-fr",
            "--check",
            "--panels-dir",
            str(prepared["panels_dir"]),
        ]
    )
    assert code == 1
    assert "different manifest" in capsys.readouterr().err


def test_data_panels_check_hashes_the_audio_of_the_panel(prepared, capsys):
    """A panel names audio: replacing a clip's file must fail the check."""
    panel = read_panel(prepared["panels_dir"] / f"{PANEL}.txt")
    clip = read_manifest(prepared["data_dir"] / "fake-fr")[0]
    assert clip.id in panel.ids
    write_tone(prepared["data_dir"] / "fake-fr" / clip.audio, seconds=0.5, frequency=999)

    code = cli.main(
        [
            "--data-dir",
            str(prepared["data_dir"]),
            "data",
            "panels",
            "fake-fr",
            "--check",
            "--panels-dir",
            str(prepared["panels_dir"]),
        ]
    )
    assert code == 1
    assert "does not match its sha256" in capsys.readouterr().err


def v1_run(prepared) -> Path:
    """A finished run scored with normalizer v1, as one published before v2 is."""
    panel = read_panel(prepared["panels_dir"] / f"{PANEL}.txt")
    run_dir = prepared["runs_dir"] / "eval" / "fake-model" / "fast"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "model": {"name": "fake-model", "model": "served-id", "server_label": ""},
                "normalizer_version": 1,
                "suite": "fast",
                "audiolab_version": "0.0.0",
                "git_commit": "test",
                "panels": panel_hashes([PANEL], prepared["panels_dir"]),
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "id": clip_id,
            "panel": PANEL,
            "hypothesis": "it is a test",
            "error": "",
        }
        for clip_id in panel.ids
    ]
    (run_dir / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return run_dir


def score_command(prepared, run_dir: Path, normalizer: int | None) -> list[str]:
    argv = [
        "--data-dir",
        str(prepared["data_dir"]),
        "score",
        str(run_dir),
        "--panels-dir",
        str(prepared["panels_dir"]),
    ]
    return argv if normalizer is None else argv + ["--normalizer", str(normalizer)]


def test_score_without_the_flag_keeps_the_runs_version(prepared, capsys):
    """Re-scoring a finished run writes the same bytes, so a summary stays a function of it."""
    run_dir = v1_run(prepared)
    assert cli.main(score_command(prepared, run_dir, None)) == 0
    summary = run_dir / "summary.json"
    first = summary.read_bytes()
    report = (run_dir / "report.md").read_bytes()
    assert json.loads(first)["run"]["normalizer_version"] == 1
    assert cli.main(score_command(prepared, run_dir, None)) == 0
    assert summary.read_bytes() == first
    assert (run_dir / "report.md").read_bytes() == report


def test_score_records_the_version_it_was_run_with(prepared, capsys):
    """A v1 run re-scored with --normalizer 2 says 2, not the version in run.json."""
    run_dir = v1_run(prepared)
    assert cli.main(score_command(prepared, run_dir, 2)) == 0
    out = capsys.readouterr().out
    assert "normalizer version used: 2" in out
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run"]["normalizer_version"] == 2
    assert "normalizer version 2" in (run_dir / "report.md").read_text(encoding="utf-8")

    # v1 stays reproducible: the same records under the older rule set say 1.
    assert cli.main(score_command(prepared, run_dir, 1)) == 0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run"]["normalizer_version"] == 1


def test_compare_defaults_to_the_current_version(prepared, capsys):
    """A comparison says the version it used; with no flag that is the current one."""
    run_dir = v1_run(prepared)
    assert (
        cli.main(
            [
                "--data-dir",
                str(prepared["data_dir"]),
                "compare",
                str(run_dir),
                str(run_dir),
                "--panels-dir",
                str(prepared["panels_dir"]),
            ]
        )
        == 0
    )
    assert "normalizer version used for these numbers: 2" in capsys.readouterr().out.lower()


def test_compare_prints_the_version_it_used(prepared, capsys):
    run_dir = v1_run(prepared)
    for version in (1, 2):
        assert (
            cli.main(
                [
                    "--data-dir",
                    str(prepared["data_dir"]),
                    "compare",
                    str(run_dir),
                    str(run_dir),
                    "--panels-dir",
                    str(prepared["panels_dir"]),
                    "--normalizer",
                    str(version),
                ]
            )
            == 0
        )
        assert f"Normalizer version used for these numbers: {version}" in capsys.readouterr().out


def test_an_unknown_normalizer_version_is_refused(prepared, capsys):
    run_dir = v1_run(prepared)
    with pytest.raises(SystemExit) as exit_info:
        cli.main(score_command(prepared, run_dir, 3))
    assert exit_info.value.code == 2
    assert "normalizer" in capsys.readouterr().err
