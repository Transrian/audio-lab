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

    with pytest.raises(SystemExit):
        cli.main([*command, "--concurrency", "0"])
    assert "at least 1" in capsys.readouterr().err


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
