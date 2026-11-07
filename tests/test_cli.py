import pytest

from audiolab import cli
from audiolab.core.gpu import parse_nvidia_smi


def test_help_lists_commands(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["--help"])
    assert exit_info.value.code == 0
    assert "gpus" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert cli.main([]) == 0
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
