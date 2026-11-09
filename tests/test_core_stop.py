"""Stopping a long task: the first signal asks, the second one ends the process."""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

from audiolab import cli
from audiolab.core.stop import (
    EXIT_STOPPED,
    Stopped,
    install_signal_handlers,
    raise_if_stopped,
    request_stop,
    stop_requested,
)


def test_nothing_stops_until_a_stop_is_asked_for():
    assert not stop_requested()
    raise_if_stopped()
    request_stop()
    assert stop_requested()
    with pytest.raises(Stopped, match="run the same command again to resume"):
        raise_if_stopped("preparation stopped")


@pytest.mark.parametrize("received", [signal.SIGINT, signal.SIGTERM])
def test_the_first_signal_only_asks_for_a_stop_and_says_so(received, capsys):
    install_signal_handlers()
    os.kill(os.getpid(), received)  # handled before kill() returns, in this thread
    assert stop_requested()
    assert "stopping after the current step" in capsys.readouterr().err


def test_the_second_signal_ends_the_process_at_once():
    script = (
        "import os, signal, time\n"
        "from audiolab.core.stop import install_signal_handlers, stop_requested\n"
        "install_signal_handlers()\n"
        "os.kill(os.getpid(), signal.SIGTERM)\n"
        "assert stop_requested()\n"
        "os.kill(os.getpid(), signal.SIGTERM)\n"
        "time.sleep(5)\n"
        "print('still running')\n"
    )
    finished = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    assert finished.returncode == -signal.SIGTERM
    assert "still running" not in finished.stdout


def test_the_command_turns_a_stop_into_one_line_and_its_own_exit_code(monkeypatch, capsys):
    def stopping(args):
        raise Stopped("preparation of fleurs-fr stopped")

    monkeypatch.setattr(cli, "cmd_data_list", stopping)
    assert cli.main(["data", "list"]) == EXIT_STOPPED
    assert capsys.readouterr().err.strip() == (
        "audiolab: preparation of fleurs-fr stopped; run the same command again to resume"
    )
