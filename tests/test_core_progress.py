"""Progress bars: on a terminal only, and the plain lines stay."""

from __future__ import annotations

import io

from audiolab.core import progress as progress_module
from audiolab.core.progress import progress, say


def test_a_bar_stays_off_when_standard_error_is_not_a_terminal(capsys):
    with progress(range(3), desc="clips", unit="clip") as bar:
        assert list(bar) == [0, 1, 2]
        assert bar.disable
    assert capsys.readouterr().err == ""


def test_a_bar_draws_on_a_terminal_and_counts_bytes_in_binary_units(monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    screen = Terminal()
    monkeypatch.setattr(progress_module.sys, "stderr", screen)
    with progress(desc="archive.tar.gz", unit="B", total=4 * 1024 * 1024) as bar:
        bar.update(2 * 1024 * 1024)
    assert "archive.tar.gz" in screen.getvalue()
    assert "2.00M" in screen.getvalue()


def test_the_plain_lines_of_a_task_go_to_standard_output(capsys):
    say("update 50: loss 1.2345")
    assert capsys.readouterr().out == "update 50: loss 1.2345\n"
