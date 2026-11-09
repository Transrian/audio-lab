"""Stopping a long task between two steps, so that running it again goes on (D-021).

The first Ctrl-C or SIGTERM does not interrupt anything: it asks for a stop,
and the task stops at its next step boundary - after the clip it is writing, the
block it is downloading, the gradient update it is computing - with what it did
saved. The second one stops the process at once, the way the signal does when
nothing handles it.

A task asks :func:`stop_requested` at its boundaries, or calls
:func:`raise_if_stopped`, which raises :class:`Stopped`; the ``audiolab``
command turns that into one line and exit code :data:`EXIT_STOPPED`.
"""

from __future__ import annotations

import os
import signal
import sys
import threading

# The exit code of a task that stopped because it was asked to: 128 + SIGINT,
# what a shell reports for a command stopped with Ctrl-C.
EXIT_STOPPED = 130
SIGNALS = (signal.SIGINT, signal.SIGTERM)

_requested = threading.Event()
# The process that installed the handlers. Worker processes forked from it
# inherit them, and only this one speaks.
_installer_pid = 0


class Stopped(Exception):
    """A task stopped at a step boundary because a stop was asked for."""

    def __init__(self, what: str = "stopped") -> None:
        super().__init__(f"{what}; run the same command again to resume")


def request_stop() -> None:
    """Ask every long task of this process to stop at its next step boundary."""
    _requested.set()


def stop_requested() -> bool:
    """Whether a stop was asked for."""
    return _requested.is_set()


def raise_if_stopped(what: str = "stopped") -> None:
    """Raise :class:`Stopped` if a stop was asked for; call it at a step boundary."""
    if stop_requested():
        raise Stopped(what)


def install_signal_handlers() -> None:
    """Make the first SIGINT or SIGTERM a stop request, and the second an immediate exit.

    Calling it again puts the handlers back after a library replaced them
    (PyTorch Lightning installs its own SIGTERM handler while it trains).
    """
    global _installer_pid
    _installer_pid = _installer_pid or os.getpid()
    for received in SIGNALS:
        signal.signal(received, _on_signal)


def _on_signal(signum: int, _frame) -> None:
    if not _requested.is_set():
        request_stop()
        if os.getpid() == _installer_pid:
            print(
                f"\n{signal.Signals(signum).name}: stopping after the current step "
                "(send it again to stop at once)",
                file=sys.stderr,
                flush=True,
            )
        return
    # The second time, the signal does what it does when nothing handles it.
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
