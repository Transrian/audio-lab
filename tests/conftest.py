"""What every test starts from: no stop asked for, and the default signal handlers."""

from __future__ import annotations

import signal
import threading

import pytest

from audiolab.core import stop


@pytest.fixture(autouse=True)
def no_stop_requested(monkeypatch):
    """A stop asked for by one test, or handlers it installed, never reach the next."""
    monkeypatch.setattr(stop, "_requested", threading.Event())
    handlers = {received: signal.getsignal(received) for received in stop.SIGNALS}
    yield
    for received, handler in handlers.items():
        signal.signal(received, handler)


@pytest.fixture
def run_again(monkeypatch):
    """Call it to start what follows as the next run of a stopped command would: no stop asked."""
    return lambda: monkeypatch.setattr(stop, "_requested", threading.Event())
