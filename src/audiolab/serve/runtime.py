"""Running one checkpoint as a server in this process.

This module is the child side of the pair: ``python -m audiolab.serve`` (used by
``audiolab serve`` and started by ``audiolab eval`` for a model config with a
``serve`` block) ends up in :func:`run_foreground`. It waits for a signal rather
than polling, prints one ready line so whoever started it knows the port, and
stops the server on the way out, so the GPU is never held by a forgotten
process.
"""

from __future__ import annotations

import json
import signal
import sys
import threading
from collections.abc import Callable

from audiolab.core.gpu import select_gpu
from audiolab.serve.config import ServeConfigError, ServeSpec
from audiolab.serve.http import TranscriptionServer

DEFAULT_HOST = "127.0.0.1"
# What the parent of a child server waits for on its standard output. The line
# is the only contract between the two processes.
READY_PREFIX = "audiolab-serve-ready: "
# What to tell someone who runs a local model from an environment without NeMo.
MISSING_NEMO = (
    "this environment has no NeMo, which a local checkpoint needs. It is kept out "
    "of the default install on purpose: build the local model environment with "
    "'bash scripts/nemo-venv.sh' and run the command from it "
    "(or set AUDIOLAB_NEMO_PYTHON so 'audiolab eval' starts the server itself)"
)


def build_transcriber(
    spec: ServeSpec, transcribe: Callable[[bytes, str], str] | None = None
) -> Callable[[bytes, str], str]:
    """The function that answers requests: the backend's, or one a caller supplies.

    The GPU is selected here, before the backend imports torch, because
    ``CUDA_VISIBLE_DEVICES`` must be set before CUDA is initialised (D-008).
    """
    if transcribe is not None:
        return transcribe
    if spec.backend != "nemo":
        raise ServeConfigError(f"unknown serve backend {spec.backend!r}")
    select_gpu(spec.gpu)
    try:
        from audiolab.serve.nemo import transcriber
    except ImportError as failure:
        raise ServeConfigError(MISSING_NEMO) from failure
    return transcriber(spec)


def served_identity(spec: ServeSpec) -> dict | None:
    """What this server holds, for ``GET /v1/models``.

    It is read from the checkpoint itself rather than assumed, so whoever started
    this server can compare it with the checkpoint that was meant to be served. A
    checkpoint that cannot be identified - a test double, or a file that moved
    between the two processes - is served without one: the side that started the
    server knows what it asked for, and it refuses an identity it cannot check
    instead of trusting the port.
    """
    try:
        return spec.identity()
    except ServeConfigError:
        return None


def announce(base_url: str) -> None:
    """Tell whoever started this process which address is being served."""
    payload = {"base_url": base_url}
    print(f"{READY_PREFIX}{json.dumps(payload)}", flush=True)


def run_foreground(
    spec: ServeSpec,
    model_id: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = 0,
    transcribe: Callable[[bytes, str], str] | None = None,
    stop_when_stdin_closes: bool = False,
) -> int:
    """Load the checkpoint, serve it, and return when told to stop.

    ``port`` 0 lets the operating system choose a free port, which is then
    announced; nothing has to be guessed or checked in advance.
    """
    handler = build_transcriber(spec, transcribe)
    server = TranscriptionServer(
        handler, model_id, host=host, port=port, identity=served_identity(spec)
    )
    stop = threading.Event()
    _watch_signals(stop)
    if stop_when_stdin_closes:
        # Whoever started this process holds the read end of the pipe: when it
        # dies, even without cleaning up, the write end closes and the server
        # stops instead of holding a GPU for nobody.
        threading.Thread(target=_stop_when_stdin_closes, args=(stop,), daemon=True).start()
    server.start()
    announce(server.base_url)
    try:
        stop.wait()
    finally:
        server.stop()
    return 0


def _watch_signals(stop: threading.Event) -> None:
    """Stop on the signals a process is asked to stop with; Ctrl+C is one of them."""

    def request_stop(*_) -> None:
        stop.set()

    for received in (signal.SIGINT, signal.SIGTERM):
        signal.signal(received, request_stop)


def _stop_when_stdin_closes(stop: threading.Event) -> None:
    sys.stdin.read()
    stop.set()
