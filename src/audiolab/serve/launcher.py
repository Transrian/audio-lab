"""Start a temporary transcription server for one checkpoint (D-016).

The server runs in the calling process: it holds one model, the process already
has the GPU that model was loaded on, and a crash of the caller takes the server
with it instead of leaving a process holding the card. ``audiolab serve`` keeps
it in the foreground, ``audiolab eval`` uses :func:`serving` around a suite.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from audiolab.core.gpu import select_gpu
from audiolab.serve.config import ServeConfigError, ServeSpec
from audiolab.serve.http import TranscriptionServer


def build_transcriber(
    spec: ServeSpec, transcribe: Callable[[bytes, str], str] | None = None
) -> Callable[[bytes, str], str]:
    """The function that answers requests: the backend's, or one a caller supplies.

    A caller may pass its own function, which is how the tests exercise the
    whole path (HTTP layer, cache, runner) without NeMo; nothing in the commands
    does.
    """
    if transcribe is not None:
        return transcribe
    if spec.backend != "nemo":
        raise ServeConfigError(f"unknown serve backend {spec.backend!r}")
    from audiolab.serve.nemo import transcriber  # imported here: it loads torch

    return transcriber(spec)


def serve(
    spec: ServeSpec,
    model_id: str,
    *,
    transcribe: Callable[[bytes, str], str] | None = None,
) -> TranscriptionServer:
    """Load the checkpoint on the configured GPU and bind a port for it.

    The port is bound before this returns, so a caller never has to wait for the
    server to be ready. ``spec.gpu`` is applied through
    :func:`audiolab.core.gpu.select_gpu`, which sets ``CUDA_VISIBLE_DEVICES``
    before the backend imports torch.
    """
    if transcribe is None:
        select_gpu(spec.gpu)
    handler = build_transcriber(spec, transcribe)
    return TranscriptionServer(handler, model_id)


@contextmanager
def serving(
    spec: ServeSpec,
    model_id: str,
    *,
    transcribe: Callable[[bytes, str], str] | None = None,
) -> Iterator[str]:
    """Yield the base URL of a running server, and stop it whatever happens."""
    server = serve(spec, model_id, transcribe=transcribe).start()
    try:
        yield server.base_url
    finally:
        server.stop()
