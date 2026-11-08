"""Start one checkpoint as a server, in the environment that has NeMo.

``audiolab eval`` runs from the light environment (no PyTorch, no NeMo), so it
starts the server as a child process of the interpreter that has them: the one
``$AUDIOLAB_NEMO_PYTHON`` names, by default the interpreter running the command
(see `requirements/nemo.txt`). The child announces the address it serves on its
standard output, which is the whole contract between the two.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from queue import Empty, Queue

from audiolab.serve.config import ServeConfigError, ServeSpec
from audiolab.serve.runtime import DEFAULT_HOST, READY_PREFIX

# Where to look for the interpreter that can load a checkpoint.
NEMO_PYTHON_ENV = "AUDIOLAB_NEMO_PYTHON"
# Loading a checkpoint of a few gigabytes from disk takes a while; this is the
# ceiling, not the expected time.
READY_TIMEOUT_S = 900.0
# How long a server may take to stop, first when asked nicely (its standard
# input closes), then when told to terminate, before it is killed.
STOP_TIMEOUT_S = 30.0


def serving_python() -> str:
    """The interpreter that can import NeMo: the one the variable names, or this one."""
    return os.environ.get(NEMO_PYTHON_ENV, "").strip() or sys.executable


def serve_command(
    spec: ServeSpec, model_id: str, *, host: str = DEFAULT_HOST, port: int = 0
) -> list[str]:
    """The command that serves one checkpoint; ``port`` 0 means "choose any".

    The checkpoint is resolved here rather than in the child, so the child never
    reads the environment for it and the two processes cannot disagree about
    which file is being served.
    """
    checkpoint = spec.resolved()
    command = [
        serving_python(),
        "-m",
        "audiolab.serve",
        "--model-id",
        model_id,
        "--backend",
        spec.backend,
        "--gpu",
        spec.gpu,
        "--max-symbols",
        str(spec.max_symbols),
        "--host",
        host,
        "--port",
        str(port),
        # The child stops when its standard input ends: a parent killed without
        # cleaning up must not leave a checkpoint holding the GPU.
        "--stop-when-stdin-closes",
    ]
    command += ["--checkpoint", str(checkpoint)]
    return command


class ServingProcess:
    """A server running as a child process, and the address it announced."""

    def __init__(self, command: list[str]):
        self.command = [str(part) for part in command]
        self.base_url = ""
        self._lines: deque[str] = deque(maxlen=40)
        self._queue: Queue[str | None] = Queue()
        self.process = subprocess.Popen(  # noqa: S603 - the command is built here, not by a user
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._read_output, daemon=True).start()

    def _read_output(self) -> None:
        """Keep reading the child's output so a full pipe can never block it."""
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self._lines.append(line.rstrip("\n"))
            self._queue.put(line)
        self._queue.put(None)

    def wait_until_ready(self, timeout_s: float = READY_TIMEOUT_S) -> str:
        """Wait for the child's ready line; return the address it serves."""
        while True:
            try:
                line = self._queue.get(timeout=timeout_s)
            except Empty:
                self.stop()
                raise ServeConfigError(
                    f"the server for {self.command[0]} did not start within "
                    f"{timeout_s:.0f}s; its last output was:\n{self.output()}"
                ) from None
            if line is None:
                raise ServeConfigError(
                    f"the server stopped before it was ready (command: {' '.join(self.command)})"
                    f"; its output was:\n{self.output()}"
                )
            if line.startswith(READY_PREFIX):
                self.base_url = json.loads(line[len(READY_PREFIX) :])["base_url"]
                return self.base_url

    def output(self) -> str:
        """The last lines the child printed, for an error message."""
        return "\n".join(self._lines)

    def stop(self) -> None:
        """Stop the server: ask it, then tell it, then kill it. Safe to call twice."""
        if self.process.poll() is not None:
            return
        # Closing the pipe is the polite way: the child stops and releases the GPU.
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=STOP_TIMEOUT_S)
            return
        except subprocess.TimeoutExpired:
            self.process.terminate()
        try:
            self.process.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=STOP_TIMEOUT_S)


def start_server(
    spec: ServeSpec,
    model_id: str,
    *,
    command_factory: Callable[..., list[str]] | None = None,
    timeout_s: float = READY_TIMEOUT_S,
) -> ServingProcess:
    """Load a checkpoint in a child process and wait until it answers.

    ``command_factory`` replaces the command; the tests use it to exercise this
    side of the protocol without NeMo.
    """
    factory = command_factory or serve_command
    process = ServingProcess(factory(spec, model_id))
    process.wait_until_ready(timeout_s)
    return process


@contextmanager
def serving(
    spec: ServeSpec,
    model_id: str,
    *,
    command_factory: Callable[..., list[str]] | None = None,
) -> Iterator[str]:
    """Yield the address of a server for this checkpoint, and stop it afterwards."""
    process = start_server(spec, model_id, command_factory=command_factory)
    try:
        yield process.base_url
    finally:
        process.stop()
