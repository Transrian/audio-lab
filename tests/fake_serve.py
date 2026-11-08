"""A stand-in for ``python -m audiolab.serve``, for the tests.

It speaks the real protocol - the ready line on standard output, then the HTTP
contract - with a transcribing function that returns a fixed text, so the
launcher, the process handling and the evaluation runner can be tested on a CPU
with no NeMo and no model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audiolab.serve.runtime import run_foreground  # noqa: E402

TEXT = "bonjour tout le monde"


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    stop_when_stdin_closes = "--stop-when-stdin-closes" in sys.argv

    def transcribe(audio: bytes, language: str = "") -> str:
        return TEXT

    from audiolab.serve.config import ServeSpec

    return run_foreground(
        ServeSpec(checkpoint="not-a-real-checkpoint.nemo"),
        "fake-model",
        port=port,
        transcribe=transcribe,
        stop_when_stdin_closes=stop_when_stdin_closes,
    )


if __name__ == "__main__":
    sys.exit(main())
