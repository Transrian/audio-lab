"""``python -m audiolab.serve``: run one checkpoint as a server.

``audiolab eval`` starts this for a model config with a ``serve`` block, in the
interpreter that has NeMo (``$AUDIOLAB_NEMO_PYTHON``). It can also be run by
hand: the options are deliberately the resolved facts of one checkpoint, so the
command line of a running GPU process says what it is serving.
"""

from __future__ import annotations

import argparse
import sys

from audiolab.serve.config import ServeConfigError, ServeSpec
from audiolab.serve.runtime import DEFAULT_HOST, run_foreground


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m audiolab.serve",
        description="Serve one local checkpoint through an OpenAI-compatible API.",
        epilog="Started by 'audiolab eval' for a model config with a 'serve' block; "
        "in normal use, run 'audiolab serve --model configs/models/<name>.yaml' instead.",
    )
    parser.add_argument("--model-id", required=True, help="the id GET /v1/models reports")
    parser.add_argument("--backend", default="nemo", help="backend to load the checkpoint with")
    parser.add_argument("--checkpoint", required=True, help="a .nemo path or a Hub id")
    parser.add_argument("--gpu", default="0", help="GPU to load the model on: index or UUID prefix")
    parser.add_argument("--max-symbols", type=int, default=10, help="greedy decoding symbol budget")
    parser.add_argument("--host", default=DEFAULT_HOST, help="address to listen on")
    parser.add_argument("--port", type=int, default=0, help="port to listen on; 0 picks a free one")
    parser.add_argument(
        "--stop-when-stdin-closes",
        action="store_true",
        help="stop when standard input ends, so a crashed caller leaves no GPU busy",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = ServeSpec(
        backend=args.backend,
        checkpoint=args.checkpoint,
        gpu=args.gpu,
        max_symbols=args.max_symbols,
    )
    try:
        return run_foreground(
            spec,
            args.model_id,
            host=args.host,
            port=args.port,
            stop_when_stdin_closes=args.stop_when_stdin_closes,
        )
    except ServeConfigError as failure:
        print(f"{build_parser().prog}: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
