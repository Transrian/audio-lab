"""The ``audiolab`` command.

Each area adds its subcommands here. A subcommand is a function that receives the
parsed arguments and returns an exit code.
"""

from __future__ import annotations

import argparse
import sys

from audiolab import __version__
from audiolab.core.gpu import list_gpus


def cmd_gpus(args: argparse.Namespace) -> int:
    gpus = list_gpus()
    if not gpus:
        print("No NVIDIA GPU found.")
        return 1
    print(f"{'index':<6} {'memory':>9}  {'name':<28} uuid")
    for gpu in gpus:
        memory = f"{gpu.memory_mib / 1024:.0f} GiB"
        print(f"{gpu.index:<6} {memory:>9}  {gpu.name:<28} {gpu.uuid}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiolab",
        description="A lab for speech experiments.",
    )
    parser.add_argument("--version", action="version", version=f"audiolab {__version__}")
    commands = parser.add_subparsers(title="commands", metavar="<command>")

    gpus = commands.add_parser("gpus", help="list the GPUs that --gpu can select")
    gpus.set_defaults(func=cmd_gpus)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
