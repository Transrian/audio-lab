#!/usr/bin/env python3
"""Report the characters of prepared training labels the base model cannot write.

Since 2026-11-09 every training label is written through `plain_label`
(`src/audiolab/data/train.py`), so a preparation that ran that code holds no
character outside the base model's tokenizer vocabulary. This script is the
proof, after each preparation: it reads the manifest of every named dataset
(`$AUDIOLAB_DATA/<dataset>/manifest.jsonl`), calls
`audiolab.data.train.unwritable(clip.text)` on every clip, and reports, per
dataset and per unwritable character, how many clips contain it.

A dataset with no such character prints nothing: silence is the pass.

The script is strictly read-only: it reads manifest files and prints to
standard output. Nothing under the data root is created, moved or renamed,
not even a temporary file. See the adjacent README for the question, the
reproduction command and the limits.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

from audiolab.data.store import read_manifest
from audiolab.data.train import unwritable


def train_sets(data_dir: Path) -> list[str]:
    """The prepared `*-train*` datasets, sorted by name."""
    return sorted(
        path.name for path in data_dir.iterdir() if path.is_dir() and "-train" in path.name
    )


def dataset_character_counts(data_dir: Path, dataset: str) -> Counter:
    """How many clips of one prepared dataset hold each unwritable character."""
    counts: Counter = Counter()
    for clip in read_manifest(data_dir / dataset):
        counts.update(unwritable(clip.text))
    return counts


def display(char: str) -> str:
    """The character as the table prints it: itself, or its code point."""
    return char if char.isprintable() else f"U+{ord(char):04X}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Per prepared dataset and per character, report how many of its clips "
            "hold a character the base model cannot write "
            "(`audiolab.data.train.unwritable`). A dataset with none prints nothing."
        ),
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        help="dataset names under the data root (default: every *-train* directory)",
    )
    parser.add_argument(
        "--data-dir",
        help="root of the prepared datasets (default: $AUDIOLAB_DATA)",
    )
    args = parser.parse_args(argv)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        env = os.environ.get("AUDIOLAB_DATA")
        if not env:
            parser.error("$AUDIOLAB_DATA is not set (load it: set -a; . ./.env; set +a)")
        data_dir = Path(env)
    if not data_dir.is_dir():
        print(f"error: {data_dir} is not a directory", file=sys.stderr)
        return 2

    names = args.datasets or train_sets(data_dir)
    if not names:
        print(f"error: no prepared dataset in {data_dir}", file=sys.stderr)
        return 2

    rows: list[tuple[str, str, int]] = []
    for name in names:
        try:
            counts = dataset_character_counts(data_dir, name)
        except (FileNotFoundError, ValueError) as failure:
            print(f"error: {name}: {failure}", file=sys.stderr)
            return 2
        for char, clips in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            rows.append((name, display(char), clips))

    if rows:
        print("| dataset | character | clips |")
        print("| --- | --- | --- |")
        for name, char, clips in rows:
            print(f"| {name} | `{char}` | {clips} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
