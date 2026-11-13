#!/usr/bin/env python3
"""Session overlap between the VoxPopuli evaluation set and the prepared train set.

Reproduces the figures reported in the leakage table of
``docs/stt/datasets.md``, in the card of ``voxpopuli-en-nonnative`` and in
proposed decision 0014. A *session* is the prefix of the VoxPopuli clip id
before the language tag, for example ``20090310-0900-PLENARY-7`` from
``20090310-0900-PLENARY-7-en_20090310-11:10:07_2``: clips of one plenary
sitting share the room, the microphones and the background. The population is
the prepared data, not the corpora: the 737-clip ``voxpopuli-en-nonnative``
evaluation set and the prepared 12 000-clip ``voxpopuli-fr-train`` sample.

No audio is opened and no server is contacted; the script reads the two
``manifest.jsonl`` files and prints the overlap table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from audiolab.core.paths import data_dir

#: A session is the id's prefix before the language tag. The tag is one of the
#: VoxPopuli languages, followed by the per-utterance suffix.
SESSION = re.compile(r"^([0-9]{8}-[0-9]{4}-.+?)-(en|fr|de|es|fi|it|nl)_")


def session_of(clip_id: str) -> str:
    """The session of a prepared clip id, or raise on a malformed id."""
    match = SESSION.match(clip_id.split(":", 1)[1])
    if match is None:
        raise ValueError(f"{clip_id!r} does not hold a session before a language tag")
    return match.group(1)


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def overlap(eval_rows: list[dict], train_rows: list[dict]) -> dict:
    by_session: tuple[dict[str, list[dict]], dict[str, list[dict]]] = ({}, {})
    for rows, bucket in ((eval_rows, by_session[0]), (train_rows, by_session[1])):
        for row in rows:
            bucket.setdefault(session_of(row["id"]), []).append(row)
    shared = set(by_session[0]) & set(by_session[1])
    eval_hits = [row for row in eval_rows if session_of(row["id"]) in shared]
    train_hits = [row for row in train_rows if session_of(row["id"]) in shared]
    return {
        "eval_sessions": len(by_session[0]),
        "train_sessions": len(by_session[1]),
        "shared_sessions": len(shared),
        "eval_clips": (len(eval_rows), len(eval_hits)),
        "train_clips": (len(train_rows), len(train_hits)),
        "eval_hours": sum(row["duration_s"] for row in eval_hits) / 3600.0,
        "train_hours": sum(row["duration_s"] for row in train_hits) / 3600.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="prepared datasets (default: $AUDIOLAB_DATA, else ./data)",
    )
    parser.add_argument("--eval", default="voxpopuli-en-nonnative")
    parser.add_argument("--train", default="voxpopuli-fr-train")
    args = parser.parse_args(argv)
    root = args.data_dir or data_dir()
    try:
        eval_rows = load_rows(root / args.eval / "manifest.jsonl")
        train_rows = load_rows(root / args.train / "manifest.jsonl")
    except OSError as failure:
        print(f"sessions.py: {failure}", file=sys.stderr)
        return 2
    table = overlap(eval_rows, train_rows)
    eval_total, eval_hit = table["eval_clips"]
    train_total, train_hit = table["train_clips"]
    print("session = clip id's prefix before the language tag; population = the prepared manifests")
    print(f"  {args.eval}: {len(eval_rows)} clips, {table['eval_sessions']} sessions")
    print(f"  {args.train}: {len(train_rows)} clips, {table['train_sessions']} sessions")
    print(f"shared sessions: {table['shared_sessions']} of {table['eval_sessions']}")
    print(
        f"  evaluation clips in shared sessions: {eval_hit} of {eval_total}"
        f" ({table['eval_hours']:.2f} h)"
    )
    print(
        f"  train clips in shared sessions: {train_hit} of {train_total}"
        f" ({table['train_hours']:.2f} h)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
