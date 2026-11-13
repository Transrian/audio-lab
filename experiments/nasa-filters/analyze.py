#!/usr/bin/env python3
"""The facts behind the NASA filter review and the re-screen, in five tables.

The NASA screen of 2026-11-09 was mixed, and two questions were left open:
what NASA's Curious Universe show is worth, and what its edge-word gate
silently drops. This script answers both with the files that already exist,
so nothing is re-run and nothing under the data, runs or cache roots is
touched:

1. per show: the episodes, clips, hours and mean ``base_wer`` of the current
   ``nasa-en-train`` preparation;
2. Curious Universe: how many of its 7 listed episodes are in that set, and
   what share of the set they are;
3. the first screen's hallucinations: every non-empty answer of the run on
   the ``nonspeech`` panel, with the category of the clip it was said on, and
   how many answers look like podcast speech;
4. the edge-word gate: per show, how many clips Parakeet was asked about at
   the preparation (its answers are in the per-clip cache) and how many the
   gate dropped, and what is recorded about the dropped clips;
5. characters: how many clips of the current set hold a character the base
   model cannot write (the same check as
   [label-text](../label-text/check_text.py): ``audiolab.data.train.unwritable``).

The script is strictly read-only: it reads manifests, the Parakeet per-clip
answer cache, one evaluation run's records and prints to standard output.
It opens no audio, sends no request, starts no server, and writes nothing
under the data, runs or cache roots, not even a temporary file. See the
adjacent README for the environment, the command and the limits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from audiolab.data.store import read_manifest
from audiolab.data.train import unwritable

# The source's episodes, as the manifest writes them: "nasa-en-train:<slug>".
GROUP_PREFIX = "nasa-en-train:"
# The clip's start, in centiseconds from the episode's start: the last part of
# its id, as the preparation writes it. The clip the edge gate asked about is
# cached under that start in seconds with two decimals, so the two are the same
# number.
KEY = re.compile(r"^(\d+\.\d{2})-(\d+\.\d{2})$")
# What a word is for the speech-likeness count: a token with a letter (the
# unknown token and stray symbols are not words).
WORD = re.compile(r"[a-z]")


def show_of(author: str) -> str:
    """The show's name, out of the manifest's "NASA, <show>" attribution."""
    return author.removeprefix("NASA, ")


def clip_start_cs(clip_id: str) -> int:
    """The clip's start in centiseconds, the last hyphen-separated part of its id."""
    return int(clip_id.rsplit("-", 1)[1])


# Table 1 and 2: the current set, per show.


def show_rows(clips) -> list[tuple[str, dict]]:
    """Per show: its episodes, clips, hours and mean base_wer in one set."""
    per_show: dict[str, dict] = {}
    for clip in clips:
        row = per_show.setdefault(
            show_of(clip.source_author),
            {"episodes": set(), "clips": 0, "seconds": 0.0, "base_wer": 0.0},
        )
        row["episodes"].add(clip.group)
        row["clips"] += 1
        row["seconds"] += clip.duration_s
        row["base_wer"] += clip.base_wer or 0.0
    rows = []
    for show, row in per_show.items():
        rows.append(
            (
                show,
                {
                    "episodes": len(row["episodes"]),
                    "clips": row["clips"],
                    "seconds": row["seconds"],
                    "base_wer": row["base_wer"],
                },
            )
        )
    return rows


def table_by_show(clips) -> str:
    rows = sorted(show_rows(clips), key=lambda item: -item[1]["clips"])
    lines = [
        "## 1. The current set, per show",
        "",
        "| show | episodes | clips | hours | mean base_wer |",
        "| --- | --- | --- | --- | --- |",
    ]
    for show, row in rows:
        mean_wer = row["base_wer"] / row["clips"]
        lines.append(
            f"| {show} | {row['episodes']} | {row['clips']} | {row['seconds'] / 3600:.1f} "
            f"| {100 * mean_wer:.1f} % |"
        )
    return "\n".join(lines)


def table_curious(clips) -> str:
    total = len(clips)
    curious = [clip for clip in clips if "curious" in show_of(clip.source_author).lower()]
    episodes = {clip.group for clip in curious}
    lines = [
        "## 2. Curious Universe in the current set",
        "",
        "| episodes | clips | share of the set's clips |",
        "| --- | --- | --- |",
        f"| {len(episodes)} | {len(curious)} "
        f"| {100 * len(curious) / total:.2f} % ({len(curious)} of {total}) |",
    ]
    return "\n".join(lines)


# Table 3: the first screen's hallucinations.


def nonempty_answers(records: list[dict], panel: str) -> list[dict]:
    """The run's non-empty answers on one panel, newest record per item."""
    latest = {}
    for record in records:
        latest[record["id"]] = record
    return [
        record
        for record in sorted(latest.values(), key=lambda record: record["id"])
        if record["panel"] == panel and record.get("hypothesis", "").strip()
    ]


def words(text: str) -> list[str]:
    """The answer's words: its lowercased tokens, lettered, edge punctuation off.

    The unknown token and stray symbols are not words: the scorer reads them
    as word breaks, and they should not make an interjection look like speech.
    "Yeah," and "yeah." are one word, so the punctuation goes before the count.
    """
    return [token.strip(".,;:!?…").lower() for token in text.split() if WORD.search(token.lower())]


def speech_like(text: str) -> bool:
    """Whether an answer looks like podcast speech rather than an interjection.

    A bare "Oh", a repeated "Yeah, yeah, yeah." or a siren read as "BIT! BIT!"
    is a sound the model made of nothing; a stretch of at least three words
    with two different words in it ("So they use the two microphones.") is the
    shape of the podcast speech the run learns, so it counts as speech.
    """
    seen = words(text)
    return len(seen) >= 3 and len(set(seen)) >= 2


def table_hallucinations(records: list[dict], categories: dict[str, str], panel: str) -> str:
    answers = nonempty_answers(records, panel)
    lines = [
        f"## 3. The first screen's non-empty answers on {panel}",
        "",
        f"{len(answers)} non-empty answers (the hallucination rate is this "
        "over the panel's clips).",
        "",
        "| category | answer | looks like podcast speech |",
        "| --- | --- | --- |",
    ]
    for record in answers:
        category = categories.get(record["id"], "?")
        mark = "yes" if speech_like(record["hypothesis"]) else ""
        lines.append(f"| {category} | {record['hypothesis']!r} | {mark} |")
    like = sum(speech_like(record["hypothesis"]) for record in answers)
    with_token = sum("⁇" in record["hypothesis"] for record in answers)
    lines += [
        "",
        f"{like} of the {len(answers)} answers look like podcast speech "
        "(three or more words with two of them different); "
        f"{with_token} hold the tokenizer's unknown token `⁇`.",
    ]
    return "\n".join(lines)


# Table 4: the edge gate, from the Parakeet per-clip answer cache.


def cache_starts(path: Path) -> set[int]:
    """The clip starts (centiseconds) of one episode's cached answers."""
    starts = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        key = json.loads(line)["key"]
        match = KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"{path}:{number}: the key {key!r} is not a clip stretch")
        starts.add(round(float(match.group(1)) * 100))
    return starts


def table_edge_gate(cache_dir: Path, suffix: str, clips, old_clips) -> str:
    """Per show: the clips the gate asked about, kept, and dropped.

    Every labelled clip of a preparation is asked of the base model at its
    edges, and the answer is cached under the clip's start. A kept clip is in
    the set the preparation wrote; a clip that is in the cache but in no set
    the gate wrote is one the gate dropped. The sets compared are the first
    (typographic) preparation and the current one, so the clips lost when the
    labels were rewritten are not counted as gate drops.
    """
    episode_of_clip = {
        clip.group.removeprefix(GROUP_PREFIX): show_of(clip.source_author)
        for clip in list(clips) + list(old_clips)
    }
    new_starts: Counter = Counter()
    for clip in clips:
        new_starts[(clip.group.removeprefix(GROUP_PREFIX), clip_start_cs(clip.id))] += 1
    old_starts: Counter = Counter()
    for clip in old_clips:
        old_starts[(clip.group.removeprefix(GROUP_PREFIX), clip_start_cs(clip.id))] += 1

    new_by_slug: dict[str, Counter] = {}
    for slug, start in new_starts:
        new_by_slug.setdefault(slug, Counter())[start] += 1
    old_by_slug: dict[str, Counter] = {}
    for slug, start in old_starts:
        old_by_slug.setdefault(slug, Counter())[start] += 1

    asked_by_episode: dict[str, set[int]] = {}
    row_by_show: dict[str, dict] = {}
    for path in sorted(cache_dir.glob(f"*{suffix}")):
        slug = path.name[: -len(suffix)]
        show = episode_of_clip.get(slug, "(episode with no kept clip)")
        asked = cache_starts(path)
        asked_by_episode[slug] = asked
        # A clip of the first set passed the gate, so its answer is cached: a
        # kept start with no cache line is a broken preparation, not a count.
        for start in old_by_slug.get(slug, ()):
            if start not in asked:
                raise ValueError(f"{slug}: the kept clip at {start} cs has no cached answer")
        kept_old = sum(start in asked for start in old_by_slug.get(slug, ()))
        dropped = len(asked) - kept_old
        row = row_by_show.setdefault(
            show,
            {
                "episodes": 0,
                "asked": 0,
                "kept_old": 0,
                "gate_dropped": 0,
                "rewrite_dropped": 0,
            },
        )
        row["episodes"] += 1
        row["asked"] += len(asked)
        row["kept_old"] += kept_old
        row["gate_dropped"] += dropped

    # Clips of the first set the current set does not hold: the label rewrite
    # dropped them (a character still outside the vocabulary, or an edge word
    # the rewritten label does not hear). Their answers stay in the cache.
    unmatched: list[str] = []
    for slug, starts in old_by_slug.items():
        show = episode_of_clip.get(slug, "(episode with no kept clip)")
        for start, count in starts.items():
            if new_starts.get((slug, start), 0):
                continue
            if start not in asked_by_episode.get(slug, set()):
                unmatched.append(f"{slug}:{start}")
                continue
            row_by_show[show]["rewrite_dropped"] += count
    for slug, starts in new_by_slug.items():
        for start in starts:
            if start not in asked_by_episode.get(slug, set()):
                unmatched.append(f"{slug}:{start}")

    rows = list(row_by_show.items())
    lines = [
        "## 4. The edge-word gate, per show",
        "",
        "Every labelled clip of the first preparation was asked of the base model at its",
        "edges, and every answer is in the per-clip cache. `asked` counts those answers;",
        "`gate-dropped` the clips they left out of the first set; `rewrite-dropped` the",
        "clips the first set held that the current set does not (the label rewrite).",
        "",
        "| show | episodes | asked | kept (first set) | gate-dropped | rewrite-dropped |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for show, row in sorted(rows, key=lambda item: -item[1]["gate_dropped"]):
        lines.append(
            f"| {show} | {row['episodes']} | {row['asked']} | {row['kept_old']} "
            f"| {row['gate_dropped']} | {row['rewrite_dropped']} |"
        )
    asked = sum(row["asked"] for row in row_by_show.values())
    dropped = sum(row["gate_dropped"] for row in row_by_show.values())
    rewritten = sum(row["rewrite_dropped"] for row in row_by_show.values())
    lines += [
        "",
        f"{dropped} of the {asked} asked clips were dropped by the gate "
        f"({100 * dropped / asked:.1f} %); {rewritten} clips were lost in the label rewrite.",
        "The dropped clips' `base_wer` is not recorded: the cache holds the base model's",
        "raw text for each asked clip but not its label, so the error of a dropped clip",
        "cannot be computed from the existing files. Re-running the base model to get it",
        "is what this analysis must not do.",
    ]
    if unmatched:
        lines.append(
            f"NOTE: {len(unmatched)} clips have no matching cached start (first: {unmatched[:3]})"
        )
    return "\n".join(lines)


# Table 5: the characters of the current set's labels.


def table_characters(clips, dataset: str) -> str:
    counts: Counter = Counter()
    for clip in clips:
        counts.update(unwritable(clip.text))
    if not counts:
        return (
            f"## 5. Characters of {dataset}'s labels\n\n"
            "0 clips hold a character the base model cannot write."
        )
    lines = [
        f"## 5. Characters of {dataset}'s labels",
        "",
        "| character | clips |",
        "| --- | --- |",
    ]
    for char, number in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{char}` | {number} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "The five tables of the NASA filter review: the current set per show, "
            "Curious Universe's share, the first screen's hallucinations, the edge "
            "gate's drops per show, and the current set's unwritable characters. "
            "Strictly read-only: manifests, the Parakeet per-clip cache and one "
            "evaluation run's records, printed as markdown."
        )
    )
    parser.add_argument(
        "--data-dir",
        help="root of the prepared datasets (default: $AUDIOLAB_DATA)",
    )
    parser.add_argument(
        "--runs-dir",
        help="root of the run outputs (default: $AUDIOLAB_RUNS)",
    )
    parser.add_argument(
        "--cache-dir",
        help="the download cache (default: $AUDIOLAB_CACHE)",
    )
    parser.add_argument(
        "--dataset",
        default="nasa-en-train",
        help="the current prepared set (default: nasa-en-train)",
    )
    parser.add_argument(
        "--old-dataset",
        default="nasa-en-train.curly-2026-11-09",
        help="the first, typographic, preparation (default: nasa-en-train.curly-2026-11-09)",
    )
    parser.add_argument(
        "--eval-run",
        help="the first screen's evaluation run directory (default: "
        "<runs-dir>/eval/screen-nasa-en/full)",
    )
    parser.add_argument(
        "--panel",
        default="nonspeech-full",
        help="the non-speech panel of that run (default: nonspeech-full)",
    )
    parser.add_argument(
        "--base-model",
        default="stt-c-parakeet-tdt-0.6B-v3",
        help="the base model whose per-clip answers are cached "
        "(default: stt-c-parakeet-tdt-0.6B-v3)",
    )
    args = parser.parse_args(argv)

    resolved = {}
    for variable, attr in (
        ("AUDIOLAB_DATA", "data_dir"),
        ("AUDIOLAB_RUNS", "runs_dir"),
        ("AUDIOLAB_CACHE", "cache_dir"),
    ):
        value = getattr(args, attr)
        if value is None:
            value = os.environ.get(variable)
            if not value:
                parser.error(f"${variable} is not set (load it: set -a; . ./.env; set +a)")
        resolved[attr] = value
    data_dir, runs_dir, cache_dir = (
        Path(resolved["data_dir"]),
        Path(resolved["runs_dir"]),
        Path(resolved["cache_dir"]),
    )

    try:
        clips = read_manifest(data_dir / args.dataset)
        old_clips = read_manifest(data_dir / args.old_dataset)
        categories = {clip.id: clip.category for clip in read_manifest(data_dir / "nonspeech")}
    except (FileNotFoundError, ValueError) as failure:
        print(f"analyze: {failure}", file=sys.stderr)
        return 2

    if args.eval_run:
        eval_run = Path(args.eval_run)
    else:
        eval_run = runs_dir / "eval" / "screen-nasa-en" / "full"
    records_path = eval_run / "records.jsonl"
    if not records_path.is_file():
        print(f"analyze: {records_path} is not a finished evaluation run", file=sys.stderr)
        return 2
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    clip_cache = cache_dir / args.dataset
    suffix = f".{args.base_model}.clips.jsonl"
    if not any(clip_cache.glob(f"*{suffix}")):
        print(f"analyze: no {suffix} files in {clip_cache}", file=sys.stderr)
        return 2

    print(table_by_show(clips))
    print()
    print(table_curious(clips))
    print()
    print(table_hallucinations(records, categories, args.panel))
    print()
    print(table_edge_gate(clip_cache, suffix, clips, old_clips))
    print()
    print(table_characters(clips, args.dataset))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
