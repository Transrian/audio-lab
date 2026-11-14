#!/usr/bin/env python3
"""Measure the label error of the agreement pipeline on clips that also hold a human label.

The talk pipeline keeps a clip when Whisper and Qwen3-ASR agree within
``MAX_DISAGREEMENT`` of the normalized words, and its label is the Qwen text
(``plain_label`` of it). That error was never measured: no talk source has a
human label. Two prepared sets do — ``nasa-en-train`` (NASA's own
near-verbatim transcripts) and ``peertube-edu-captions-fr`` (human captions) —
and this script measures the pipeline on a fixed sample of each:

- ``transcribe`` draws the sample (``random.Random(20261110).sample`` over the
  manifest's ids sorted, written to the cache so the draw is fixed), then asks
  Whisper for the transcript of every sampled clip's own audio, and Qwen after,
  every request of one model before the other's (D-020), as many at once as the
  model takes (D-022). Every answer is cached per clip under
  ``$AUDIOLAB_CACHE/label-audit/<model>/<clip id>.json`` (D-017), keyed on the
  model id, the language, the request and the audio's sha256 — a wrong reuse is
  a bug, and ``tests/test_data_label_audit.py`` proves a changed key is asked
  again. ``audiolab.data.relabel`` (the ``nasa-en-train-qwen`` fetcher) reads
  the same Qwen answers, so a relabel never asks for what the audit already
  asked.
- ``score`` applies the pipeline's agreement rule — imported from
  ``audiolab.data.talks``, not reimplemented — to decide which clips it would
  have kept, and scores the kept and the rejected against the human label with
  the project's scorer (``audiolab.eval.metrics.counts``, the normalized view,
  normalizer version ``NORMALIZER_VERSION``): the WER of the kept labels, the
  share the rule rejects and the WER of those, and the same split by
  ``base_wer`` bucket (the English arm only — the French arm has none).
- ``examples`` classifies the word-level differences of the kept clips into
  filler insertions, numbers, name spellings and other, and prints 30 of them
  so a reader can judge how much of the measured error is the style difference
  between a model label and a human transcript, not a word heard wrong. The
  differences are in the normalized view — what the WER counts — and the
  normalizer already drops the bare interjections (``uh``, ``um``, ``hmm``)
  and reads numbers as words, so a filler example is a discourse word the
  human transcript did not write, and a number example is a number read
  differently.

Everything is read-only on the prepared datasets and the models' answers; the
only writes are to the audit's own cache directory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from audiolab.core.audio import load_audio
from audiolab.core.parallel import map_requests
from audiolab.core.paths import cache_dir, data_dir
from audiolab.core.progress import say
from audiolab.core.stop import Stopped, raise_if_stopped, stop_requested
from audiolab.data.label_audit import (
    SAMPLE_SEED,
    answer_path,
    read_answer,
    read_sample,
    sample_path,
    write_answer,
    write_sample,
)
from audiolab.data.store import Clip, read_manifest
from audiolab.data.talk_clips import wav_bytes
from audiolab.data.talks import (
    LABEL_MODEL,
    MAX_DISAGREEMENT,
    SEGMENT_MODEL,
    client,
    disagreement,
    served,
)
from audiolab.data.train import plain_label
from audiolab.eval.metrics import counts, tokens
from audiolab.eval.normalize import NORMALIZER_VERSION, normalize

# What one arm measures: how many clips, heard in which language, against whose
# human text. The draw and the models are the same for every arm.
AUDITS = {
    "nasa-en-train": {
        "count": 3000,
        "language": "en",
        "human": "NASA's own near-verbatim transcripts",
    },
    "peertube-edu-captions-fr": {
        "count": 1000,
        "language": "fr",
        "human": "the ministry's human captions",
    },
}
# The `base_wer` buckets: how the base model did on the clip's human label,
# fixed here so a re-run of the audit reads them the same way. The French arm
# has no `base_wer` and is not split.
BASE_WER_BUCKETS = (
    ("base_wer < 0.1", 0.0, 0.1),
    ("0.1 <= base_wer < 0.2", 0.1, 0.2),
    ("0.2 <= base_wer < 0.3", 0.2, 0.3),
    ("base_wer >= 0.3", 0.3, float("inf")),
)
# The word-level categories of the examples, in the order they are checked.
CATEGORIES = ("filler insertions", "numbers", "name spellings", "other")
# A discourse word the human transcript dropped and the model writes. The bare
# interjections (uh, um, ...) are already dropped by the normalizer, so these
# are the fillers the WER still sees. Checked on the token itself, lowercased.
FILLERS = frozenset(
    "yeah well okay ok so like you know mean just actually basically "
    "literally kinda sorta right gonna wanna gotta euh heu hum".split()
)
# What a number is read as, in the normalized view, in the two languages of the
# audit. A number difference is one of these on either side.
NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty "
    "thirty forty fifty sixty seventy eighty ninety hundred thousand million "
    "billion percent un deux trois quatre cinq six sept huit neuf dix onze "
    "douze cent mille vingt trente quarante cinquante soixante quatre-vingt "
    "pour".split()
)
# The examples a reader can judge: at most 10 of one category, 30 in all.
EXAMPLES_PER_CATEGORY = 10
TOTAL_EXAMPLES = 30
# A kept clip this many words or fewer is shown whole in the examples; a
# longer one is cut at that word.
EXAMPLE_MAX_WORDS = 30
# A capitalized word at one of these points is the start of a sentence, not a
# name.
SENTENCE_ENDS = (".", "?", "!", "…", ")", "]")


def audit_dataset(dataset: str) -> dict:
    if dataset not in AUDITS:
        raise SystemExit(f"unknown audit dataset {dataset!r}; known: {', '.join(sorted(AUDITS))}")
    return AUDITS[dataset]


def sampled_clips(dataset: str, spec: dict, root: Path) -> list[Clip]:
    """The fixed sample of the dataset, as manifest clips, in the draw's order."""
    clips = {clip.id: clip for clip in read_manifest(root / dataset)}
    path = sample_path(cache_dir(), dataset)
    sample = read_sample(
        path,
        dataset=dataset,
        count=spec["count"],
        seed=SAMPLE_SEED,
        manifest_ids=sorted(clips),
    )
    if sample is None:
        sample = write_sample(
            path,
            dataset=dataset,
            count=spec["count"],
            seed=SAMPLE_SEED,
            manifest_ids=sorted(clips),
        )
        say(
            f"  {dataset}: drew {len(sample)} of {len(clips)} clips; the draw is fixed in the cache"
        )
    return [clips[clip_id] for clip_id in sample]


def transcribe(dataset: str, root: Path, verbose: bool) -> None:
    """Both models on every sampled clip: every Whisper request, then every Qwen one."""
    spec = audit_dataset(dataset)
    language, cache = spec["language"], cache_dir()
    clips = sampled_clips(dataset, spec, root)
    dataset_dir = root / dataset
    if verbose:
        say(f"{dataset}: {len(clips)} clips, {language}, against {spec['human']}")
    for model, label in ((SEGMENT_MODEL, "Whisper"), (LABEL_MODEL, "Qwen")):
        missing = [
            clip
            for clip in clips
            if read_answer(
                answer_path(cache, model, clip.id),
                model=model,
                language=language,
                audio_sha256=clip.sha256,
                request={},
            )
            is None
        ]
        if not missing:
            if verbose:
                say(f"  {label}: every answer in cache, nothing to ask")
            continue
        if verbose:
            say(f"  {label}: {len(missing)} answers to ask, {len(clips) - len(missing)} in cache")
        transcribe_ = client(model)
        failing = threading.Event()
        try:
            one_pass = served(transcribe_, language).one_pass()
            map_requests(
                partial(ask, dataset_dir, cache, one_pass.transcribe, language, model, failing),
                missing,
                one_pass.concurrency,
                should_stop=lambda f=failing: stop_requested() or f.is_set(),
                failing=failing,
            )
            raise_if_stopped(f"the {label} pass of the {dataset} audit stopped")
        finally:
            transcribe_.close()
    if verbose:
        say(f"{dataset}: every sampled clip holds both answers")


def ask(
    dataset_dir: Path,
    cache: Path,
    transcribe_: Callable[[bytes], dict],
    language: str,
    model: str,
    failing: threading.Event,
    clip: Clip,
) -> None:
    """One clip's answer from one model, cached on the way.

    The audio is the clip's own WAV as the pipeline sends it: 16 kHz mono
    re-encoded to 16-bit, the bytes ``clip_text`` would send.
    """
    if failing.is_set() or stop_requested():
        raise Stopped("the audit stopped")
    wav = wav_bytes(load_audio(dataset_dir / clip.audio).samples)
    text = " ".join(str(transcribe_(wav)["text"]).split())
    write_answer(
        answer_path(cache, model, clip.id),
        model=model,
        language=language,
        audio_sha256=clip.sha256,
        request={},
        text=text,
    )


def load_answers(dataset: str, spec: dict, root: Path) -> list[dict]:
    """Every sampled clip with its two answers and its human label, in the draw's order."""
    cache = cache_dir()
    language = spec["language"]
    rows = []
    for clip in sampled_clips(dataset, spec, root):
        row = {"id": clip.id, "human": clip.text, "base_wer": clip.base_wer, "language": language}
        for model, key in ((SEGMENT_MODEL, "whisper"), (LABEL_MODEL, "qwen")):
            text = read_answer(
                answer_path(cache, model, clip.id),
                model=model,
                language=language,
                audio_sha256=clip.sha256,
                request={},
            )
            if text is None:
                raise SystemExit(
                    f"{clip.id}: no cached {key} answer; run the transcribe step first"
                )
            row[key] = text
        rows.append(row)
    return rows


def verdicts(rows: list[dict]) -> list[dict]:
    """The agreement rule on every row: kept, its label, and its word counts.

    The rule is the pipeline's own (``audiolab.data.talks``), applied the way
    the pipeline applies it: Whisper's text against the Qwen text, in the
    normalized view, over the whole clip.
    """
    out = []
    for row in rows:
        language = row["language"]
        d = disagreement(row["whisper"], row["qwen"], language)
        label = plain_label(row["qwen"])
        c = counts(row["human"], label, language, "normalized")
        out.append(
            {
                **row,
                "kept": d <= MAX_DISAGREEMENT,
                "label": label,
                "errors": c.errors,
                "reference_words": c.reference_words,
                "per_clip_wer": c.wer,
            }
        )
    return out


def corpus_wer(values: list[dict]) -> float:
    """The corpus WER of a group: errors over reference words, as the panels report it."""
    errors = sum(value["errors"] for value in values)
    words = sum(value["reference_words"] for value in values)
    return errors / words if words else float("nan")


def summarize(rows: list[dict]) -> dict:
    """The numbers one arm reports, over the kept, the rejected and each bucket."""
    scored = verdicts(rows)
    kept = [row for row in scored if row["kept"]]
    rejected = [row for row in scored if not row["kept"]]
    summary = {
        "normalizer_version": NORMALIZER_VERSION,
        "clips": len(scored),
        "kept": {
            "count": len(kept),
            "share": len(kept) / len(scored) if scored else 0.0,
            "wer": corpus_wer(kept),
            "median_per_clip_wer": _median(kept),
        },
        "rejected": {
            "count": len(rejected),
            "share": len(rejected) / len(scored) if scored else 0.0,
            "wer": corpus_wer(rejected),
            "median_per_clip_wer": _median(rejected),
        },
    }
    if any(row["base_wer"] is not None for row in scored):
        buckets = []
        for name, low, high in BASE_WER_BUCKETS:
            in_bucket = [row for row in scored if low <= row["base_wer"] < high]
            if not in_bucket:
                continue
            kept_bucket = [row for row in in_bucket if row["kept"]]
            rejected_bucket = [row for row in in_bucket if not row["kept"]]
            buckets.append(
                {
                    "bucket": name,
                    "clips": len(in_bucket),
                    "kept_count": len(kept_bucket),
                    "kept_wer": corpus_wer(kept_bucket),
                    "rejected_wer": corpus_wer(rejected_bucket),
                }
            )
        summary["base_wer_buckets"] = buckets
    return summary


def _median(rows: list[dict]) -> float:
    values = [row["per_clip_wer"] for row in rows]
    return float(statistics.median(values)) if values else float("nan")


def render_summary(dataset: str, spec: dict, summary: dict) -> str:
    """The arm's numbers as a block of Markdown, for the report and the docs."""
    lines = [
        f"**{dataset}** — {summary['clips']} clips against {spec['human']} "
        f"(normalizer v{summary['normalizer_version']}):"
        "",
        f"- the agreement rule keeps **{summary['kept']['count']}** clips "
        f"({summary['kept']['share']:.1%}); the kept labels score "
        f"**{summary['kept']['wer']:.1%}** WER against the human text "
        f"(median per-clip WER {summary['kept']['median_per_clip_wer']:.1%})",
        f"- it rejects **{summary['rejected']['count']}** clips "
        f"({summary['rejected']['share']:.1%}); the Qwen label of the rejected ones "
        f"would have scored **{summary['rejected']['wer']:.1%}** WER",
    ]
    buckets = summary.get("base_wer_buckets")
    if buckets:
        lines.append("- by how the base model did on the human label (`base_wer`):")
        lines.append("")
        lines.append("  | bucket | clips | kept | kept WER | rejected WER |")
        lines.append("  |---|---|---|---|---|")
        for bucket in buckets:
            kept = f"{bucket['kept_count']} / {bucket['clips']}"
            rejected_wer = (
                f"{bucket['rejected_wer']:.1%}" if bucket["clips"] - bucket["kept_count"] else "—"
            )
            lines.append(
                f"  | {bucket['bucket']} | {bucket['clips']} | {kept} "
                f"| {bucket['kept_wer']:.1%} | {rejected_wer} |"
            )
    return "\n".join(lines)


# --- the examples: what a kept label differs from the human text on ---------


@dataclass(frozen=True)
class Diff:
    """One word-level difference of one kept clip, with the raw words it sits on."""

    kind: str  # "substitution", "insertion", "deletion"
    reference_token: str = ""
    hypothesis_token: str = ""
    reference_raw: str = ""
    hypothesis_raw: str = ""
    reference_sentence_start: bool = False
    hypothesis_sentence_start: bool = False


def align_ops(reference: list[str], hypothesis: list[str]) -> list[tuple[str, str, str]]:
    """The word-level edit path: (kind, reference token, hypothesis token)."""
    n, m = len(reference), len(hypothesis)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            cost = 0 if reference[i] == hypothesis[j] else 1
            table[i + 1][j + 1] = min(table[i][j] + cost, table[i + 1][j] + 1, table[i][j + 1] + 1)
    ops: list[tuple[str, str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and table[i][j]
            == table[i - 1][j - 1] + (0 if reference[i - 1] == hypothesis[j - 1] else 1)
        ):
            kind = "substitution" if reference[i - 1] != hypothesis[j - 1] else "equal"
            ops.append((kind, reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and table[i][j] == table[i - 1][j] + 1:
            ops.append(("deletion", reference[i - 1], ""))
            i -= 1
        else:
            ops.append(("insertion", "", hypothesis[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def raw_under(tokens: list[str], text: str, language: str) -> list[tuple[str, bool]] | None:
    """Under each normalized token: the raw word that produced it, and whether
    that word starts a sentence.

    The normalizer is per-word for everything that matters here (casing,
    punctuation, numbers read as words), so the tokens a side's words produce
    are found by normalizing each raw word alone, and the raw word is carried
    on every token it produces. When the words do not re-compose the whole
    text's tokens (a context the per-word read misses), the side has no
    mapping and the caller leaves its raw fields empty.
    """
    raw = text.split()
    mapped: list[tuple[str, bool]] = []
    for index, word in enumerate(raw):
        produced = normalize(word, language).split()
        starts = index == 0 or raw[index - 1].rstrip().endswith(SENTENCE_ENDS)
        if not produced:
            continue
        mapped.extend([(word, starts)] * len(produced))
    if len(mapped) != len(tokens):
        return None
    return mapped


def clip_diffs(row: dict) -> list[Diff]:
    """The differences of one kept clip, in the normalized view, with the raw words."""
    reference = tokens(row["human"], row["language"], "normalized")
    hypothesis = tokens(row["label"], row["language"], "normalized")
    reference_map = raw_under(reference, row["human"], row["language"])
    hypothesis_map = raw_under(hypothesis, row["label"], row["language"])
    diffs: list[Diff] = []
    i = j = 0
    for kind, ref_token, hyp_token in align_ops(reference, hypothesis):
        if kind == "equal":
            i += ref_token is not None
            j += hyp_token is not None
            continue
        ref_word, ref_starts = reference_map[i] if reference_map and ref_token else ("", False)
        hyp_word, hyp_starts = hypothesis_map[j] if hypothesis_map and hyp_token else ("", False)
        diffs.append(Diff(kind, ref_token, hyp_token, ref_word, hyp_word, ref_starts, hyp_starts))
        if ref_token:
            i += 1
        if hyp_token:
            j += 1
    return diffs


def category(diffs: list[Diff]) -> str:
    """Which of the example categories a clip's differences fall in.

    Numbers first: any difference on a word a number is read as, or a digit.
    Then names: a word a name would be — capitalized, alphabetic, mid-sentence
    — substituted for another. Then fillers: every difference an inserted
    discourse word, the style a near-verbatim transcript leaves out. Other
    otherwise.
    """
    for diff in diffs:
        for token in (diff.reference_token, diff.hypothesis_token):
            if token.isdigit() or token.lower() in NUMBER_WORDS:
                return "numbers"
    for diff in diffs:
        if diff.kind != "substitution":
            continue
        if (
            diff.reference_raw
            and diff.hypothesis_raw
            and not diff.reference_sentence_start
            and not diff.hypothesis_sentence_start
            and _name_like(diff.reference_raw)
            and _name_like(diff.hypothesis_raw)
        ):
            return "name spellings"
    if all(diff.kind == "insertion" for diff in diffs):
        inserted = {diff.hypothesis_token.lower() for diff in diffs}
        if inserted <= FILLERS:
            return "filler insertions"
    return "other"


def _name_like(word: str) -> bool:
    """A raw word in the shape of a name: it starts with a capital and holds letters."""
    return bool(word) and word[0].isupper() and any(character.isalpha() for character in word)


def examples(dataset: str, spec: dict, root: Path) -> str:
    """The kept clips' differences, classified, 30 of them as a Markdown table."""
    scored = verdicts(load_answers(dataset, spec, root))
    kept = [row for row in scored if row["kept"] and row["errors"] > 0]
    by_category: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for row in sorted(kept, key=lambda row: row["id"]):
        by_category[category(clip_diffs(row))].append(row)
    counts_per = {c: len(rows) for c, rows in by_category.items()}
    picked: list[tuple[str, dict]] = []
    for c in CATEGORIES:
        picked.extend((c, row) for row in by_category[c][:EXAMPLES_PER_CATEGORY])
        if len(picked) >= TOTAL_EXAMPLES:
            break
    lines = [
        f"## {dataset}: the kept labels against the human text, word by word",
        "",
        f"Differences in the normalized view (normalizer v{NORMALIZER_VERSION}), what the WER "
        "counts; the bare interjections it drops (uh, um, hmm) are not in it. "
        f"{', '.join(f'{c} {counts_per[c]}' for c in CATEGORIES)} of the {len(kept)} kept clips "
        "have differences; at most 10 of a category are shown.",
        "",
        "| clip | category | human | qwen label |",
        "|---|---|---|---|",
    ]
    for c, row in picked[:TOTAL_EXAMPLES]:
        lines.append(
            f"| `{row['id']}` | {c} | {_excerpt(row['human'])} | {_excerpt(row['label'])} |"
        )
    return "\n".join(lines)


def _excerpt(text: str) -> str:
    words = text.split()
    if len(words) <= EXAMPLE_MAX_WORDS:
        return text.replace("|", "\\|")
    return " ".join(words[:EXAMPLE_MAX_WORDS]).replace("|", "\\|") + " …"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=None, help="$AUDIOLAB_DATA")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in (
        ("transcribe", "draw the sample and ask both models for every sampled clip"),
        ("score", "apply the agreement rule and score the kept and the rejected"),
        ("examples", "the 30 classified differences of the kept clips"),
    ):
        command = sub.add_parser(name, help=help_)
        command.add_argument("dataset", choices=sorted(AUDITS))
    args = parser.parse_args()
    root = args.data_dir or data_dir()
    spec = audit_dataset(args.dataset)
    if args.command == "transcribe":
        transcribe(args.dataset, root, verbose=True)
    elif args.command == "score":
        rows = load_answers(args.dataset, spec, root)
        summary = summarize(rows)
        print(render_summary(args.dataset, spec, summary))
        out = cache_dir() / "label-audit" / f"{args.dataset}-scored.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in verdicts(rows))
            + "\n",
            encoding="utf-8",
        )
        say(f"per-clip rows: {out}")
    elif args.command == "examples":
        print(examples(args.dataset, spec, root))


if __name__ == "__main__":
    main()
