#!/usr/bin/env python3
"""The rebuild impact report: what each exclusion rule removes from each prepared training set.

Read-only over the prepared data of the data root (default ``$AUDIOLAB_DATA``):
for every prepared training set — a directory whose name contains ``-train``,
excluding every name containing ``.pre-`` or ``.curly-`` (a renamed or
superseded preparation) — the exclusion rules the corrected fetchers apply are
applied to the set's current ``manifest.jsonl`, and the clips each rule removes
are counted. Nothing is written, no audio is opened, no network request is
sent, and the report depends on no server, model or GPU: the same data root
gives the same table on every machine, and the script is safe to run
repeatedly. It reports which evaluation sets it found prepared, so an
evaluation set that is missing at run time is visible, not silent.

The rules are never reimplemented here: every rule is the committed function
the fetcher uses, applied to the set's clips. The one-to-one mapping:

- **evaluation group or duplicate class of an evaluation group** —
  :func:`audiolab.data.duplicates.duplicate_classes` and
  :func:`audiolab.data.duplicates.excluded_talks`
  (``src/audiolab/data/duplicates.py``), with the evaluation groups the
  prepared evaluation sets of the source hold
  (:func:`audiolab.data.train.evaluation_groups` on the source's
  ``evaluation_datasets``), so a v2 evaluation set counts as soon as it is
  prepared;
- **re-upload of a talk already read in the same set (one talk per class)** —
  the same duplicate classes and the fetchers' committed selection: the first
  talk of a class in the reading order
  (:func:`audiolab.data.talk_listing.reading_order` with the seed of
  :data:`audiolab.data.talks.SEED` and the source's ``spread_by``), applied to
  the set's own clips, after the evaluation exclusion the fetchers make first
  (the loops of :func:`audiolab.data.talks.fetch_talks` and
  :func:`audiolab.data.nasa.fetch_nasa_en_train`);
- **NASA reserved-test-episode class** —
  :data:`audiolab.data.nasa.RESERVED_TEST_EPISODES` through the same
  :func:`audiolab.data.duplicates.excluded_talks`;
- **SLR57 speaker mapping** —
  :func:`audiolab.data.african_french.excluded_train_speakers` on the groups of
  the prepared ``african-accented-fr-test``;
- **VoxPopuli shared session** — the committed pin
  :data:`audiolab.data.voxpopuli_train.EVALUATION_SESSIONS` with
  :func:`audiolab.data.voxpopuli_train.session_of` on each clip id;
- **babble utterance** — the committed pin
  :func:`audiolab.data.noisy.load_babble_utterances` with
  :func:`audiolab.data.librispeech_pc.babble_exclusions` on each clip's corpus
  utterance id.

The table has one line per training set: for each rule, the clips it removes
from the set's current manifest, and the total — the union, so a clip several
rules remove counts once. The only rules that can remove the same clip are the
two NASA ones: the evaluation-class rule reads the prepared ``nasa-en-test`` /
``nasa-en-test-v2`` manifests, the reserved-episode rule the committed list,
and on a machine where both test sets are prepared they cover the same clips.

A set may have been prepared from an earlier pinning of its listing: a clip of
a talk that is no longer in the current pinning gets no rule count (it has no
class and no place in the reading order) and is listed under the table; the
re-preparation reads the current pinning, so those clips leave the set on
their own. A malformed manifest is still a hard error.

The rebuild list is the sets a re-preparation with the corrected fetchers must
re-run: every set where one of the five evaluation rules removes a clip; every
sized (``-30k`` / ``-40k``) set where the within-set re-upload rule removes a
clip; and every set derived from a set of the list — the ``-<k>k`` larger
preparation of its source (:func:`audiolab.data.datasets.base_dataset`) and
the re-labeling of :mod:`audiolab.data.relabel` (``nasa-en-train-qwen`` reads
the clips of ``nasa-en-train``). A 12 000-clip set that only loses within-set
re-uploads is not rebuilt: the re-uploads are dropped whenever the set is
re-prepared for another reason, and re-preparing it for that reason alone is
not a rebuild. The two babble pools ``mls-fr-train`` and ``librispeech-en-train``
are never rebuilt: their manifests are the committed babble of the noisy
evaluation sets (:mod:`audiolab.data.noisy`), and a re-preparation of the noisy
sets is refused when the pools no longer give the committed utterances.

Exit codes: 0 a table is printed; 1 a manifest is malformed (the set is named
on stderr); 2 the data root holds no prepared training sets.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from audiolab.core.paths import data_dir
from audiolab.data import (
    african_french,
    librispeech_pc,
    nasa,
    noisy,
    relabel,
    talk_sources,
    voxpopuli_train,
)
from audiolab.data import duplicates as dup
from audiolab.data.datasets import base_dataset
from audiolab.data.store import MANIFEST_NAME, Clip, read_manifest
from audiolab.data.talk_listing import read_listing, reading_order
from audiolab.data.talks import PEERTUBE_SOURCES, SEED
from audiolab.data.train import evaluation_groups

#: The table's columns, in order.
RULE_EVAL_CLASS = "evaluation group or its class"
RULE_REUPLOAD = "re-upload of a talk already read"
RULE_NASA = "nasa reserved-test-episode class"
RULE_SLR57 = "slr57 speaker"
RULE_VOX = "voxpopuli shared session"
RULE_BABBLE = "babble utterance"
RULES = (RULE_EVAL_CLASS, RULE_REUPLOAD, RULE_NASA, RULE_SLR57, RULE_VOX, RULE_BABBLE)
#: The rules a clip removed by any of them forces the set into the rebuild list.
EVAL_RULES = (RULE_EVAL_CLASS, RULE_NASA, RULE_SLR57, RULE_VOX, RULE_BABBLE)
#: The two sets that supply the noisy sets' babble: never re-prepared.
BABBLE_POOLS = ("mls-fr-train", "librispeech-en-train")
#: Training-set names that are renamed or superseded preparations.
SKIP_MARKERS = (".pre-", ".curly-")
#: The prepared evaluation sets whose groups are the evaluation groups of the
#: NASA source: its test set and the v2 that derives from it (nasa's
#: NAME_TEST / NAME_TEST_V2).
NASA_EVAL_SETS = (nasa.NAME_TEST, nasa.NAME_TEST_V2)
#: The prepared set a re-labeling reads: the name maps to the set of its clips
#: (relabel.SOURCE, the row of fetch.py's FETCHERS that prepares it).
RELABEL_SOURCES: dict[str, str] = {"nasa-en-train-qwen": relabel.SOURCE}
#: The test part whose speakers feed the SLR57 rule.
SLR57_TEST_SET = "african-accented-fr-test"

#: The short column titles of the table.
COLUMN = {
    RULE_EVAL_CLASS: "eval class",
    RULE_REUPLOAD: "in-set re-upload",
    RULE_NASA: "nasa reserved",
    RULE_SLR57: "slr57 speaker",
    RULE_VOX: "vox session",
    RULE_BABBLE: "babble",
}


class ImpactError(RuntimeError):
    """A prepared manifest the report cannot read: the set is named in it."""


def load_clips(root: Path, name: str) -> list[Clip]:
    """The clips of one prepared set; a malformed manifest names the set."""
    try:
        return read_manifest(root / name)
    except (OSError, ValueError) as error:
        raise ImpactError(f"{name}: {error}") from error


def training_sets(root: Path) -> list[str]:
    """The prepared training sets: the ``-train`` directories that hold a manifest."""
    return sorted(
        entry.name
        for entry in root.iterdir()
        if entry.is_dir()
        and "-train" in entry.name
        and not any(marker in entry.name for marker in SKIP_MARKERS)
        and (entry / MANIFEST_NAME).is_file()
    )


@dataclass(frozen=True)
class RuleInputs:
    """What the rules read that is not a set's own manifest, computed once.

    ``talk_classes`` maps a talk source's base name to the duplicate class of
    each listed talk: the four talk bases of
    :data:`audiolab.data.talk_sources.SOURCES` (the PeerTube bases share one
    union population, as the fetcher computes it) plus
    :data:`nasa.NAME` for the NASA episodes; ``talk_order`` the same talks in
    the preparation's reading order; ``eval_groups`` the groups of the prepared
    evaluation sets the source lists.
    """

    talk_classes: Mapping[str, Mapping[str, str]]
    talk_order: Mapping[str, tuple[str, ...]]
    eval_groups: Mapping[str, frozenset[str]]
    nasa_reserved: frozenset[str]
    slr57_excluded: Mapping[str, str]
    vox_sessions: frozenset[str]
    babble: Mapping[str, list[str]]


def compute_rule_inputs(root: Path) -> tuple[RuleInputs, list[tuple[str, str, int | None]]]:
    """The rule inputs from the committed pins and the prepared data of ``root``.

    The returned status lists, per source, each evaluation set it lists and
    whether the data root holds it prepared (``None`` when not), so the report
    says what it read rather than assuming it.
    """
    classes: dict[str, dict[str, str]] = {}
    order: dict[str, tuple[str, ...]] = {}
    groups: dict[str, frozenset[str]] = {}
    status: list[tuple[str, str, int | None]] = []

    def probe(source_name: str, dataset: str) -> None:
        if (root / dataset / MANIFEST_NAME).is_file():
            status.append(
                (source_name, dataset, len({clip.group for clip in load_clips(root, dataset)}))
            )
        else:
            status.append((source_name, dataset, None))

    for name, source in talk_sources.SOURCES.items():
        if name == nasa.NAME:
            continue  # the NASA source is handled below, with its own classes
        listed = read_listing(source)[1]
        if source.name in PEERTUBE_SOURCES:
            # The PeerTube sources are one corpus listed three times: the
            # classes span the union, as the fetcher computes them.
            population, domain = dup.peertube_talks(), "peertube"
        else:
            population, domain = listed, source.name
        classes[name] = dup.duplicate_classes(population, domain=domain)
        order[name] = tuple(talk.id for talk in reading_order(listed, SEED, source.spread_by))
        for dataset in source.evaluation_datasets:
            probe(name, dataset)
        groups[name] = evaluation_groups(*source.evaluation_datasets, root=root)
    episodes = read_listing(nasa.SOURCE)[1]
    classes[nasa.NAME] = dup.duplicate_classes(episodes, domain="nasa")
    order[nasa.NAME] = tuple(
        episode.id for episode in reading_order(episodes, SEED, nasa.SOURCE.spread_by)
    )
    for dataset in NASA_EVAL_SETS:
        probe(nasa.NAME, dataset)
    groups[nasa.NAME] = evaluation_groups(*NASA_EVAL_SETS, root=root)
    probe("african-accented-fr-train", SLR57_TEST_SET)
    return (
        RuleInputs(
            talk_classes=classes,
            talk_order=order,
            eval_groups=groups,
            nasa_reserved=frozenset(nasa.RESERVED_TEST_EPISODES),
            slr57_excluded=african_french.excluded_train_speakers(
                evaluation_groups(SLR57_TEST_SET, root=root)
            ),
            vox_sessions=voxpopuli_train.EVALUATION_SESSIONS,
            babble=noisy.load_babble_utterances(),
        ),
        status,
    )


def _kind_of(base: str) -> str:
    """The rules a set's source carries: the fetcher that prepares it decides.

    The NASA base is checked before the talk sources: nasa.SOURCE sits in
    talk_sources.SOURCES for the shared list-talks and fetch-audio commands,
    but its preparation is nasa.fetch_nasa_en_train, not the talk loop.
    """
    if base in (nasa.NAME, "nasa-en-train-qwen"):
        return "nasa"
    if base in talk_sources.SOURCES:
        return "talk"
    if base == "african-accented-fr-train":
        return "slr57"
    if base == "voxpopuli-fr-train":
        return "voxpopuli"
    return "other"


def _in_set_reuploads(
    order: tuple[str, ...],
    present: set[str],
    classes: Mapping[str, str],
    left_out: set[str],
) -> set[str]:
    """The talks a preparation does not read of this set's own clips.

    The fetchers' loop restricted to the set: the talks in the preparation's
    reading order, the ones the evaluation exclusion already left out skipped,
    and each other class read once — the first talk of it standing for it.
    """
    removed: set[str] = set()
    seen: set[str] = set()
    for talk_id in order:
        if talk_id not in present or talk_id in left_out:
            continue
        class_id = classes[talk_id]
        if class_id in seen:
            removed.add(talk_id)
        else:
            seen.add(class_id)
    return removed


@dataclass(frozen=True)
class SetImpact:
    """One training set: its clips, what each rule removes of them, and the talks that are
    no longer in the source's pinned listing (the set was prepared from an earlier one)."""

    name: str
    clips: int
    removed: Mapping[str, frozenset[str]] = field(compare=False)
    stale: frozenset[str] = frozenset()

    def count(self, rule: str) -> int:
        return len(self.removed[rule])

    @property
    def total(self) -> int:
        union: set[str] = set()
        for clip_ids in self.removed.values():
            union |= clip_ids
        return len(union)


def impact_on_set(name: str, clips: list[Clip], inputs: RuleInputs) -> SetImpact:
    """The clips each rule removes from one set's clips.

    A clip of a talk that is no longer in the source's pinned listing is not
    counted by any rule: it has no class and no place in the reading order, and
    the re-preparation reads the current listing, so it leaves the set on its
    own. It is recorded in :attr:`SetImpact.stale` and the report says so.
    """
    base = base_dataset(name)
    kind = _kind_of(base)
    by_group: dict[str, list[Clip]] = {}
    for clip in clips:
        by_group.setdefault(clip.group.partition(":")[2], []).append(clip)
    removed: dict[str, set[str]] = {rule: set() for rule in RULES}
    stale: frozenset[str] = frozenset()

    if kind in ("talk", "nasa"):
        key = nasa.NAME if kind == "nasa" else base
        classes = inputs.talk_classes[key]
        stale = frozenset(set(by_group) - set(classes))
        excluded = dup.excluded_talks(classes, inputs.eval_groups[key])
        for talk_id in excluded:
            if talk_id in by_group:
                removed[RULE_EVAL_CLASS] |= {clip.id for clip in by_group[talk_id]}
        left_out = set(excluded)
        if kind == "nasa":
            nasa_excluded = dup.excluded_talks(classes, inputs.nasa_reserved)
            for talk_id in nasa_excluded:
                if talk_id in by_group:
                    removed[RULE_NASA] |= {clip.id for clip in by_group[talk_id]}
            left_out |= set(nasa_excluded)
        for talk_id in _in_set_reuploads(inputs.talk_order[key], set(by_group), classes, left_out):
            removed[RULE_REUPLOAD] |= {clip.id for clip in by_group[talk_id]}

    if kind == "slr57":
        for speaker in inputs.slr57_excluded:
            if speaker in by_group:
                removed[RULE_SLR57] |= {clip.id for clip in by_group[speaker]}

    if kind == "voxpopuli":
        for clip in clips:
            if voxpopuli_train.session_of(clip.id) in inputs.vox_sessions:
                removed[RULE_VOX].add(clip.id)

    blocked = librispeech_pc.babble_exclusions(
        (clip.id.partition(":")[2] for clip in clips), inputs.babble
    )
    if blocked:
        removed[RULE_BABBLE] = set(blocked)

    return SetImpact(
        name, len(clips), {rule: frozenset(ids) for rule, ids in removed.items()}, stale
    )


def audit(names: Iterable[str], root: Path, inputs: RuleInputs) -> list[SetImpact]:
    """The report's rows: one per prepared training set, in name order."""
    return [impact_on_set(name, load_clips(root, name), inputs) for name in sorted(names)]


def derivation_of(name: str) -> str | None:
    """The set a name derives from: its ``-<k>k`` source, or the set a re-labeling reads."""
    base = base_dataset(name)
    if base != name:
        return base
    return RELABEL_SOURCES.get(name)


def _clips(n: int) -> str:
    return f"{n} clip{'s' if n != 1 else ''}"


def rebuild_list(
    rows: list[SetImpact],
) -> tuple[list[tuple[str, tuple[str, ...]]], list[tuple[str, str]]]:
    """The sets to re-prepare, and the sets that are not, each with its reason.

    A set is rebuilt when one of the evaluation rules removes a clip of it, when
    it is a sized set the within-set re-upload rule removes a clip of, or when
    it derives from a set of the list (closed over derivations). The babble
    pools never are.
    """
    names = [row.name for row in rows]
    reasons: dict[str, list[str]] = {}
    for row in rows:
        own = [
            f"{COLUMN[rule]}: {_clips(row.count(rule))}" for rule in EVAL_RULES if row.count(rule)
        ]
        if row.count(RULE_REUPLOAD) and base_dataset(row.name) != row.name:
            own.append(f"{COLUMN[RULE_REUPLOAD]}: {_clips(row.count(RULE_REUPLOAD))} (sized set)")
        if own:
            reasons[row.name] = own
    changed = True
    while changed:
        changed = False
        for name in names:
            source = derivation_of(name)
            if source and source in reasons and name not in reasons:
                reasons[name] = [f"derived from {source}"]
                changed = True
    for pool in BABBLE_POOLS:
        reasons.pop(pool, None)
    not_rebuilt: list[tuple[str, str]] = []
    for row in rows:
        if row.name in reasons:
            continue
        if row.name in BABBLE_POOLS and row.count(RULE_BABBLE):
            not_rebuilt.append((row.name, "babble pool: the noisy sets' committed babble"))
        elif row.count(RULE_REUPLOAD) and base_dataset(row.name) == row.name:
            not_rebuilt.append((row.name, "only in-set re-uploads in a 12 000-clip set"))
    return [(name, tuple(whys)) for name, whys in sorted(reasons.items())], not_rebuilt


def render(
    root: Path, rows: list[SetImpact], inputs: RuleInputs, status: list[tuple[str, str, int | None]]
) -> str:
    """The report as printed: the evaluation sets read, the table, the rebuild list."""
    lines = [
        "# Impact of the new exclusion rules on the prepared training sets",
        "",
        f"data root: {root}",
        "",
        "evaluation sets found prepared (the evaluation-class rule reads their manifests):",
    ]
    for source, dataset, group_count in status:
        if group_count is None:
            lines.append(f"  - {dataset} (evaluation set of {source}): NOT prepared")
        else:
            lines.append(
                f"  - {dataset} (evaluation set of {source}): prepared, {group_count} groups"
            )
    babble_ids = sum(len(ids) for ids in inputs.babble.values())
    lines.append(
        f"committed pins: {len(inputs.vox_sessions)} voxpopuli evaluation sessions; "
        f"{babble_ids} babble utterance ids over {len(inputs.babble)} noisy sets; "
        f"{len(inputs.nasa_reserved)} nasa reserved test episodes"
    )
    lines.append("")
    header = "| set | clips | " + " | ".join(COLUMN[rule] for rule in RULES) + " | total |"
    lines.append(header)
    lines.append("|" + "---|" * (len(RULES) + 3))
    totals = {rule: 0 for rule in RULES}
    for row in rows:
        line = [row.name, str(row.clips)]
        for rule in RULES:
            totals[rule] += row.count(rule)
            line.append(str(row.count(rule)))
        line.append(str(row.total))
        lines.append("| " + " | ".join(line) + " |")
    total_all = sum(row.total for row in rows)
    totals_line = ["total", str(sum(row.clips for row in rows))]
    for rule in RULES:
        totals_line.append(str(totals[rule]))
    totals_line.append(str(total_all))
    lines.append("| " + " | ".join(totals_line) + " |")
    lines.append(
        "the `total` column is the union: a clip several rules remove "
        "(the two nasa rules cover the same clips) counts once; the rule columns may count it twice"
    )
    stale_rows = [(row.name, row.stale) for row in rows if row.stale]
    if stale_rows:
        lines.append("")
        lines.append(
            "clips of talks that are no longer in the source's pinned listing "
            "(prepared from an earlier listing): they carry no rule count; the "
            "re-preparation reads the current listing, so they leave the set on its own:"
        )
        for name, stale in stale_rows:
            lines.append(f"  - {name}: {len(stale)} talks — " + ", ".join(sorted(stale)))
    lines.append("")
    rebuild, not_rebuilt = rebuild_list(rows)
    lines.append(
        "rebuild list — the sets a re-preparation with the corrected fetchers must re-run:"
    )
    for name, why in rebuild:
        lines.append(f"- {name} — " + "; ".join(why))
    if not rebuild:
        lines.append("- (none)")
    lines.append("")
    lines.append("not rebuilt:")
    for name, why in not_rebuilt:
        lines.append(f"- {name} — {why}")
    for pool in BABBLE_POOLS:
        if pool not in [name for name, _ in not_rebuilt] and any(row.name == pool for row in rows):
            lines.append(f"- {pool} — babble pool, never re-prepared")
    if not not_rebuilt and not any(row.name in BABBLE_POOLS for row in rows):
        lines.append("- (none)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only report of the new exclusion rules over the prepared training sets."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="the prepared data root (default $AUDIOLAB_DATA)",
    )
    args = parser.parse_args(argv)
    root = args.data_dir or data_dir()
    if not root.is_dir():
        print(f"impact.py: no prepared data under {root}", file=sys.stderr)
        return 2
    names = training_sets(root)
    if not names:
        print(f"impact.py: no prepared training sets under {root}", file=sys.stderr)
        return 2
    try:
        inputs, status = compute_rule_inputs(root)
        rows = audit(names, root, inputs)
    except ImpactError as error:
        print(f"impact.py: {error}", file=sys.stderr)
        return 1
    print(render(root, rows, inputs, status), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
