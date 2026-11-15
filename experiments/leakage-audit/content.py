#!/usr/bin/env python3
"""Content-level leakage audit: text shingles between evaluation groups and training groups.

Finds, over the prepared manifests only (no audio is opened, nothing is written
under the data directory):

1. every pair (evaluation group, training group) whose 6-word text shingles
   overlap: shared shingles ``k >= MIN_SHARED`` and ``k`` at least
   ``EVAL_MIN_RATIO`` of the evaluation group's shingles (a pair is labelled
   ``SAME GROUP`` when the group ids are equal);
2. the same between training sets (for the mix): pairs of talk groups
   (``group_kind == "recording"``) across two different training sets, and
   within one training set, with ``k >= MIN_SHARED`` and ratio at least
   ``TRAIN_MIN_RATIO`` of the smaller group's shingles;
3. the re-upload shortlist: the train-vs-train pairs with ratio at least
   ``REUPLOAD_RATIO`` and different group ids — the same speech under two
   different ids, inside or across training sets.

Method: the same 6-word text shingles as the 2026-11-15 leak audits
(lowercase, NFKD, ``[a-z0-9]+`` words), and for the triage of a pair the
word-recall method of those audits' ``twins`` script: for each evaluation clip,
the fraction of its words that also occur in the text of the time-overlapping
training clips of the paired group.

The evaluation sets are the datasets that have panel files in
``configs/panels/`` (the registry has no "kind" field), minus ``nonspeech`` and
``nonspeech-music``. The training sets are the prepared directories whose name
contains ``-train``, including ``-Nk`` and ``-qwen`` suffixes, excluding every
name containing ``.pre-`` or ``.curly-`` (renamed or superseded preparations).

Read-only: the script streams the ``manifest.jsonl`` files one dataset at a
time through :func:`audiolab.data.store.read_manifest` and writes nothing.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import re
import sys
import unicodedata
from pathlib import Path

from audiolab.core.paths import data_dir
from audiolab.data.store import read_manifest

#: 6-word text shingles, lowercase NFKD words of ``[a-z0-9]+``.
N_WORDS = 6
#: Report a pair only from this many shared shingles up (a noise floor:
#: 15 shared 6-grams is ~20 consecutive identical words).
MIN_SHARED = 15
#: Evaluation group: report from this fraction of its shingles shared up.
EVAL_MIN_RATIO = 0.05
#: Training-vs-training groups: report from this fraction of the smaller
#: group's shingles shared up.
TRAIN_MIN_RATIO = 0.30
#: Train-vs-train ratio at which a pair with two different ids is a
#: re-upload (the same speech under two ids) for the mix.
REUPLOAD_RATIO = 0.75

#: Training set names to skip: renamed or superseded preparations.
SKIP_MARKERS = (".pre-", ".curly-")
#: Evaluation sets that are not speech scored against training.
NON_SPEECH_EVAL_SETS = {"nonspeech", "nonspeech-music"}
#: The panel directory, resolved from this file (repo root / configs / panels).
PANELS_DIR = Path(__file__).resolve().parents[2] / "configs" / "panels"

_WORDS = re.compile(r"[a-z0-9]+")
#: The clip id's trailing centiseconds: the clip's start in the recording.
_START = re.compile(r"-(\d+)$")


def words(text: str) -> list[str]:
    """The NFKD-lowercase ``[a-z0-9]+`` words of a label."""
    return _WORDS.findall(unicodedata.normalize("NFKD", text.lower()))


def shingles(text: str) -> set[str]:
    """The 6-word shingles of a label (empty for fewer than 6 words)."""
    w = words(text)
    return {" ".join(w[i : i + N_WORDS]) for i in range(len(w) - N_WORDS + 1)}


def clip_start_s(clip_id: str, group: str) -> float | None:
    """The clip's start (s) in its recording, from the id's centisecond suffix.

    Talk sources name a clip ``<group>-<centiseconds>`` (a PeerTube uuid, a
    NASA slug, a DebConf/FOSDEM video id); read-speech corpora do not, and
    yield ``None``.
    """
    local = clip_id.split(":", 1)[1]
    group_local = group.split(":", 1)[1] if ":" in group else group
    if group_local and not local.startswith(group_local):
        return None
    match = _START.search(local)
    if match is None:
        return None
    return int(match.group(1)) / 100


@dataclasses.dataclass
class Group:
    """One source group (talk, speaker or sentence) of one prepared set."""

    set: str
    name: str
    kind: str
    shingles: set[str] = dataclasses.field(default_factory=set)
    # (start_s or None, end_s, text, sha256) per clip
    clips: list[tuple[float | None, float, str, str]] = dataclasses.field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return (self.set, self.name)

    def add(self, clip) -> None:
        """One :class:`audiolab.data.store.Clip` of this group."""
        start = clip_start_s(clip.id, clip.group)
        self.clips.append((start, (start or 0.0) + clip.duration_s, clip.text, clip.sha256))
        self.shingles |= shingles(clip.text)


def load_groups(set_name: str, root: Path) -> dict[str, Group]:
    """The groups of one prepared dataset, read from its manifest only."""
    groups: dict[str, Group] = {}
    for clip in read_manifest(root / set_name):
        name = clip.group.split(":", 1)[1]
        group = groups.setdefault(name, Group(set=set_name, name=name, kind=clip.group_kind))
        group.add(clip)
    return groups


def word_recall(
    eval_clips: list[tuple[float | None, float, str, str]],
    train_clips: list[tuple[float | None, float, str, str]],
) -> tuple[float | None, int, int]:
    """Mean word recall of the evaluation clips' words in the overlapping train clips.

    Method of the 2026-11-15 leak audits' ``twins`` script: an evaluation clip
    is counted when its time span overlaps a training clip of the paired group;
    its recall is the fraction of its words (``[a-z0-9]+`` words) that also
    occur in the text of the time-overlapping training clips. Returns
    ``(mean, clips counted, clips with recall >= 0.7)``, or ``(None, 0, 0)``
    when no clip carries a start time on both sides (no talk).
    """
    timed_train = [c for c in train_clips if c[0] is not None]
    if not timed_train:
        return None, 0, 0
    sims: list[float] = []
    for start, end, text, _ in eval_clips:
        if start is None:
            continue
        cw = words(text)
        if not cw:
            continue
        overlap = [c for c in timed_train if c[0] < end and start < c[1]]
        if not overlap:
            continue
        tw = set(w for c in overlap for w in words(c[2]))
        sims.append(sum(1 for w in cw if w in tw) / len(cw))
    if not sims:
        return None, 0, 0
    return sum(sims) / len(sims), len(sims), sum(1 for s in sims if s >= 0.7)


@dataclasses.dataclass
class Pair:
    """One reported pair of groups, with its overlap numbers."""

    left: tuple[str, str]
    right: tuple[str, str]
    shared: int
    left_shingles: int
    right_shingles: int

    @property
    def same_group(self) -> bool:
        return self.left[1] == self.right[1]


@dataclasses.dataclass
class Finding:
    """An evaluation-vs-training pair with its triage evidence."""

    pair: Pair
    ratio: float
    shared_sha: int = 0
    shared_sha_clips: tuple[int, int] = (0, 0)  # (eval clips, train clips)
    recall: float | None = None
    recall_clips: int = 0
    recall_ge07: int = 0
    samples: list[tuple[str, str]] = dataclasses.field(default_factory=list)

    @property
    def verdict(self) -> str:
        """The category of the pair under OD-1 / OD-11 (see the README)."""
        eval_kind = _KINDS.get(self.pair.left, "")
        if self.pair.same_group:
            if eval_kind == "speaker":
                return "shared corpus speaker (SAME GROUP)"
            return "same recording (evaluation group in training, SAME GROUP)"
        if eval_kind == "speaker":
            if self.shared_sha:
                return "same recording (shared audio sha256)"
            return "same text, other reader (different speaker, no shared audio)"
        if eval_kind != "recording":
            # Not a talk group and not a corpus speaker id: outside the four
            # OD-1 categories, no automatic verdict.
            return "ADVISOR (not a talk, not a corpus speaker id)"
        if self.ratio >= TRAIN_MIN_RATIO:
            return "same recording (ratio >= 30 %)"
        if self.recall is not None and self.recall >= 0.7:
            return "same recording (word recall >= 0.7)"
        return "ADVISOR (5-30 %, no word recall >= 0.7)"


#: group key -> group_kind, filled by audit()
_KINDS: dict[tuple[str, str], str] = {}


def _evidence(finding: Finding, groups: dict[str, dict[str, Group]]) -> None:
    """Fill the evidence fields of one reported pair in place."""
    eval_set, eval_name = finding.pair.left
    train_set, train_name = finding.pair.right
    eg = groups[eval_set][eval_name]
    tg = groups[train_set][train_name]
    eval_sha_set = {c[3] for c in eg.clips}
    train_sha_set = {c[3] for c in tg.clips}
    common = eval_sha_set & train_sha_set
    finding.shared_sha = len(common)
    finding.shared_sha_clips = (
        sum(1 for c in eg.clips if c[3] in common),
        sum(1 for c in tg.clips if c[3] in common),
    )
    finding.recall, finding.recall_clips, finding.recall_ge07 = word_recall(eg.clips, tg.clips)
    # Three evaluation/training text pairs: the evaluation clips that overlap
    # training clips in time (the ones the recall counted), each next to the
    # texts of the overlapping training clips.
    for start, end, text, _ in eg.clips:
        if len(finding.samples) >= 3:
            break
        if start is None:
            continue
        overlap = [c for c in tg.clips if c[0] is not None and c[0] < end and start < c[1]]
        if not overlap:
            continue
        finding.samples.append(
            (text[:160], " || ".join(c[2][:120] for c in overlap[:2]) or "(no text)")
        )


@dataclasses.dataclass
class Audit:
    """The audit result over the prepared data of one run."""

    eval_sets: list[str]
    train_sets: list[str]
    findings: list[Finding]  # evaluation vs training
    train_pairs: list[Pair]  # talk groups across two different training sets
    within_pairs: list[Pair]  # talk groups inside one training set
    reuploads: list[Pair]  # train-vs-train, ratio >= REUPLOAD_RATIO, different ids
    groups: dict[str, dict[str, Group]]  # set -> group name -> Group


def audit(eval_sets: list[str], train_sets: list[str], root: Path) -> Audit:
    """Run the audit over the prepared manifests under ``root`` (read-only)."""
    global _KINDS
    _KINDS = {}

    eval_groups: dict[str, dict[str, Group]] = {}
    for name in eval_sets:
        eval_groups[name] = load_groups(name, root)
    for _name, gs in eval_groups.items():
        for g in gs.values():
            _KINDS[g.key] = g.kind
    allsh: set[str] = set()
    for gs in eval_groups.values():
        for g in gs.values():
            allsh |= g.shingles

    train_groups: dict[str, dict[str, Group]] = {}
    raw_eval_pairs: list[Pair] = []
    owners: dict[str, set[tuple[str, str]]] = collections.defaultdict(set)
    sizes: dict[tuple[str, str], int] = {}
    pair_count: dict[tuple[tuple[str, str], tuple[str, str]], int] = collections.Counter()

    for name in train_sets:
        groups = load_groups(name, root)
        train_groups[name] = groups
        for g in groups.values():
            _KINDS[g.key] = g.kind

        # 1. Evaluation vs this training set: one index of the training
        #    shingles that occur in some evaluation set, then the hit count
        #    per evaluation group.
        index: dict[str, set[str]] = collections.defaultdict(set)
        for g in groups.values():
            for s in g.shingles:
                if s in allsh:
                    index[s].add(g.name)
        for eval_set, evgs in eval_groups.items():
            for eg in evgs.values():
                if len(eg.shingles) < MIN_SHARED:
                    continue
                hits: dict[str, int] = collections.Counter()
                for s in eg.shingles:
                    for tname in index.get(s, ()):
                        hits[tname] += 1
                for tname, k in hits.items():
                    if k >= MIN_SHARED and k / len(eg.shingles) >= EVAL_MIN_RATIO:
                        raw_eval_pairs.append(
                            Pair(
                                (eval_set, eg.name),
                                (name, tname),
                                k,
                                len(eg.shingles),
                                len(groups[tname].shingles),
                            )
                        )

        # 2. Training vs training: per shared shingle, count the pairs of
        #    talk groups that hold it (across sets, and within one set).
        for g in groups.values():
            if g.kind != "recording":
                continue
            sizes[g.key] = len(g.shingles)
            for s in g.shingles:
                mine = owners[s]
                if g.key in mine:
                    continue
                for other in mine:
                    a, b = sorted((other, g.key))
                    pair_count[(a, b)] += 1
                mine.add(g.key)

    findings = []
    all_groups = {**eval_groups, **train_groups}
    for p in raw_eval_pairs:
        f = Finding(pair=p, ratio=p.shared / max(1, p.left_shingles))
        _evidence(f, all_groups)
        findings.append(f)
    findings.sort(key=lambda f: (-f.ratio, -f.pair.shared, f.pair.left, f.pair.right))

    all_train_pairs: list[tuple[Pair, bool]] = []  # (pair, within a single set)
    for (a, b), k in pair_count.items():
        if k < MIN_SHARED:
            continue
        p = Pair(a, b, k, sizes[a], sizes[b])
        if p.shared / min(p.left_shingles, p.right_shingles) < TRAIN_MIN_RATIO:
            continue
        all_train_pairs.append((p, a[0] == b[0]))
    train_pairs = [p for p, within in all_train_pairs if not within]
    within_pairs = [p for p, within in all_train_pairs if within]

    def order(p: Pair):
        ratio = p.shared / min(p.left_shingles, p.right_shingles)
        return (-ratio, -p.shared, p.left, p.right)

    train_pairs.sort(key=order)
    within_pairs.sort(key=order)
    reuploads = sorted(
        (
            p
            for p in train_pairs + within_pairs
            if p.shared / min(p.left_shingles, p.right_shingles) >= REUPLOAD_RATIO
            and p.left[1] != p.right[1]
        ),
        key=order,
    )

    return Audit(
        eval_sets=eval_sets,
        train_sets=train_sets,
        findings=findings,
        train_pairs=train_pairs,
        within_pairs=within_pairs,
        reuploads=reuploads,
        groups={**eval_groups, **train_groups},
    )


def _panel_datasets(panels_dir: Path) -> set[str]:
    """The dataset names behind the panel files ``<dataset>-{fast,full,sealed}.txt``."""
    datasets: set[str] = set()
    for path in panels_dir.glob("*.txt"):
        for suffix in ("-fast", "-full", "-sealed"):
            if path.name.endswith(suffix + ".txt"):
                datasets.add(path.name[: -len(suffix) - 4])
    return datasets


def discover(data_root: Path, panels_dir: Path = PANELS_DIR) -> tuple[list[str], list[str]]:
    """The evaluation and training set names of the prepared data.

    Evaluation sets: the datasets that have panel files in ``panels_dir``
    (the registry has no "kind" field), minus the non-speech sets, limited to
    the prepared ones. Training sets: the prepared directories whose name
    contains ``-train`` (including ``-Nk`` and ``-qwen``), excluding every name
    containing a skip marker (``.pre-``, ``.curly-``).
    """
    eval_sets = sorted(
        _panel_datasets(panels_dir) - NON_SPEECH_EVAL_SETS if panels_dir.is_dir() else set()
    )
    eval_sets = [s for s in eval_sets if (data_root / s / "manifest.jsonl").is_file()]
    train_sets = sorted(
        d.name
        for d in data_root.iterdir()
        if d.is_dir()
        and "-train" in d.name
        and not any(marker in d.name for marker in SKIP_MARKERS)
        and (d / "manifest.jsonl").is_file()
    )
    return eval_sets, train_sets


def _label(key: tuple[str, str]) -> str:
    return f"{key[0]}:{key[1]}"


def _print_pair_table(pairs: list[Pair], header: list[str]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(["---"] * len(header)) + "|")
    for p in pairs:
        ratio = p.shared / min(p.left_shingles, p.right_shingles)
        print(
            f"| {_label(p.left)} | {_label(p.right)} | {p.shared} "
            f"| {p.left_shingles} | {p.right_shingles} | {ratio:.1%} "
            f"| {'SAME GROUP' if p.same_group else ''} |"
        )
    if not pairs:
        print("| (none) |  |  |  |  |  |  |")


def report(result: Audit, data_root: Path) -> None:
    """The Markdown tables of the audit on stdout, sorted by ratio."""
    print(f"# Content-level leakage audit — prepared data under `{data_root}`")
    print()
    print(
        "6-word text shingles (NFKD lowercase, `[a-z0-9]+` words); a pair is "
        f"reported from shared `k >= {MIN_SHARED}` and a ratio of at least "
        f"{EVAL_MIN_RATIO:.0%} of the evaluation group's shingles."
    )
    print()
    print("## Population")
    print()
    print("| set | clips | groups |")
    print("|---|---|---|")
    for name in result.eval_sets + result.train_sets:
        gs = result.groups[name]
        clips = sum(len(g.clips) for g in gs.values())
        print(f"| {name} | {clips} | {len(gs)} |")
    print()
    print(
        f"## Evaluation vs training ({len(result.findings)} pairs, "
        "sorted by ratio of the evaluation group)"
    )
    print()
    print(
        "| evaluation group | training group | shared k | eval shingles | "
        "train shingles | ratio | flag |"
    )
    print("|---|---|---|---|---|---|---|")
    for f in result.findings:
        p = f.pair
        print(
            f"| {_label(p.left)} | {_label(p.right)} | {p.shared} "
            f"| {p.left_shingles} | {p.right_shingles} | {f.ratio:.1%} "
            f"| {'SAME GROUP' if p.same_group else ''} |"
        )
    if not result.findings:
        print("| (none) |  |  |  |  |  |  |")
    print()
    print(
        f"## Training vs training, across sets ({len(result.train_pairs)} talk-group pairs "
        f"with ratio >= {TRAIN_MIN_RATIO:.0%} of the smaller group)"
    )
    print()
    _print_pair_table(
        result.train_pairs,
        [
            "training group A",
            "training group B",
            "shared k",
            "A shingles",
            "B shingles",
            "ratio",
            "flag",
        ],
    )
    print()
    print(
        f"## Training vs training, within one set ({len(result.within_pairs)} talk-group pairs "
        f"with ratio >= {TRAIN_MIN_RATIO:.0%} of the smaller group)"
    )
    print()
    _print_pair_table(
        result.within_pairs,
        [
            "training group A",
            "training group B",
            "shared k",
            "A shingles",
            "B shingles",
            "ratio",
            "flag",
        ],
    )
    print()
    print(
        f"## Re-upload shortlist (train-vs-train, ratio >= {REUPLOAD_RATIO:.0%}, "
        "different group ids)"
    )
    print()
    _print_pair_table(
        result.reuploads,
        [
            "training group A",
            "training group B",
            "shared k",
            "A shingles",
            "B shingles",
            "ratio",
            "flag",
        ],
    )


def triage(result: Audit) -> None:
    """Per-pair triage evidence: shared audio, word recall, three text pairs."""
    print()
    print("## Triage evidence, per evaluation-vs-training pair")
    print()
    for f in result.findings:
        p = f.pair
        recall = "n/a" if f.recall is None else f"{f.recall:.2f}"
        print(
            f"- {_label(p.left)} -> {_label(p.right)}: verdict {f.verdict!r}; "
            f"k={p.shared}, ratio={f.ratio:.1%}, shared audio sha256 "
            f"{f.shared_sha} clips ({f.shared_sha_clips[0]} eval / {f.shared_sha_clips[1]} train), "
            f"word recall {recall} over {f.recall_clips} time-overlapping eval clips "
            f"({f.recall_ge07} with recall >= 0.7)"
        )
        for i, (ev_text, tr_text) in enumerate(f.samples, 1):
            print(f"    {i}. EVAL: {ev_text}")
            print(f"       TRAIN: {tr_text}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="prepared datasets (default: $AUDIOLAB_DATA, else ./data)",
    )
    parser.add_argument(
        "--panels-dir",
        type=Path,
        default=PANELS_DIR,
        help="panel files naming the evaluation sets (default: configs/panels)",
    )
    parser.add_argument("--triage", action="store_true", help="print the per-pair triage evidence")
    args = parser.parse_args(argv)
    root = args.data_dir or data_dir()
    if not root.is_dir():
        print(f"content.py: no prepared data under {root}", file=sys.stderr)
        return 2
    eval_sets, train_sets = discover(root, args.panels_dir)
    if not eval_sets or not train_sets:
        print(
            f"content.py: nothing to compare (eval={eval_sets}, train={train_sets})",
            file=sys.stderr,
        )
        return 2
    print(
        f"evaluation sets: {', '.join(eval_sets)}\ntraining sets: {', '.join(train_sets)}\n",
        file=sys.stderr,
    )
    result = audit(eval_sets, train_sets, root)
    report(result, root)
    if args.triage:
        triage(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
