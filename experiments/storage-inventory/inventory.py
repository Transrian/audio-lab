"""Plan 09 (P-07) storage inventory: list what is stored, never touch it.

Reads the three storage roots named by the environment ($AUDIOLAB_CACHE,
$AUDIOLAB_DATA, $AUDIOLAB_RUNS) and reports, per directory: size, file count,
newest file mtime, who in the repository references the name, and a
classification (keep / candidate / unknown) that plan 09 takes to the owner
(Q8). The script is strictly read-only on the storage tree: it uses
``os.scandir`` and ``stat`` only, and writes only to stdout. Nothing under
the storage roots is created, moved or renamed, not even a temporary file.

Classification follows plan 09 (``plans/09-retention-p07.md``, git-ignored):

- ``keep (rule above)`` -- the plan's "What must be kept regardless" list
  (its lines 16-28), encoded below with a comment per rule:
  * every prepared dataset used by a frozen panel (the ``# dataset:`` header
    of ``configs/panels/*.txt``) or by ``configs/train/*.yaml``;
  * every talk-source answer cache (``$AUDIOLAB_CACHE/<talk source>/``):
    the model answers are kept, the downloaded audio inside it is a
    candidate (reported separately in the talk table);
  * ``$AUDIOLAB_RUNS/train/screen-peertube-fr`` (control of plan 02);
  * ``$AUDIOLAB_DATA/mls-fr-train`` and ``$AUDIOLAB_DATA/librispeech-en-train``
    (plan 01 T4 takes its babble noise from them);
  * ``$AUDIOLAB_DATA/nasa-en-train.curly-2026-11-09`` and
    ``$AUDIOLAB_RUNS/train/screen-nasa-en``, until plan 05 Part C is
    documented;
  * the evaluation run directories cited by ``docs/stt/screening.md`` or
    ``docs/stt/evaluation.md`` (they hold the per-clip answers the docs
    quote), and the transcript cache that holds those answers; the docs can
    also quote a run's numbers without naming the directory — those runs are
    listed in ``DOC_CITED_EVAL_RUNS``, with the doc line of each citation;
- ``keep (referenced)`` -- the name appears in ``docs/``, ``configs/`` or
  ``experiments/`` (the reference check of the plan's step 2, which also
  covers the train ``dataset:`` values and the panel headers).
- ``candidate: re-creatable`` -- a download cache of a dataset that is
  prepared and whose archive checksum is pinned in the fetcher's code, so
  the archive can be downloaded again; and, per the plan's step 5, a
  screening checkpoint a committed ``configs/train/screen-*.yaml`` can
  retrain.
- ``candidate: unreferenced`` -- nothing references the name and no rule
  keeps it.
- ``unknown`` -- the script cannot say what an entry is (global rule 13:
  report it, do not improvise).

The two special cases of the plan:

- talk caches (step 4): the downloaded audio (re-downloadable from the
  pinned listings) is separated from the model answers (expensive, and what
  makes larger preparations cheap, plan 03) and both are reported per
  source; the file layout is the one ``src/audiolab/data/talks.py`` writes
  (``<talk id>.wav`` audio, ``<talk id>.<model>.jsonl`` answers, plus
  ``pages/`` of cached episode pages for NASA);
- screening checkpoints (step 5): the size of each ``model.nemo`` is
  reported, and the small evaluation answers that must stay are named.

Nothing is ever deleted, moved or renamed; the output is the only deliverable
besides this script.

Exit codes: 0 for a complete inventory; 2 when a root is missing or not a
directory (nothing is printed to stdout); 3 when some path could not be read
(the report lists it under "Warnings: paths that could not be read" and the
affected rows are measured as 0 B).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import stat as stat_module
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REPO_DIRS = ("docs", "configs", "experiments")
RUNS_DOCS = ("docs/stt/screening.md", "docs/stt/evaluation.md")

# A fetcher module pins a download when its source holds one of these: a
# literal digest table, the ESC-50 csv sha, or the Hub's published sha256 of a
# pinned revision (the huggingface helpers list it).
PIN_MARKERS = (
    "PINNED_SHA256",
    "PINNED_MD5",
    "PINNED_AUDIO_SHA256",
    "PINNED_RECORDINGS",
    "EXPECTED_CSV_SHA256",
    "huggingface_sha256_map",
    "huggingface_entries",
)

# Plan 09, "What must be kept regardless" (plans/09-retention-p07.md, lines
# 16-28): the parts of the rule that name concrete entries. The rest of the
# list (panel datasets, train datasets, talk answer caches, the evaluation
# runs the docs cite) is computed from the repository, not hard-coded.
BABBLE_NOISE_DATASETS = frozenset({"mls-fr-train", "librispeech-en-train"})
NASA_CURLY = "nasa-en-train.curly-2026-11-09"
PEERTUBE_CONTROL = "screen-peertube-fr"  # control of plan 02
NASA_SCREEN = "screen-nasa-en"  # until plan 05 Part C is documented

# Evaluation run directories the docs cite in prose WITHOUT naming the
# directory, so the name grep (eval_runs_cited_by_docs, over RUNS_DOCS) cannot
# find them. The plan's keep list covers every "evaluation run directory
# cited by docs/stt/screening.md or docs/stt/evaluation.md"; these are the
# citations that name no directory. Each value states the citation, with the
# doc line it comes from, so a reader can verify it. An entry leaves this
# table only when the citation leaves the docs.
DOC_CITED_EVAL_RUNS: dict[str, str] = {
    "diag-african-apostrophe": (
        "apostrophe diagnostic quoted in docs/stt/screening.md (line 164) "
        "without naming the directory; re-scored copy of the kept uncleaned "
        "run — its numbers are reproducible from this run only"
    ),
}

# The label-model id before the rename (talks.py's LABEL_MODEL is now
# stt-v-qwen3-asr-1.7B). Answer files under this old id were written before
# the rename; where they survive, this cache directory is the only copy.
OLD_FORMAT_LABEL_ID = "stt-a-qwen3-asr-1.7B"

TALK_AUDIO_SUFFIXES = (
    ".wav",
    ".video",
    ".partial.wav",
    ".mp4",
    ".webm",
    ".mkv",
    ".ogg",
    ".flac",
)
TALK_ANSWER_SUFFIXES = (".jsonl", ".json")


@dataclass
class Entry:
    """One inventory row: a directory (or one file) under a storage root."""

    label: str  # e.g. "$AUDIOLAB_CACHE/debconf-en-train"
    path: Path
    kind: str  # cache | data | runs | train | eval
    size: int = 0
    files: int = 0
    newest: float = 0.0
    is_summary: bool = False
    summary_of: list[str] = field(default_factory=list)
    klass: str = ""
    reasons: list[str] = field(default_factory=list)
    refs: list[str] = field(default_factory=list)
    extras: list[str] = field(default_factory=list)


@dataclass
class RepoInfo:
    """What the repository says about names: references and the keep rule."""

    root: Path
    files: list[tuple[Path, str]]
    panel_datasets: dict[str, list[str]]
    train_datasets: dict[str, list[str]]
    doc_eval_runs: dict[str, list[str]]


def name_pattern(name: str) -> re.Pattern[str]:
    """A name as an identifier: no word character or hyphen may touch it.

    Hyphens are treated as part of an identifier, so ``screen-peertube-fr``
    does not match inside ``screen-peertube-fr-train``, while a file name
    like ``screen-mls-fr.yaml`` does count as a reference.
    """
    return re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")


def _problem(problems: list[str] | None, message: str) -> None:
    """Record a path the inventory could not read: reported, never improvised."""
    if problems is not None:
        problems.append(message)


def measure(path: Path, problems: list[str] | None = None) -> tuple[int, int, float]:
    """Total bytes, file count and newest mtime of everything under ``path``.

    An ``os.scandir`` walk with ``stat`` only: it never opens a file and
    never writes anything, so the storage tree stays bit for bit untouched.
    A path that cannot be read is counted as 0 B and recorded in ``problems``
    (the report lists it, and ``main`` then exits 3 instead of 0).
    """
    try:
        info = path.stat(follow_symlinks=False)
    except OSError as error:
        _problem(problems, f"cannot stat {path}: {error}")
        return 0, 0, 0.0
    if stat_module.S_ISREG(info.st_mode):
        return info.st_size, 1, info.st_mtime
    total = 0
    count = 0
    newest = 0.0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as handle:
                children = list(handle)
        except OSError as error:
            _problem(problems, f"cannot list {current}: {error}")
            continue
        for child in children:
            try:
                info = child.stat(follow_symlinks=False)
            except OSError as error:
                _problem(problems, f"cannot stat {child.path}: {error}")
                continue
            if child.is_dir(follow_symlinks=False):
                stack.append(child.path)
            elif child.is_file(follow_symlinks=False):
                total += info.st_size
                count += 1
                if info.st_mtime > newest:
                    newest = info.st_mtime
    return total, count, newest


def repo_files(repo_root: Path, exclude_dir: Path | None = None) -> list[tuple[Path, str]]:
    """The text files of docs/, configs/ and experiments/ (the reference scope).

    The inventory's own topic directory is excluded: a tool must not count as
    a user of the directories it reports.
    """
    found: list[tuple[Path, str]] = []
    for name in REPO_DIRS:
        base = repo_root / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix == ".pyc":
                continue
            if exclude_dir is not None and exclude_dir in path.parents:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            found.append((path, text))
    return found


def find_refs(name: str, info: RepoInfo) -> list[str]:
    """Repository files (of the reference scope) that mention ``name``."""
    pattern = name_pattern(name)
    hits: list[str] = []
    for path, text in info.files:
        if pattern.search(text):
            rel = path.relative_to(info.root).as_posix()
            if rel not in hits:
                hits.append(rel)
    return sorted(hits)


def panel_datasets(panels_dir: Path) -> dict[str, list[str]]:
    """Dataset -> panel files, from the ``# dataset:`` header (panels.py format)."""
    datasets: dict[str, list[str]] = {}
    if not panels_dir.is_dir():
        return datasets
    for path in sorted(panels_dir.glob("*.txt")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if line.startswith("# dataset:"):
                name = line.removeprefix("# dataset:").strip().split(" ", 1)[0]
                datasets.setdefault(name, []).append(path.name)
                break
    return datasets


def train_datasets(train_dir: Path) -> dict[str, list[str]]:
    """Dataset -> config files, from ``dataset:`` values and mix ``name:`` entries."""
    datasets: dict[str, list[str]] = {}
    if not train_dir.is_dir():
        return datasets
    for path in sorted(train_dir.glob("*.yaml")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        in_datasets = False
        for line in lines:
            stripped = line.strip()
            if re.match(r"^dataset:\s*\S", line):
                datasets.setdefault(stripped.split(":", 1)[1].strip(), []).append(path.name)
            elif stripped == "datasets:":
                in_datasets = True
            elif in_datasets:
                if stripped.startswith("- name:"):
                    datasets.setdefault(stripped.split(":", 1)[1].strip(), []).append(
                        f"{path.name} (mix)"
                    )
                elif re.match(r"^\S", line):
                    in_datasets = False
    return datasets


def eval_runs_cited_by_docs(
    files: list[tuple[Path, str]], repo_root: Path, eval_dir: Path
) -> dict[str, list[str]]:
    """runs/eval children whose name docs/stt/screening.md or evaluation.md cite."""
    cited: dict[str, list[str]] = {}
    if not eval_dir.is_dir():
        return cited
    for path, text in files:
        rel = path.relative_to(repo_root).as_posix()
        if rel not in RUNS_DOCS:
            continue
        for child in sorted(eval_dir.iterdir()):
            if child.is_dir() and name_pattern(child.name).search(text):
                cited.setdefault(child.name, []).append(rel)
    return cited


def build_info(repo_root: Path, runs_root: Path) -> RepoInfo:
    files = repo_files(repo_root, exclude_dir=Path(__file__).resolve().parent)
    return RepoInfo(
        root=repo_root,
        files=files,
        panel_datasets=panel_datasets(repo_root / "configs" / "panels"),
        train_datasets=train_datasets(repo_root / "configs" / "train"),
        doc_eval_runs=eval_runs_cited_by_docs(files, repo_root, runs_root / "eval"),
    )


def talk_sources() -> set[str]:
    """The talk sources: the cache directories that hold their model answers."""
    from audiolab.data.talk_sources import SOURCES

    return set(SOURCES)


def fetcher_datasets() -> dict[str, str]:
    """Dataset -> fetcher module name, for every fetchable dataset."""
    from audiolab.data.fetch import FETCHERS

    return {name: function.__module__ for name, function in FETCHERS.items()}


def pinned_fetchers() -> dict[str, str]:
    """Dataset -> fetcher module, only the fetchers that pin a checksum."""
    pinned: dict[str, str] = {}
    for name, module_name in fetcher_datasets().items():
        module = importlib.import_module(module_name)
        source = Path(module.__file__).read_text(encoding="utf-8")
        if any(marker in source for marker in PIN_MARKERS):
            pinned[name] = module_name
    return pinned


def is_prepared(data_root: Path, name: str) -> bool:
    return (data_root / name / "manifest.jsonl").is_file()


def human(size: int) -> str:
    """Bytes as the owner reads them: GiB down to B, one decimal below TiB."""
    for unit, factor in (("TiB", 1 << 40), ("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if size >= factor and (unit != "KiB" or size >= 100 * factor):
            value = size / factor
            return f"{value:.1f} {unit}" if value < 10 else f"{value:.0f} {unit}"
    return f"{size} B"


def newest_label(mtime: float) -> str:
    if not mtime:
        return "-"
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")


def talk_split(path: Path, problems: list[str] | None = None) -> tuple[int, int, int]:
    """(downloaded audio, model answers, other input) under one talk cache.

    The layout is what ``src/audiolab/data/talks.py`` writes: ``<talk id>.wav``
    (and the video a stop left behind) are re-downloadable from the pinned
    listings; the per-model ``.jsonl``/``.json`` answers are the expensive
    part; everything else (NASA's cached ``pages/``, leftovers) is other input.
    """
    audio = answers = other = 0
    try:
        with os.scandir(path) as handle:
            children = list(handle)
    except OSError as error:
        _problem(problems, f"cannot list {path}: {error}")
        return 0, 0, 0
    for child in children:
        try:
            info = child.stat(follow_symlinks=False)
        except OSError as error:
            _problem(problems, f"cannot stat {child.path}: {error}")
            continue
        if child.is_dir(follow_symlinks=False):
            other += measure(Path(child.path), problems)[0]
        elif child.is_file(follow_symlinks=False):
            if child.name.endswith(TALK_ANSWER_SUFFIXES):
                answers += info.st_size
            elif child.name.endswith(TALK_AUDIO_SUFFIXES):
                audio += info.st_size
            else:
                other += info.st_size
    return audio, answers, other


def content_note(
    path: Path,
    problems: list[str] | None = None,
    cache_root: Path | None = None,
) -> str:
    """What an unrecognised cache directory holds, counted, not guessed."""
    answers = 0
    audio = 0
    other = 0
    old_format: list[str] = []
    try:
        with os.scandir(path) as handle:
            children = list(handle)
    except OSError as error:
        _problem(problems, f"cannot list {path}: {error}")
        return ""
    for child in children:
        try:
            child.stat(follow_symlinks=False)
        except OSError as error:
            _problem(problems, f"cannot stat {child.path}: {error}")
            continue
        if not child.is_file(follow_symlinks=False):
            continue
        if OLD_FORMAT_LABEL_ID in child.name:
            old_format.append(child.name)
        if child.name.endswith(TALK_ANSWER_SUFFIXES):
            answers += 1
        elif child.name.endswith(TALK_AUDIO_SUFFIXES):
            audio += 1
        else:
            other += 1
    parts = []
    if answers:
        parts.append(f"{answers} model-answer files")
    if audio:
        parts.append(f"{audio} audio files")
    if other:
        parts.append(f"{other} other files")
    note = "holds " + ", ".join(parts) + " (no current pipeline reuses it)" if parts else ""
    if old_format:
        extra = _old_format_note(path, answers + audio + other, old_format, cache_root, problems)
        if extra:
            note = f"{note}; {extra}" if note else extra
    return note


def _old_format_note(
    path: Path,
    total_files: int,
    old_format: list[str],
    cache_root: Path | None,
    problems: list[str] | None = None,
) -> str:
    """What pre-rename label-model answers this directory holds, checked, not assumed.

    The current talk caches keep the renamed id (stt-v-qwen3-asr-1.7B, the
    LABEL_MODEL in src/audiolab/data/talks.py). Whether the old-id answers
    exist anywhere else is checked in the sibling cache directories, so the
    note tells the owner exactly what deleting this directory would lose.
    """
    if cache_root is None:
        return ""
    try:
        with os.scandir(cache_root) as handle:
            siblings = [child.name for child in handle if child.is_dir(follow_symlinks=False)]
    except OSError as error:
        _problem(problems, f"cannot list {cache_root}: {error}")
        return ""
    only_here = [
        name
        for name in old_format
        if not any(
            (cache_root / sibling / name).is_file() for sibling in siblings if sibling != path.name
        )
    ]
    if not only_here:
        return ""
    return (
        f"{len(only_here)} of its {total_files} files are {OLD_FORMAT_LABEL_ID}.* "
        "answers (pre-rename model id) that exist in no other cache directory "
        "here (the current cache keeps the renamed id stt-v-qwen3-asr-1.7B): "
        "deleting this directory loses them"
    )


def df_avail(path: Path) -> str:
    """The device's free space as ``df -h`` prints it, or '' when df is absent.

    Both figures come from the same statfs field (f_bavail): this report
    shows it in binary GiB rounded to the nearest, and ``df -h`` rounds
    1024-base units up — 792.49 GiB is "792 GiB" here and "793G" there.
    The gap is display rounding only; printing both, in place, explains it
    instead of leaving it looked up. A hung df (stale mount) must not hang
    the inventory: after 5 s it is treated like df being absent.
    """
    try:
        result = subprocess.run(
            ["df", "-h", "--output=avail", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            env={**os.environ, "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    lines = result.stdout.splitlines()
    return lines[-1].strip() if len(lines) >= 2 else ""


def eval_model_name(eval_path: Path) -> str:
    """The model an evaluation run scored, from its run.json; '' when unknown."""
    for run_json in sorted(eval_path.glob("*/run.json")):
        try:
            data = json.loads(run_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        name = data.get("model", {}).get("name")
        if isinstance(name, str) and name:
            return name
    return ""


def checkpoint_extras(entry: Entry, runs_root: Path) -> None:
    """What a checkpoint row must show: the .nemo size and the answers that stay."""
    name = entry.path.name
    nemo = entry.path / "model.nemo"
    if nemo.is_file():
        entry.extras.append(f"model.nemo {human(nemo.stat().st_size)}")
    if (runs_root / "eval" / name).is_dir():
        entry.extras.append(
            f"its evaluation answers (runs/eval/{name}, the transcript cache) must stay"
        )


def classify(
    entry: Entry,
    data_root: Path,
    runs_root: Path,
    repo_train_dir: Path,
    info: RepoInfo,
    talk: set[str],
    pinned: dict[str, str],
    datasets: dict[str, str],
    problems: list[str] | None = None,
    cache_root: Path | None = None,
) -> None:
    """Plan 09's classes, in the order the plan applies them."""
    name = entry.path.name

    # -- keep (rule above): the plan's "What must be kept regardless" (16-28).
    if entry.kind == "data":
        reasons: list[str] = []
        if name in info.panel_datasets:
            reasons.append("frozen panel: " + ", ".join(info.panel_datasets[name]))
        if name in info.train_datasets:
            reasons.append("configs/train dataset: " + ", ".join(info.train_datasets[name]))
        if name in BABBLE_NOISE_DATASETS:
            reasons.append("babble noise of plan 01 T4 (plan 09 keep rule)")
        if name == NASA_CURLY:
            reasons.append("kept until plan 05 Part C is documented (plan 09 keep rule)")
        if reasons:
            entry.klass = "keep (rule above)"
            entry.reasons = reasons
            return
        if (entry.path / "manifest.jsonl").is_file():
            entry.klass = "candidate: unreferenced"
            entry.reasons = ["prepared dataset; nothing references it"]
            return
        entry.klass = "unknown"
        entry.reasons = [
            "no manifest.jsonl and unreferenced: the inventory cannot say what this is"
        ]
        return

    if entry.kind == "cache" and name in talk:
        entry.klass = "keep (rule above)"
        entry.reasons = [
            "talk-source answer cache: the model answers are kept (plan 09 keep rule); "
            "the downloaded audio is a candidate (see the talk table)"
        ]
        return

    if entry.kind == "train":
        checkpoint_extras(entry, runs_root)
        if name == PEERTUBE_CONTROL:
            entry.klass = "keep (rule above)"
            entry.reasons = ["control run of plan 02 (plan 09 keep rule)"]
            return
        if name == NASA_SCREEN:
            entry.klass = "keep (rule above)"
            entry.reasons = ["kept until plan 05 Part C is documented (plan 09 keep rule)"]
            return
        # Plan 09 step 5: the screening checkpoints. The keeps are the runs a
        # plan still uses (the two named above); the rest are candidates.
        if not name.startswith("screen-"):
            entry.klass = "unknown"
            entry.reasons = ["not a screen-* checkpoint: the inventory cannot say what this is"]
            return
        recipe = repo_train_dir / f"{name}.yaml"
        if recipe.is_file():
            entry.klass = "candidate: re-creatable"
            entry.reasons = [
                f"screen checkpoint; retrainable from configs/train/{recipe.name} and "
                "the prepared data (plan 09 step 5: only the screens a plan still "
                "uses are kept, the rest are candidates)"
            ]
        else:
            # No committed recipe names this checkpoint: it is not retrainable
            # from the repository. Say so, and let the owner decide.
            entry.klass = "candidate: unreferenced"
            entry.reasons = [
                "screen checkpoint without a configs/train recipe of its own: "
                "not retrainable from the repository (plan 09 step 5 lists it "
                "among the candidates)"
            ]
        # The same basename may still be referenced — as its evaluation run,
        # which is kept separately. Show the reference so the reader sees why
        # the checkpoint row (plan 09 step 5) is still only a candidate.
        other_refs = [ref for ref in find_refs(name, info) if ref != f"configs/train/{name}.yaml"]
        if other_refs:
            entry.extras.append(
                "the same basename is also referenced by "
                + shown_refs(other_refs)
                + f"; its evaluation run (runs/eval/{name}) is kept separately, "
                "and as a checkpoint this entry is classified by plan 09 step 5"
            )
        return

    if entry.kind == "eval" and name in info.doc_eval_runs:
        entry.klass = "keep (rule above)"
        entry.reasons = [
            "evaluation run cited by "
            + ", ".join(info.doc_eval_runs[name])
            + " (plan 09 keep rule): it holds the per-clip answers the docs quote"
        ]
        return

    # The docs can quote an evaluation run's numbers in prose without naming
    # the directory; the name grep above cannot find those. DOC_CITED_EVAL_RUNS
    # lists them, with the doc line of each citation (see the constant).
    if entry.kind == "eval" and name in DOC_CITED_EVAL_RUNS:
        entry.klass = "keep (rule above)"
        entry.reasons = [
            "evaluation run cited by the docs (plan 09 keep rule): " + DOC_CITED_EVAL_RUNS[name]
        ]
        return

    if entry.kind == "runs" and name == "transcripts":
        entry.klass = "keep (rule above)"
        entry.reasons = [
            "transcript cache: the per-clip answers of every evaluation run "
            "(docs/stt/evaluation.md; plan 09 keep rule)"
        ]
        return

    # -- Plan 09 step 3: the download caches of prepared, checksum-pinned
    # datasets are re-creatable even when the dataset name is referenced.
    if entry.kind == "cache" and name in datasets:
        module = pinned.get(name)
        if module is not None and is_prepared(data_root, name):
            entry.klass = "candidate: re-creatable"
            entry.reasons = [
                f"dataset prepared; the archive checksum is pinned in {module}: "
                "re-downloadable (plan 09 step 3)"
            ]
            return
        if module is not None:
            entry.klass = "candidate: unreferenced"
            entry.reasons = [
                "the dataset is not prepared here: the cache is the only copy of "
                f"the download, pinned in {module}"
            ]
            return
        entry.klass = "candidate: unreferenced"
        entry.reasons = [
            "dataset download without a checksum pinned in code: "
            "not re-downloadable in a verifiable way"
        ]
        return

    # -- keep (referenced): the name appears in docs/, configs/ or experiments/.
    refs = find_refs(name, info)
    if refs:
        shown = ", ".join(refs[:3]) + (f" (+{len(refs) - 3} more)" if len(refs) > 3 else "")
        entry.klass = "keep (referenced)"
        entry.reasons = [f"referenced in: {shown}"]
        entry.refs = refs
        if entry.kind == "eval":
            model = eval_model_name(entry.path)
            if model and model != name:
                entry.extras.append(f"evaluates the {model} checkpoint (run.json)")
        return

    # -- candidate: unreferenced, with what the entry holds.
    entry.klass = "candidate: unreferenced"
    entry.reasons = ["nothing in docs/, configs/ or experiments/ references it; no keep rule"]
    if entry.kind == "eval":
        model = eval_model_name(entry.path)
        if model:
            entry.extras.append(f"evaluates the {model} checkpoint (run.json)")
    if entry.kind == "cache":
        note = content_note(entry.path, problems, cache_root)
        if note:
            entry.extras.append(note)
    return


def collect(
    root: Path,
    env_name: str,
    kind: str,
    problems: list[str] | None = None,
) -> list[Entry]:
    """Every child of one storage root, measured."""
    entries: list[Entry] = []
    if not root.is_dir():
        return entries
    try:
        with os.scandir(root) as handle:
            children = sorted(handle, key=lambda child: child.name)
    except OSError as error:
        _problem(problems, f"cannot list {root}: {error}")
        return entries
    for child in children:
        entry = Entry(label=f"${env_name}/{child.name}", path=Path(child.path), kind=kind)
        entry.size, entry.files, entry.newest = measure(entry.path, problems)
        entries.append(entry)
    return entries


def summary_entry(label: str, path: Path, kind: str, children: list[Entry]) -> Entry:
    return Entry(
        label=label,
        path=path,
        kind=kind,
        is_summary=True,
        summary_of=[child.label for child in children],
        size=sum(child.size for child in children),
        files=sum(child.files for child in children),
        newest=max((child.newest for child in children), default=0.0),
    )


def shown_refs(refs: list[str]) -> str:
    if len(refs) > 3:
        return ", ".join(refs[:3]) + f" (+{len(refs) - 3} more)"
    return ", ".join(refs)


def render_table(entries: list[Entry]) -> list[str]:
    lines = [
        "| path | size | files | newest | class | references / notes |",
        "|---|---|---|---|---|---|",
    ]
    for entry in entries:
        if entry.is_summary:
            klass = "summary of the rows below"
            notes = f"{len(entry.summary_of)} entries"
        else:
            klass = entry.klass
            notes = "; ".join(entry.reasons + entry.extras)
            if entry.refs and klass != "keep (referenced)":
                notes = f"{notes}; referenced in {shown_refs(entry.refs)}"
        cells = (
            entry.label,
            human(entry.size),
            str(entry.files),
            newest_label(entry.newest),
            klass,
            notes or "-",
        )
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        help="prepared datasets (default: $AUDIOLAB_DATA)",
    )
    parser.add_argument(
        "--runs-dir",
        help="runs and transcript cache (default: $AUDIOLAB_RUNS)",
    )
    parser.add_argument(
        "--cache-dir",
        help="download cache (default: $AUDIOLAB_CACHE)",
    )
    parser.add_argument(
        "--repo-root",
        help="repository holding docs/, configs/, experiments/ "
        "(default: the repository that contains this script)",
    )
    args = parser.parse_args(argv)

    def resolve(flag: str | None, env: str) -> Path | None:
        value = flag or os.environ.get(env)
        return Path(value) if value else None

    data_root = resolve(args.data_dir, "AUDIOLAB_DATA")
    runs_root = resolve(args.runs_dir, "AUDIOLAB_RUNS")
    cache_root = resolve(args.cache_dir, "AUDIOLAB_CACHE")
    default_root = Path(__file__).resolve().parent.parent.parent
    repo_root = Path(args.repo_root) if args.repo_root else default_root
    missing = [
        name
        for name, root in (
            ("$AUDIOLAB_DATA (or --data-dir)", data_root),
            ("$AUDIOLAB_RUNS (or --runs-dir)", runs_root),
            ("$AUDIOLAB_CACHE (or --cache-dir)", cache_root),
        )
        if root is None or not root.is_dir()
    ]
    if missing:
        print(
            "error: "
            + ", ".join(missing)
            + " does not name an existing directory; load the environment "
            "(set -a; . ./.env; set +a) or pass the flags",
            file=sys.stderr,
        )
        return 2
    assert data_root is not None and runs_root is not None and cache_root is not None

    info = build_info(repo_root, runs_root)
    talk = talk_sources()
    datasets = fetcher_datasets()
    pinned = pinned_fetchers()
    problems: list[str] = []

    cache_entries = collect(cache_root, "AUDIOLAB_CACHE", "cache", problems)
    data_entries = collect(data_root, "AUDIOLAB_DATA", "data", problems)
    runs_entries = collect(runs_root, "AUDIOLAB_RUNS", "runs", problems)
    train_entries = collect(runs_root / "train", "AUDIOLAB_RUNS/train", "train", problems)
    eval_entries = collect(runs_root / "eval", "AUDIOLAB_RUNS/eval", "eval", problems)

    # train and eval are containers listed again below: show them as summary
    # rows so their size is counted once, in the class totals.
    for entry in runs_entries:
        children = (
            train_entries
            if entry.path.name == "train"
            else (eval_entries if entry.path.name == "eval" else ())
        )
        if children:
            entry.is_summary = True
            entry.summary_of = [child.label for child in children]

    repo_train = repo_root / "configs" / "train"
    for entry in cache_entries + data_entries + runs_entries + train_entries + eval_entries:
        classify(
            entry,
            data_root,
            runs_root,
            repo_train,
            info,
            talk,
            pinned,
            datasets,
            problems,
            cache_root,
        )

    # The talk table (plan 09 step 4) and the checkpoint extras.
    talk_rows: list[tuple[Entry, int, int, int]] = []
    for entry in cache_entries:
        if entry.kind == "cache" and entry.path.name in talk:
            audio, answers, other = talk_split(entry.path, problems)
            entry.extras.append(
                f"downloaded audio {human(audio)} (candidate) / "
                f"model answers {human(answers)} (keep) / other input {human(other)}"
            )
            talk_rows.append((entry, audio, answers, other))

    lines: list[str] = [
        "# Storage inventory (plan 09, P-07): list, never delete",
        "",
        f"Measured {datetime.now().astimezone().isoformat(timespec='seconds')} by "
        "`experiments/storage-inventory/inventory.py`, strictly read-only "
        "(os.scandir + stat; nothing written under the storage roots). Nothing was "
        "deleted, moved or renamed. Sizes count every file, including partial "
        "downloads and logs.",
        "",
    ]

    roots = [
        (summary_entry("$AUDIOLAB_CACHE", cache_root, "cache", cache_entries), cache_entries),
        (summary_entry("$AUDIOLAB_DATA", data_root, "data", data_entries), data_entries),
        (summary_entry("$AUDIOLAB_RUNS", runs_root, "runs", runs_entries), runs_entries),
    ]
    lines += ["| root | size | files | newest |", "|---|---|---|---|"]
    for entry, _ in roots:
        lines.append(
            f"| {entry.label} | {human(entry.size)} | {entry.files} | "
            f"{newest_label(entry.newest)} |"
        )
    lines.append("")

    # Free space of the device(s) behind the roots, re-measured now. The
    # binary-GiB figure and df -h's figure are printed together: the same
    # value, differently rounded (nearest GiB here, 1024-base units up there).
    root_pairs = [
        ("AUDIOLAB_CACHE", cache_root),
        ("AUDIOLAB_DATA", data_root),
        ("AUDIOLAB_RUNS", runs_root),
    ]
    free_lines = []
    for device in sorted({root.stat().st_dev for _, root in root_pairs}):
        on_device = [(env, root) for env, root in root_pairs if root.stat().st_dev == device]
        usage = shutil.disk_usage(on_device[0][1])
        names = ", ".join(f"${env}" for env, _ in on_device)
        line = f"{names}: {human(usage.free)} free of {human(usage.total)}"
        avail = df_avail(on_device[0][1])
        if avail:
            line += (
                f" (df -h: {avail} free — the same value; df rounds 1024-base "
                "units up, this line rounds to the nearest GiB)"
            )
        free_lines.append(line)
    lines.append("Free space: " + "; ".join(free_lines))
    lines.append("")

    sections = [
        ("## $AUDIOLAB_CACHE (download cache)", cache_entries),
        ("## $AUDIOLAB_DATA (prepared datasets)", data_entries),
        ("## $AUDIOLAB_RUNS (top level)", runs_entries),
    ]
    if train_entries:
        sections.append(("## $AUDIOLAB_RUNS/train (screening checkpoints)", train_entries))
    if eval_entries:
        sections.append(("## $AUDIOLAB_RUNS/eval (evaluation runs)", eval_entries))
    for title, entries in sections:
        lines.append(title)
        lines.append("")
        lines += render_table(entries)
        lines.append("")

    if talk_rows:
        lines.append("## Talk caches: downloaded audio vs model answers (plan 09 step 4)")
        lines.append("")
        lines.append(
            "The audio is re-downloadable from the pinned listings; the answers are "
            "what makes larger preparations cheap (plan 03) and are kept."
        )
        lines.append("")
        lines.append(
            "| talk source | total | downloaded audio (candidate) | model answers (keep) "
            "| other input |"
        )
        lines.append("|---|---|---|---|---|")
        for entry, audio, answers, other in talk_rows:
            lines.append(
                f"| {entry.label} | {human(entry.size)} | {human(audio)} | {human(answers)} "
                f"| {human(other)} |"
            )
        lines.append("")

    leaves = [
        entry
        for entries in (cache_entries, data_entries, runs_entries, train_entries, eval_entries)
        for entry in entries
        if not entry.is_summary
    ]
    classes = [
        "keep (rule above)",
        "keep (referenced)",
        "candidate: re-creatable",
        "candidate: unreferenced",
        "unknown",
    ]
    lines.append("## Totals per class (every leaf row above)")
    lines.append("")
    lines.append("| class | entries | size |")
    lines.append("|---|---|---|")
    for klass in classes:
        group = [entry for entry in leaves if entry.klass == klass]
        lines.append(f"| {klass} | {len(group)} | {human(sum(e.size for e in group))} |")
    keep = [entry for entry in leaves if entry.klass.startswith("keep")]
    lines.append(f"| **keep (both)** | {len(keep)} | {human(sum(e.size for e in keep))} |")
    lines.append(f"| **all** | {len(leaves)} | {human(sum(e.size for e in leaves))} |")
    lines.append("")

    unknown = [entry for entry in leaves if entry.klass == "unknown"]
    lines.append("## Unknown (global rule 13: reported, not improvised)")
    lines.append("")
    if unknown:
        for entry in unknown:
            lines.append(f"- `{entry.label}`: {'; '.join(entry.reasons)}")
    else:
        lines.append("none: every entry was classified.")
    lines.append("")

    if problems:
        # The affected rows are measured as 0 B above; the list below says
        # exactly which paths could not be read, so the report is honest.
        lines.append("## Warnings: paths that could not be read")
        lines.append("")
        for problem in problems:
            lines.append(f"- {problem}")
        lines.append("")

    print("\n".join(lines))
    return 3 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
