"""Panels: the frozen id lists a run is scored on (decision 0002).

A panel is a list of clip ids, never a list of texts or files: it points into a
manifest that lives in the data directory, so the repository can hold the panel
without holding any dataset content.

The three panels of a dataset are built once and then frozen. Building them
again from another manifest (a different download, a changed selection rule)
would give different files, which is why ``build`` refuses to overwrite an
existing panel and ``check`` compares a panel with what the current data and
the stored seed would produce.

A **suite** (``configs/suites/<name>.yaml``) is the list of panels one report is
built from. It is read here, next to the panels it names, so that the evaluation
command and the training leakage guard read it the same way.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from audiolab.core.paths import data_dir
from audiolab.data.store import (
    Clip,
    manifest_id_sha256,
    pcm_sha256_of,
    read_manifest,
    verify_clips,
)

PANEL_NAMES = ("fast", "full", "sealed")
DEFAULT_SEED = 20261107
DEFAULT_PANELS_DIR = Path("configs/panels")
FULL_CLIPS = 600
FAST_CLIPS = 100
SEALED_GROUP_SHARE = 0.2
# The non-speech battery is sized per family, so that every panel holds every
# family. Only the clips generated here can be sealed: ESC-50 is public and
# widely copied, so a sealed copy of it would prove little.
NONSPEECH_SEALED = {"silence": 10, "noise": 10, "esc50": 0}
NONSPEECH_FAST = {"silence": 10, "noise": 10, "esc50": 40}


@dataclass(frozen=True)
class Panel:
    """One frozen id list plus what is needed to rebuild and check it.

    The header records what the panel is (dataset and its real size, the split
    it came from), what the panel holds (its item count, the audio hash of
    exactly those items) and which manifest and seed produced it, so a reader
    can check the file without the data directory and a scoring run can check
    the audio it is about to use.
    """

    name: str
    dataset: str
    ids: list[str]
    items: int  # what the header declares, read back so the file can be checked
    seed: int
    manifest_sha256: str
    clips_sha256: str
    dataset_line: str
    built: str

    @property
    def title(self) -> str:
        return f"{self.dataset}-{self.name}"

    def text(self) -> str:
        header = "\n".join(
            [
                f"# panel: {self.title}",
                f"# dataset: {self.dataset_line}",
                f"# items: {len(self.ids)}  (clip ids below, one per line)",
                f"# clips_sha256: {self.clips_sha256}  (sha256 of the sorted id/audio hashes)",
                f"# manifest_sha256: {self.manifest_sha256}  (sha256 of the sorted manifest ids)",
                f"# seed: {self.seed}",
                f"# built: {self.built}  (date the panel was frozen; never rebuilt)",
                "# audio and reference text live under $AUDIOLAB_DATA",
            ]
        )
        return header + "\n" + "\n".join(self.ids) + "\n"


def panel_path(title: str, panels_dir: str | Path = DEFAULT_PANELS_DIR) -> Path:
    """Where one panel lives; the CLI passes the directory as a string."""
    return Path(panels_dir) / f"{title}.txt"


@dataclass(frozen=True)
class SuiteSpec:
    """The panels a suite scores."""

    name: str
    panels: list[str]


class SuiteError(ValueError):
    """A suite file that cannot be used as written."""


def load_suite(path: str | Path) -> SuiteSpec:
    """Read ``configs/suites/<name>.yaml``.

    One reader for the evaluation command and for the training leakage guard: a
    suite the eval command accepts is exactly the suite the guard checks, so the
    two cannot drift apart when the format changes.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("panels"), list) or not raw["panels"]:
        raise SuiteError(f"{path} must hold a non-empty 'panels' list")
    name = str(raw.get("name") or path.stem)
    return SuiteSpec(name=name, panels=[str(panel) for panel in raw["panels"]])


def build_panels(
    dataset: str,
    clips: list[Clip],
    panels_dir: str | Path = DEFAULT_PANELS_DIR,
    built: str | None = None,
) -> list[Panel]:
    """Build the three panels of a dataset, refusing to replace existing files."""
    panels_dir = Path(panels_dir)
    for name in PANEL_NAMES:
        path = panel_path(f"{dataset}-{name}", panels_dir)
        if path.is_file():
            raise FileExistsError(
                f"{path} already exists: panels are frozen (decision 0002). "
                "Remove the file only if the owner asked for new panels."
            )
    if not clips:
        raise ValueError(f"cannot build panels for {dataset}: the manifest is empty")
    chosen = select_panels(clips, DEFAULT_SEED)
    by_id = {clip.id: clip for clip in clips}
    panels = [
        Panel(
            name=name,
            dataset=dataset,
            ids=chosen[name],
            items=len(chosen[name]),
            seed=DEFAULT_SEED,
            manifest_sha256=manifest_id_sha256(clips),
            clips_sha256=pcm_sha256_of([by_id[clip_id] for clip_id in chosen[name]]),
            dataset_line=dataset_line(dataset, clips),
            built=built or date.today().isoformat(),
        )
        for name in PANEL_NAMES
    ]
    panels_dir.mkdir(parents=True, exist_ok=True)
    for panel in panels:
        panel_path(panel.title, panels_dir).write_text(panel.text(), encoding="utf-8")
    return panels


def dataset_line(dataset: str, clips: list[Clip]) -> str:
    """The dataset's real size, per split, as the panel header records it.

    A reader of a panel should not have to guess how much audio it came from,
    nor assume one split: the non-speech battery is generated clips plus ESC-50,
    and a panel of it is drawn from both.
    """
    groups = len({clip.group for clip in clips})
    splits = Counter(clip.source_split for clip in clips)
    parts = ", ".join(f"{split} {count}" for split, count in sorted(splits.items()))
    return f"{dataset} ({len(clips)} clips in {groups} groups; {parts})"


def select_panels(clips: list[Clip], seed: int) -> dict[str, list[str]]:
    """The ids of the three panels: a pure function of the clips and the seed."""
    if clips[0].language == "none":
        return _select_nonspeech(clips, seed)
    return _select_speech(clips, seed)


def _select_speech(clips: list[Clip], seed: int) -> dict[str, list[str]]:
    """A fifth of the groups sealed, up to 600 utterances, then 100 of those.

    Entire groups go to the sealed panel, so no speaker (or sentence) that the
    sealed number covers is scored anywhere else.
    """
    members = _members_by_group(clips)
    groups = sorted(members)
    sealed = _take(groups, _sealed_group_count(len(groups)), seed)
    remaining = [group for group in groups if group not in set(sealed)]
    return {
        "sealed": _flatten(sealed, members),
        "full": _take_balanced(remaining, members, FULL_CLIPS, seed),
        "fast": _fast_subset(remaining, members, seed),
    }


def _select_nonspeech(clips: list[Clip], seed: int) -> dict[str, list[str]]:
    """Sealed, full and fast panels drawn family by family.

    Families are silence, noise and ESC-50 sounds. The whole battery is shuffled
    once and split by family, keeping the shuffled order; the first clips of a
    family are sealed and the rest form its full panel. The battery has 300
    clips, too few for three disjoint panels, so the fast panel is an evenly
    spread subset of the full one. Drawing per family guarantees that silences,
    the case where models are known to invent text, are scored in every panel.
    """
    panels: dict[str, list[str]] = {"sealed": [], "full": [], "fast": []}
    for family, ids in sorted(_families(clips, seed).items()):
        sealed_count = NONSPEECH_SEALED[family]
        full = ids[sealed_count:]
        panels["sealed"] += ids[:sealed_count]
        panels["full"] += full
        panels["fast"] += _evenly_spaced(full, NONSPEECH_FAST[family])
    return {name: sorted(ids) for name, ids in panels.items()}


def _families(clips: list[Clip], seed: int) -> dict[str, list[str]]:
    """Shuffled clip ids per family; the family is the first word of the id."""
    ids = sorted(clip.id for clip in clips)
    random.Random(_panel_seed(seed, "nonspeech")).shuffle(ids)
    families: dict[str, list[str]] = defaultdict(list)
    for clip_id in ids:
        family = clip_id.split(":", 1)[1].split("-")[0]
        if family not in NONSPEECH_SEALED:
            raise ValueError(f"{clip_id}: unknown non-speech family {family!r}")
        families[family].append(clip_id)
    return families


def _evenly_spaced(items: list[str], count: int) -> list[str]:
    """Take ``count`` items spread over a list, without favouring its start."""
    step = len(items) / count
    return [items[int(index * step)] for index in range(count)]


def _members_by_group(clips: list[Clip]) -> dict[str, list[str]]:
    """Clip ids per group, sorted by id, so the choice never depends on manifest order."""
    members: dict[str, list[str]] = defaultdict(list)
    for clip in sorted(clips, key=lambda clip: clip.id):
        members[clip.group].append(clip.id)
    return dict(members)


def _sealed_group_count(group_count: int) -> int:
    """Twenty percent of the groups, at least one, and never all of them."""
    return max(1, min(group_count - 1, round(group_count * SEALED_GROUP_SHARE)))


def _take(groups: list[str], count: int, seed: int) -> list[str]:
    """Pick groups deterministically from an already sorted list."""
    chosen = random.Random(_panel_seed(seed, "sealed")).sample(groups, count)
    return sorted(chosen)


def _panel_seed(seed: int, panel: str) -> int:
    """One seed per panel, so changing one panel's rule does not move the others."""
    digest = hashlib.sha256(f"{seed}:{panel}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _take_balanced(
    groups: list[str], members: dict[str, list[str]], limit: int, seed: int
) -> list[str]:
    """Take one utterance per group in turn until ``limit`` is reached.

    Taking one group after another instead would hand a whole panel to the
    speaker with the most recordings, so groups are served in turn and a group
    with fewer utterances simply stops contributing. When the groups are
    exhausted the panel is shorter than ``limit``: the sealed groups are
    reserved, and 20 % sealed plus 600 utterances cannot both hold on a small
    dataset. Panel sizes are recorded in the panel header, so a reader sees the
    real size instead of an assumption.
    """
    order = _rank(groups, seed)
    taken = dict.fromkeys(order, 0)
    chosen: list[str] = []
    while len(chosen) < limit:
        added = 0
        for group in order:
            if len(chosen) < limit and taken[group] < len(members[group]):
                chosen.append(members[group][taken[group]])
                taken[group] += 1
                added += 1
        if added == 0:
            break  # no group has an utterance left
    return sorted(chosen)


def _fast_subset(groups: list[str], members: dict[str, list[str]], seed: int) -> list[str]:
    """One utterance from each group in turn, which spreads 100 items over many groups."""
    return _take_balanced(groups, members, FAST_CLIPS, seed)


def _rank(groups: list[str], seed: int) -> list[str]:
    """A deterministic shuffling of group names, independent of the input order."""
    shuffled = sorted(set(groups))
    random.Random(_panel_seed(seed, "full")).shuffle(shuffled)
    return shuffled


def _flatten(groups: list[str], members: dict[str, list[str]]) -> list[str]:
    return sorted(clip_id for group in groups for clip_id in members[group])


def read_panel(path: Path) -> Panel:
    """Read a panel file back into its ids and its header values."""
    header: dict[str, str] = {}
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            keyword, _, value = line[2:].partition(":")
            keyword = keyword.strip()
            if keyword == "dataset":
                # The size in that line is part of it: it is checked, not decoration.
                header["dataset"] = value.strip()
            elif keyword in (
                "panel",
                "items",
                "clips_sha256",
                "manifest_sha256",
                "seed",
                "built",
            ):
                header[keyword] = value.split("(")[0].strip()
        elif line.strip():
            ids.append(line.strip())
    wanted = ("dataset", "items", "clips_sha256", "manifest_sha256", "seed", "built")
    missing = [key for key in wanted if key not in header]
    if missing:
        raise ValueError(f"{path} has no header field {missing}")
    name = path.stem.rsplit("-", 1)[-1]
    if name not in PANEL_NAMES:
        raise ValueError(f"{path} is not named <dataset>-<panel>.txt")
    return Panel(
        name=name,
        dataset=header["dataset"].split(" ")[0],
        ids=ids,
        items=int(header["items"]),
        seed=int(header["seed"]),
        manifest_sha256=header["manifest_sha256"],
        clips_sha256=header["clips_sha256"],
        dataset_line=header["dataset"],
        built=header["built"],
    )


def check_panel(path: Path, clips: list[Clip], dataset_dir: Path | None = None) -> list[str]:
    """What makes a panel unusable with the current data; an empty list is good.

    With ``dataset_dir``, the audio of the panel's own clips is hashed as well:
    a panel names audio, so a panel whose ids are right and whose files were
    replaced is still not the panel an earlier run scored.
    """
    panel = read_panel(path)
    problems = []
    current = manifest_id_sha256(clips)
    if panel.dataset_line != dataset_line(panel.dataset, clips):
        problems.append(
            f"header says {panel.dataset_line!r}, the manifest says "
            f"{dataset_line(panel.dataset, clips)!r}"
        )
    if panel.manifest_sha256 != current:
        problems.append(
            f"prepared from a different manifest ({panel.manifest_sha256[:12]} != {current[:12]})"
        )
    known = {clip.id for clip in clips}
    unknown = [clip_id for clip_id in panel.ids if clip_id not in known]
    if unknown:
        problems.append(f"{len(unknown)} ids are not in the manifest (first: {unknown[0]})")
    expected = select_panels(clips, panel.seed)[panel.name]
    if panel.ids != expected:
        problems.append(f"ids differ from what the current data and seed {panel.seed} produce")
    if panel.items != len(panel.ids):
        problems.append(f"the header says {panel.items} items, the file lists {len(panel.ids)}")
    chosen = [clip for clip in clips if clip.id in set(panel.ids)]
    current_audio = pcm_sha256_of(chosen)
    if panel.clips_sha256 != current_audio:
        problems.append(
            f"the audio of these ids is not the audio the panel recorded "
            f"({panel.clips_sha256[:12]} != {current_audio[:12]})"
        )
    if dataset_dir is not None:
        problems.extend(verify_clips(dataset_dir, chosen))
    return problems


def load_dataset_clips(dataset: str) -> list[Clip]:
    """Read the manifest of one prepared dataset from the data directory."""
    return read_manifest(data_dir() / dataset)
