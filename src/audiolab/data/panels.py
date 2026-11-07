"""Panels: the frozen id lists a run is scored on (decision 0002).

A panel is a list of clip ids, never a list of texts or files: it points into a
manifest that lives in the data directory, so the repository can hold the panel
without holding any dataset content.

The three panels of a dataset are built once and then frozen. Building them
again from another manifest (a different download, a changed selection rule)
would give different files, which is why ``build`` refuses to overwrite an
existing panel and ``check`` compares a panel with what the current data and
the stored seed would produce.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from audiolab.core.paths import data_dir
from audiolab.data.store import Clip, manifest_id_sha256, read_manifest

PANEL_NAMES = ("fast", "full", "sealed")
DEFAULT_SEED = 20261107
DEFAULT_PANELS_DIR = Path("configs/panels")
FULL_CLIPS = 600
FAST_CLIPS = 100
SEALED_GROUP_SHARE = 0.2
# The non-speech battery is small enough to size directly: sealed 60, full 240, fast 60.
NONSPEECH_CLIPS = {"sealed": 60, "full": 240, "fast": 60}
NONSPEECH_FIRST = ("silence", "noise")  # ordered; ESC-50 classes come last


@dataclass(frozen=True)
class Panel:
    """One frozen id list plus what is needed to rebuild and check it."""

    name: str
    dataset: str
    ids: list[str]
    seed: int
    manifest_sha256: str
    built: str

    @property
    def title(self) -> str:
        return f"{self.dataset}-{self.name}"

    def text(self) -> str:
        header = "\n".join(
            [
                f"# panel: {self.title}",
                f"# dataset: {self.dataset}",
                f"# manifest_sha256: {self.manifest_sha256}  (sha256 of the sorted manifest ids)",
                f"# seed: {self.seed}",
                f"# built: {self.built}  (date the panel was frozen; never rebuilt)",
                "# one clip id per line; audio and reference text live under $AUDIOLAB_DATA",
            ]
        )
        return header + "\n" + "\n".join(self.ids) + "\n"


def panel_path(title: str, panels_dir: Path = DEFAULT_PANELS_DIR) -> Path:
    return panels_dir / f"{title}.txt"


def build_panels(
    dataset: str,
    clips: list[Clip],
    panels_dir: Path = DEFAULT_PANELS_DIR,
    built: str | None = None,
) -> list[Panel]:
    """Build the three panels of a dataset, refusing to replace existing files."""
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
    panels = [
        Panel(
            name=name,
            dataset=dataset,
            ids=chosen[name],
            seed=DEFAULT_SEED,
            manifest_sha256=manifest_id_sha256(clips),
            built=built or date.today().isoformat(),
        )
        for name in PANEL_NAMES
    ]
    panels_dir.mkdir(parents=True, exist_ok=True)
    for panel in panels:
        panel_path(panel.title, panels_dir).write_text(panel.text(), encoding="utf-8")
    return panels


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
    """Sealed clips first, then a full battery and a fast subset of it.

    The battery has 300 clips, so the fast panel is a subset of the full one
    rather than a third disjoint list: 60 + 240 + 60 clips would need 360. Both
    come from the same order, so the fast panel keeps the proportions of the
    full one, which is what a shorter run needs.
    """
    ordered = _battery_order(clips, seed)
    end_of_sealed = NONSPEECH_CLIPS["sealed"]
    full = ordered[end_of_sealed : end_of_sealed + NONSPEECH_CLIPS["full"]]
    return {
        "sealed": ordered[:end_of_sealed],
        "full": full,
        "fast": _evenly_spaced(full, NONSPEECH_CLIPS["fast"]),
    }


def _evenly_spaced(items: list[str], count: int) -> list[str]:
    """Take ``count`` items spread over a list, without favouring its start."""
    step = len(items) / count
    return sorted(items[int(index * step)] for index in range(count))


def _battery_order(clips: list[Clip], seed: int) -> list[str]:
    """Silences, then noises, then ESC-50 sounds, in a shuffled order within each family."""
    ids = sorted(clip.id for clip in clips)
    random.Random(_panel_seed(seed, "nonspeech")).shuffle(ids)
    return sorted(ids, key=_family_rank)  # stable: the shuffle survives within a family


def _family_rank(clip_id: str) -> int:
    """Silence first, noise second, everything else (ESC-50) last."""
    kind = clip_id.split(":", 1)[1].split("-")[0]
    return NONSPEECH_FIRST.index(kind) if kind in NONSPEECH_FIRST else len(NONSPEECH_FIRST)


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
            if keyword.strip() in ("panel", "dataset", "manifest_sha256", "seed", "built"):
                header[keyword.strip()] = value.split("(")[0].strip()
        elif line.strip():
            ids.append(line.strip())
    missing = [key for key in ("dataset", "manifest_sha256", "seed", "built") if key not in header]
    if missing:
        raise ValueError(f"{path} has no header field {missing}")
    name = path.stem.rsplit("-", 1)[-1]
    if name not in PANEL_NAMES:
        raise ValueError(f"{path} is not named <dataset>-<panel>.txt")
    return Panel(
        name=name,
        dataset=header["dataset"],
        ids=ids,
        seed=int(header["seed"]),
        manifest_sha256=header["manifest_sha256"],
        built=header["built"],
    )


def check_panel(path: Path, clips: list[Clip]) -> list[str]:
    """What makes a panel unusable with the current data; an empty list is good."""
    panel = read_panel(path)
    problems = []
    current = manifest_id_sha256(clips)
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
    return problems


def sealed_is_disjoint(clips: list[Clip], sealed_ids: list[str]) -> bool:
    """True when no clip outside the sealed panel shares a group with a sealed clip."""
    chosen = set(sealed_ids)
    sealed_groups = {clip.group for clip in clips if clip.id in chosen}
    other_groups = {clip.group for clip in clips if clip.id not in chosen}
    return not (sealed_groups & other_groups)


def load_dataset_clips(dataset: str) -> list[Clip]:
    """Read the manifest of one prepared dataset from the data directory."""
    return read_manifest(data_dir() / dataset)
