"""Which clips a run trains on, and the manifest NeMo reads them from.

The clips are drawn once, with a fixed seed, from a prepared dataset - or from
each dataset of a mix, with one seed per dataset - and the list is written into
the run directory. The same config therefore trains on the same clips however
often it is run, and ``train.json`` records their hash, so a later reader can
check that a reported result came from the clips it claims.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from audiolab.data.store import Clip, manifest_id_sha256, read_manifest
from audiolab.train.config import DatasetShare

# Shorter clips are usually fragments that hurt more than they teach; a clip
# whose text is empty (silence, noise) has nothing to learn from at all.
MIN_CLIP_SECONDS = 0.3


class TrainDataError(ValueError):
    """The dataset cannot provide the clips this run asks for."""


@dataclass(frozen=True)
class TrainingClips:
    """The clips of one run, and why the others were left out."""

    clips: list[Clip]
    # Clips that were candidates: inside the duration window, with text.
    available: int
    # Clips shorter than MIN_CLIP_SECONDS or longer than the run's ceiling.
    outside_duration: int
    without_text: int

    @property
    def ids_sha256(self) -> str:
        """A hash of the clip ids, which is what the run record keeps."""
        return manifest_id_sha256(self.clips)


def select_clips(clips: list[Clip], *, count: int, seed: int, max_seconds: float) -> TrainingClips:
    """Draw ``count`` clips, the same ones every time for the same seed.

    The order the dataset happens to be in must not change the answer, so the
    candidates are sorted by id before anything is drawn.
    """
    usable = [clip for clip in sorted(clips, key=lambda clip: clip.id) if clip.text.strip()]
    without_text = len(clips) - len(usable)
    within = [clip for clip in usable if MIN_CLIP_SECONDS <= clip.duration_s <= max_seconds]
    outside_duration = len(usable) - len(within)
    if len(within) < count:
        raise TrainDataError(
            f"the dataset holds {len(within)} clips between {MIN_CLIP_SECONDS}s and "
            f"{max_seconds}s, but this run asks for {count}; fetch more data or lower "
            "clips or max_clip_seconds"
        )
    drawn = random.Random(seed).sample(within, count)
    return TrainingClips(
        clips=sorted(drawn, key=lambda clip: clip.id),
        available=len(within),
        outside_duration=outside_duration,
        without_text=without_text,
    )


@dataclass(frozen=True)
class MixDatasetClips:
    """What one dataset of a mix contributed to the run."""

    name: str
    clips: list[Clip]
    available: int
    ids_sha256: str


@dataclass(frozen=True)
class MixClips:
    """The clips of a mix run: the per-dataset results and their concatenation."""

    clips: list[Clip]
    datasets: tuple[MixDatasetClips, ...]

    @property
    def ids_sha256(self) -> str:
        """A hash of the clip ids of the whole mix, which is what the run record keeps."""
        return manifest_id_sha256(self.clips)


def mix_seed(seed: int, name: str) -> int:
    """The per-dataset seed of a mix: the same seed for that dataset in every mix.

    Deriving it from the run's seed and the dataset's name - and from nothing
    else - is what makes adding a dataset to a mix leave the clips drawn from
    the other datasets untouched.
    """
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")


def select_mix(
    shares: tuple[DatasetShare, ...], *, seed: int, max_seconds: float, data_dir: Path | None
) -> MixClips:
    """Draw a mix run's clips: one draw per dataset, in the config's order.

    Each dataset is drawn with :func:`mix_seed`, so the mix is the union of
    what each dataset would have drawn alone. The result is sorted by id: ids
    are prefixed by their dataset name, so they are unique, and that is
    asserted. The same audio in two datasets is refused: training on it twice
    would count the same clips against two shares.
    """
    results: list[MixDatasetClips] = []
    for share in shares:
        directory = dataset_dir(share.name, data_dir)
        try:
            drawn = select_clips(
                read_manifest(directory),
                count=share.clips,
                seed=mix_seed(seed, share.name),
                max_seconds=max_seconds,
            )
        except TrainDataError as failure:
            raise TrainDataError(f"{share.name}: {failure}") from failure
        results.append(
            MixDatasetClips(
                name=share.name,
                clips=drawn.clips,
                available=drawn.available,
                ids_sha256=drawn.ids_sha256,
            )
        )
    all_clips = [clip for per in results for clip in per.clips]
    all_clips.sort(key=lambda clip: clip.id)
    ids = [clip.id for clip in all_clips]
    if len(set(ids)) != len(ids):
        counts = Counter(ids)
        repeated = sorted(clip_id for clip_id, number in counts.items() if number > 1)
        raise TrainDataError(f"clip ids are not unique across the mix (first: {repeated[0]})")
    # The same audio prepared twice: one clip in each of two datasets. The ids
    # are unique (they are prefixed by the dataset), so the audio hash is what
    # catches it. A repetition inside one dataset is that dataset's business,
    # not the mix's.
    by_sha: dict[str, dict[str, list[str]]] = {}
    for per in results:
        for clip in per.clips:
            by_sha.setdefault(clip.sha256, {}).setdefault(per.name, []).append(clip.id)
    duplicates = sorted(
        (sha, per_dataset) for sha, per_dataset in by_sha.items() if len(per_dataset) > 1
    )
    if duplicates:
        listing = "; ".join(
            f"{sha[:12]}…: "
            + " vs ".join(
                f"{name} ({', '.join(clip_ids)})" for name, clip_ids in sorted(per_dataset.items())
            )
            for sha, per_dataset in duplicates[:5]
        )
        raise TrainDataError(
            f"the same audio is in several datasets of the mix ({len(duplicates)} pairs of "
            f"duplicate clips, first 5: {listing}); prepare the datasets so each recording "
            "appears in only one of them"
        )
    return MixClips(clips=all_clips, datasets=tuple(results))


def load_training_clips(
    dataset_dir: Path, *, count: int, seed: int, max_seconds: float
) -> TrainingClips:
    """Read a prepared dataset's manifest and draw this run's clips from it."""
    return select_clips(read_manifest(dataset_dir), count=count, seed=seed, max_seconds=max_seconds)


def write_nemo_manifest(path: Path, entries: list[tuple[Clip, Path]]) -> int:
    """Write the manifest NeMo reads: one line per clip, with absolute audio paths.

    ``entries`` is a pair per clip: the clip and the dataset directory its
    audio lives in. A single-dataset run pairs every clip with the one
    directory, a mix run with each clip's own dataset, so one format serves
    both. The order of the entries is the order of the manifest, so a run
    gives it its clips sorted by id.

    NeMo's own manifest format is three fields - where the audio is, how long it
    is, and what was said - and nothing else. The audio path is absolute because
    NeMo opens it from wherever the training process happens to run.
    """
    lines = [
        json.dumps(
            {
                "audio_filepath": str((dataset_dir / clip.audio).resolve()),
                "duration": round(clip.duration_s, 3),
                "text": clip.text,
            },
            ensure_ascii=False,
        )
        for clip, dataset_dir in entries
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def write_single_nemo_manifest(path: Path, clips: list[Clip], dataset_dir: Path) -> int:
    """The manifest of a single-dataset run: every clip in the one directory."""
    return write_nemo_manifest(path, [(clip, dataset_dir) for clip in clips])


def dataset_dir(name: str, data_dir: Path | None = None) -> Path:
    """Where a prepared dataset lives; refused if nothing has prepared it."""
    from audiolab.core.paths import data_dir as default_data_dir

    root = Path(data_dir) if data_dir else default_data_dir()
    directory = root / name
    if not (directory / "manifest.jsonl").is_file():
        raise TrainDataError(
            f"{directory} holds no manifest.jsonl: prepare the dataset first "
            f"(audiolab data fetch {name})"
        )
    return directory
