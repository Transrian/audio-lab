"""Which clips a run trains on, and the manifest NeMo reads them from.

The clips are drawn once, with a fixed seed, from a prepared dataset, and the
list is written into the run directory. The same config therefore trains on the
same clips however often it is run, and ``train.json`` records their hash, so a
later reader can check that a reported result came from the clips it claims.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from audiolab.data.store import Clip, manifest_id_sha256, read_manifest

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


def load_training_clips(
    dataset_dir: Path, *, count: int, seed: int, max_seconds: float
) -> TrainingClips:
    """Read a prepared dataset's manifest and draw this run's clips from it."""
    return select_clips(read_manifest(dataset_dir), count=count, seed=seed, max_seconds=max_seconds)


def write_nemo_manifest(path: Path, clips: list[Clip], dataset_dir: Path) -> int:
    """Write the manifest NeMo reads: one line per clip, with absolute audio paths.

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
        for clip in clips
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


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
