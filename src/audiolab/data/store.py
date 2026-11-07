"""The manifest: one JSONL line per prepared clip.

The manifest is the only thing the rest of the code knows about a dataset, so
every field a metric or a panel needs has to be in it. Reference texts live
here too, in the data directory, never in the repository (decision 0009).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from audiolab.core.audio import pcm_sha256

MANIFEST_NAME = "manifest.jsonl"

LANGUAGES = ("fr", "en", "none")
GROUP_KINDS = ("speaker", "sentence", "clip")


@dataclass(frozen=True)
class Clip:
    """One prepared utterance: where its audio is, what was said, how to group it."""

    id: str
    audio: str
    sha256: str
    duration_s: float
    language: str
    text: str
    text_has_casing_punct: bool
    group: str
    group_kind: str
    source_split: str
    license: str
    category: str = ""

    def manifest_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def clip_from_dict(data: dict) -> Clip:
    """Rebuild a clip, refusing unknown or missing fields instead of guessing."""
    unknown = set(data) - set(Clip.__dataclass_fields__)
    if unknown:
        raise ValueError(f"manifest line has unknown fields: {sorted(unknown)}")
    missing = set(Clip.__dataclass_fields__) - set(data) - {"category"}
    if missing:
        raise ValueError(f"manifest line is missing fields: {sorted(missing)}")
    return Clip(**data)


def manifest_path(dataset_dir: Path) -> Path:
    return dataset_dir / MANIFEST_NAME


def write_manifest(dataset_dir: Path, clips: list[Clip]) -> Path:
    """Write the manifest, one line per clip, sorted by id."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = manifest_path(dataset_dir)
    lines = [clip.manifest_line() for clip in sorted(clips, key=lambda clip: clip.id)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def read_manifest(dataset_dir: Path) -> list[Clip]:
    """Read the manifest of a prepared dataset."""
    path = manifest_path(dataset_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"{dataset_dir} is not prepared: no {MANIFEST_NAME} (run 'audiolab data fetch')"
        )
    with path.open(encoding="utf-8") as handle:
        return [clip_from_dict(json.loads(line)) for line in handle if line.strip()]


def verify_clips(dataset_dir: Path, clips: list[Clip]) -> list[str]:
    """Return one message per clip whose audio is missing or no longer matches its hash."""
    problems = []
    for clip in clips:
        path = dataset_dir / clip.audio
        if not path.is_file():
            problems.append(f"{clip.id}: missing audio {clip.audio}")
        elif pcm_sha256(path) != clip.sha256:
            problems.append(f"{clip.id}: audio {clip.audio} does not match its sha256")
    return problems


def manifest_id_sha256(clips: list[Clip]) -> str:
    """Hash of the sorted clip ids: identifies the content of a manifest."""
    joined = "\n".join(clip.id for clip in sorted(clips, key=lambda clip: clip.id))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
