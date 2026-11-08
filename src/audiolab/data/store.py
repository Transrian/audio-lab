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
GROUP_KINDS = ("speaker", "recording", "sentence", "clip")


@dataclass(frozen=True)
class Clip:
    """One prepared utterance: where its audio is, what was said, how to group it.

    ``source_url``, ``source_title`` and ``source_author`` are empty unless the
    source's license asks for attribution per recording (CC BY on a video): the
    manifest line is what travels with the clip, so the attribution goes there.
    ``base_wer`` is set only where a model wrote the label: the normalized word
    error rate of the base model on the clip, against that label, so that a
    training mix can pick the clips the base model gets wrong.
    """

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
    source_url: str = ""
    source_title: str = ""
    source_author: str = ""
    base_wer: float | None = None

    def manifest_line(self) -> str:
        """The clip as one JSON line, without the per-recording fields it does not use.

        So the manifests of the sources that have none read as they always did.
        """
        fields = asdict(self)
        for name in PER_RECORDING_FIELDS:
            if fields[name] == Clip.__dataclass_fields__[name].default:
                del fields[name]
        return json.dumps(fields, ensure_ascii=False, sort_keys=True)


# The fields only some sources fill in, left out of a manifest line when unset.
PER_RECORDING_FIELDS = ("source_url", "source_title", "source_author", "base_wer")


def clip_from_dict(data: dict) -> Clip:
    """Rebuild a clip, refusing unknown or missing fields instead of guessing."""
    optional = {"category", *PER_RECORDING_FIELDS}
    unknown = set(data) - set(Clip.__dataclass_fields__)
    if unknown:
        raise ValueError(f"manifest line has unknown fields: {sorted(unknown)}")
    missing = set(Clip.__dataclass_fields__) - set(data) - optional
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


def pcm_sha256_of(clips: list[Clip]) -> str:
    """Hash of the sorted ``<id> <audio sha256>`` pairs of some clips.

    Two manifests can hold the same ids with different audio (a re-encoded
    download, a changed decoder), so a panel records this hash next to the ids:
    it is what an evaluation actually scored.
    """
    joined = "\n".join(
        f"{clip.id} {clip.sha256}" for clip in sorted(clips, key=lambda clip: clip.id)
    )
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
