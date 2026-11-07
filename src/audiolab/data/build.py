"""Shared helpers for turning downloaded files into a manifest."""

from __future__ import annotations

import re
from pathlib import Path

from audiolab.core.audio import Audio, pcm_sha256, write_wav
from audiolab.data.store import Clip, write_manifest

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def file_name_for(clip_id: str) -> str:
    """A file name for a clip id: the id may contain ':' and '/', the file may not."""
    return _UNSAFE.sub("_", clip_id) + ".wav"


class ManifestBuilder:
    """Collects prepared clips, writes each WAV at once, and reports progress."""

    def __init__(self, dataset_dir: Path, *, verbose: bool) -> None:
        self.dataset_dir = dataset_dir
        self.verbose = verbose
        self.clips: list[Clip] = []

    def add(self, clip_id: str, audio: Audio, **fields) -> None:
        """Store one clip's audio and remember its manifest line.

        The WAV is written immediately so that a preparation interrupted in the
        middle leaves the finished clips usable, and the manifest, written at
        the end, never advertises a clip that is not on disk.
        """
        relative = Path("audio") / file_name_for(clip_id)
        write_wav(self.dataset_dir / relative, audio)
        self.clips.append(
            Clip(
                id=clip_id,
                audio=str(relative),
                sha256=pcm_sha256(self.dataset_dir / relative),
                duration_s=round(audio.duration_s, 3),
                **fields,
            )
        )
        if self.verbose and len(self.clips) % 500 == 0:
            print(f"  ... {len(self.clips)} clips prepared")

    def write(self) -> Path:
        return write_manifest(self.dataset_dir, self.clips)
