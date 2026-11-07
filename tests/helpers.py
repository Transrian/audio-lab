"""Helpers shared by the tests. Everything here works offline and without datasets."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from audiolab.core.audio import SAMPLE_RATE
from audiolab.data.store import Clip


def write_tone(path, seconds: float = 1.0, rate: int = SAMPLE_RATE, frequency: int = 440):
    """Write a short readable tone where a test needs real audio on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(seconds * rate)) / rate
    samples = (0.3 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    sf.write(str(path), samples, rate, subtype="PCM_16")
    return path


def make_clip(clip_id: str, language: str = "fr", group: str = "", **fields) -> Clip:
    """A manifest clip with plausible defaults, so a test only states what it checks."""
    is_speech = language != "none"
    values = {
        "id": clip_id,
        "audio": f"audio/{clip_id.replace(':', '_')}.wav",
        "sha256": "0" * 64,
        "duration_s": 3.0,
        "language": language,
        "text": "bonjour tout le monde" if is_speech else "",
        "text_has_casing_punct": is_speech,
        "group": group or clip_id,
        "group_kind": "speaker" if is_speech else "clip",
        "source_split": "test",
        "license": "CC-BY-4.0",
        "category": "",
    }
    return Clip(**(values | fields))


def speech_clips(prefix: str, speakers: int = 40, per_speaker: int = 15) -> list[Clip]:
    """A fake speech dataset: several utterances per speaker."""
    return [
        make_clip(f"{prefix}:{speaker:03d}-{index:03d}", group=f"{prefix}:spk{speaker:03d}")
        for speaker in range(speakers)
        for index in range(per_speaker)
    ]
