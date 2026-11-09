"""Audio loading and writing, reduced to one format.

Every dataset is stored as 16 kHz mono 16-bit WAV, and every model receives
that format. Resampling once, at preparation time, keeps a run reproducible
(soxr is deterministic) and keeps the evaluation independent of the sample
rate a dataset happens to ship with.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import soxr

SAMPLE_RATE = 16_000
PCM_SUBTYPE = "PCM_16"


@dataclass(frozen=True)
class Audio:
    """Mono float32 samples in [-1, 1] and the rate they were sampled at."""

    samples: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / self.sample_rate

    def cut(self, start_s: float, end_s: float) -> Audio:
        """The part of this audio between two times, in frames.

        A cut that runs past either end is clamped to the audio, so an
        annotation written in seconds always gives a clip of the file it came
        from. A start at or after the end gives an empty clip, which the length
        filter of a preparation then drops.
        """
        first = max(round(start_s * self.sample_rate), 0)
        last = min(round(end_s * self.sample_rate), len(self.samples))
        return Audio(samples=self.samples[first:last], sample_rate=self.sample_rate)


def load_audio(source, sample_rate: int = SAMPLE_RATE) -> Audio:
    """Read any soundfile-supported file as mono float32 at ``sample_rate``.

    ``source`` is a path or a file-like object (an archive member, an HTTP
    response body), so a caller never has to unpack a dataset to disk.
    """
    samples, source_rate = sf.read(source, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if source_rate != sample_rate:
        mono = soxr.resample(mono, source_rate, sample_rate, quality="HQ")
    return Audio(samples=mono.astype(np.float32, copy=False), sample_rate=sample_rate)


def to_int16(samples: np.ndarray) -> np.ndarray:
    """The 16-bit samples that :func:`write_wav` stores, clipped to the representable range."""
    scaled = np.rint(np.clip(samples, -1.0, 1.0) * 32767.0)
    return scaled.astype("<i2")


def to_mono_int16(samples: np.ndarray) -> np.ndarray:
    """Average the channels of an int16 array into one channel, in integer arithmetic.

    Averaging the stored integers (not floats) keeps the result exact and
    therefore hashable: the same file always gives the same digest.
    """
    if samples.ndim == 1 or samples.shape[1] == 1:
        return samples.reshape(-1).astype("<i2")
    total = samples.astype("<i4").sum(axis=1)
    return (total // samples.shape[1]).astype("<i2")


def pcm_sha256(path, sample_rate: int = SAMPLE_RATE) -> str:
    """Hash of a clip as it is stored.

    The hash must be the same on every machine and after a re-read, so it is
    taken over the stored 16-bit samples of the file, not over a float
    conversion of them (float rounding would give another digest).
    """
    samples, _ = sf.read(str(path), dtype="int16", always_2d=True)
    return hashlib.sha256(to_mono_int16(samples).tobytes()).hexdigest()


def write_wav(path, audio: Audio) -> None:
    """Write ``audio`` as a 16-bit WAV, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(path), to_int16(audio.samples), audio.sample_rate, format="WAV", subtype=PCM_SUBTYPE
    )
