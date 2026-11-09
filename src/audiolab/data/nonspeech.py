"""The non-speech battery: what a model invents when there is nothing to hear.

300 clips: 50 digital silences and 50 low-level noises generated here (no
license, no download) plus 200 ESC-50 environmental sounds. Music is not
covered — a known gap, see `docs/stt/datasets.md`.
"""

from __future__ import annotations

import hashlib
from functools import partial
from pathlib import Path

import numpy as np

from audiolab.core.audio import SAMPLE_RATE, Audio, load_audio
from audiolab.data.build import ManifestBuilder, require_clips
from audiolab.data.download import download
from audiolab.data.esc50 import CSV_URL, EXPECTED_CSV_SHA256, fetch_esc50_clips, selected_clips
from audiolab.data.esc50 import LICENSE as ESC50_LICENSE

LICENSE = "none (generated here)"
SOURCE_SPLIT = "generated"
ESC50_SOURCE_SPLIT = "esc50"
SILENCE_CLIPS = 50
NOISE_CLIPS = 50
ESC50_CLIPS = 200
# The published size of the battery; tests patch this to a small number.
EXPECTED_CLIPS = SILENCE_CLIPS + NOISE_CLIPS + ESC50_CLIPS
MINIMUM_DURATION_S = 2.0
MAXIMUM_DURATION_S = 20.0
MINIMUM_LEVEL_DBFS = -60.0
MAXIMUM_LEVEL_DBFS = -30.0
ESC50_GROUPS = {
    "animals": (
        "dog",
        "rooster",
        "pig",
        "cow",
        "frog",
        "cat",
        "hen",
        "insects",
        "sheep",
        "crow",
    ),  # fmt: skip
    "human": (
        "crying_baby",
        "sneezing",
        "clapping",
        "breathing",
        "coughing",
        "footsteps",
        "laughing",
        "brushing_teeth",
        "snoring",
        "drinking_sipping",
    ),  # fmt: skip
    "nature": (
        "rain",
        "sea_waves",
        "crackling_fire",
        "crickets",
        "chirping_birds",
        "water_drops",
        "wind",
        "pouring_water",
        "toilet_flush",
        "thunderstorm",
    ),  # fmt: skip
    "objects": (
        "door_wood_knock",
        "mouse_click",
        "keyboard_typing",
        "door_wood_creaks",
        "can_opening",
        "washing_machine",
        "vacuum_cleaner",
        "clock_alarm",
        "clock_tick",
        "glass_breaking",
        "helicopter",
        "chainsaw",
        "siren",
        "car_horn",
        "engine",
        "train",
        "church_bells",
        "airplane",
        "fireworks",
        "hand_saw",
    ),  # fmt: skip
}


def fetch_nonspeech(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare the non-speech battery, returning the number of clips written."""
    cache = cache_dir / name
    builder = ManifestBuilder(dataset_dir, verbose=verbose)
    for index in range(SILENCE_CLIPS):
        clip_id = f"{name}:silence-{index:03d}"
        silence = digital_silence(name, index)
        builder.add(clip_id, silence, group=clip_id, category="silence", **generated_fields())
    for index in range(NOISE_CLIPS):
        clip_id = f"{name}:noise-{index:03d}"
        noise = low_level_noise(name, index)
        builder.add(clip_id, noise, group=clip_id, category="noise", **generated_fields())
    add_esc50_clips(builder, name, cache, verbose)
    require_clips(name, len(builder.clips), EXPECTED_CLIPS)
    builder.write()
    return len(builder.clips)


def generated_fields(license: str = LICENSE, split: str = SOURCE_SPLIT) -> dict:
    """Manifest fields shared by the battery clips: no speech, so no reference text.

    The split is the source of the clip: the silences and noises are generated
    here, while the ESC-50 clips come from that corpus, and a reader of the
    manifest should not see both as "generated".
    """
    return {
        "language": "none",
        "text": "",
        "text_has_casing_punct": False,
        "source_split": split,
        "license": license,
        "group_kind": "clip",
    }


def digital_silence(name: str, index: int, rate: int = SAMPLE_RATE) -> Audio:
    """Two to twenty seconds of exact silence, as if a recorder had been left running."""
    rng = np.random.default_rng(_clip_seed(name, "silence", index))
    seconds = rng.uniform(MINIMUM_DURATION_S, MAXIMUM_DURATION_S)
    return Audio(samples=np.zeros(int(seconds * rate), dtype=np.float32), sample_rate=rate)


def low_level_noise(name: str, index: int, rate: int = SAMPLE_RATE) -> Audio:
    """Two to twenty seconds of white or pink noise, at -60 to -30 dBFS RMS.

    The level is the RMS, which is what a level in dBFS normally means and what
    the battery is built on: scaling by the peak instead would put the noise far
    below the intended level (a peak of white noise is three times its RMS) and
    the clips would measure a quieter condition than the one described.
    """
    rng = np.random.default_rng(_clip_seed(name, "noise", index))
    seconds = rng.uniform(MINIMUM_DURATION_S, MAXIMUM_DURATION_S)
    white = rng.standard_normal(int(seconds * rate))
    noise = white if index % 2 == 0 else _pink(white)
    target_dbfs = rng.uniform(MINIMUM_LEVEL_DBFS, MAXIMUM_LEVEL_DBFS)
    rms = float(np.sqrt(np.mean(np.square(noise))))
    samples = noise * (10 ** (target_dbfs / 20) / rms)
    return Audio(samples=samples.astype(np.float32), sample_rate=rate)


def _clip_seed(name: str, kind: str, index: int) -> int:
    """A seed per clip, so regenerating one clip does not depend on the others."""
    return int(hashlib.sha256(f"{name}:{kind}:{index}".encode()).hexdigest()[:16], 16)


def _pink(white: np.ndarray) -> np.ndarray:
    """Pink noise by spectral shaping: a plausible fan, air conditioner or traffic bed."""
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(len(white), d=1.0)
    scaling = np.ones_like(frequencies)
    scaling[1:] = 1.0 / np.sqrt(frequencies[1:])
    return np.fft.irfft(spectrum * scaling, n=len(white))


def add_esc50_clips(builder: ManifestBuilder, name: str, cache: Path, verbose: bool) -> None:
    """Add the chosen ESC-50 clips, four per class, and record the class group."""
    csv_path = download(CSV_URL, cache / "esc50.csv", sha256=EXPECTED_CSV_SHA256)
    chosen = selected_clips(csv_path)
    if len(chosen) != ESC50_CLIPS:
        raise ValueError(f"expected {ESC50_CLIPS} ESC-50 clips, got {len(chosen)}")
    for category, path in fetch_esc50_clips(cache / "audio", chosen):
        clip_id = f"{name}:esc50-{path.stem}"
        builder.add(
            clip_id,
            partial(load_audio, path),
            group=clip_id,
            category=f"{esc50_group(category)}/{category}",
            **generated_fields(ESC50_LICENSE, split=ESC50_SOURCE_SPLIT),
        )
    if verbose:
        print(f"  ... {len(chosen)} ESC-50 clips prepared")


def esc50_group(category: str) -> str:
    """Which family a class belongs to, the unit the report breaks the rate down by.

    A class that is not listed here stops preparation: a new upstream class
    must not be silently dropped from the breakdown.
    """
    for group, categories in ESC50_GROUPS.items():
        if category in categories:
            return group
    raise ValueError(f"ESC-50 class {category!r} is not in ESC50_GROUPS")
