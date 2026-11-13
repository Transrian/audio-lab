"""The noisy derived sets: deterministic mixing, even conditions, derived panels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from helpers import make_clip
from audiolab.core.audio import SAMPLE_RATE, Audio
from audiolab.core.audio import to_int16
from audiolab.data import noisy
from audiolab.data.download import DownloadError
from audiolab.data.panels import build_panels, check_panel, panel_path, read_panel

# --- the mixing itself ---------------------------------------------------------


def _speech(seconds: float = 2.0, amplitude: float = 0.5) -> np.ndarray:
    """A tone for the first half, silence for the rest: the loud part is active."""
    samples = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    loud = len(samples) // 2
    t = np.arange(loud) / SAMPLE_RATE
    samples[:loud] = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return samples


def _noise(seconds: float = 2.0) -> np.ndarray:
    return (np.random.default_rng(7).normal(0, 0.2, int(seconds * SAMPLE_RATE))).astype(
        np.float32
    )


def _measured_snr(speech: np.ndarray, noise: np.ndarray, mix: np.ndarray) -> float:
    speech_power = noisy._active_speech_power(speech)
    noise_power = float(np.mean((mix - speech).astype(np.float64) ** 2))
    return 10 * float(np.log10(speech_power / noise_power))


@pytest.mark.parametrize("target", (10, 5, 0))
def test_snr_of_a_mix_holds_on_the_active_speech(target):
    speech, noise = _speech(), _noise()

    mix = noisy._add_noise(speech, noise, target)

    assert abs(_measured_snr(speech, noise, mix) - target) < 0.5


def test_mixing_twice_writes_identical_audio():
    speech, noise, rir = _speech(), _noise(), _speech(seconds=1.0, amplitude=0.8)

    first_mix = noisy._add_noise(speech, noise, 5)
    second_mix = noisy._add_noise(speech, noise, 5)
    first_reverb = noisy._reverb(speech, rir)
    second_reverb = noisy._reverb(speech, rir)

    assert to_int16(first_mix).tobytes() == to_int16(second_mix).tobytes()
    assert to_int16(first_reverb).tobytes() == to_int16(second_reverb).tobytes()


def test_reverb_keeps_the_length_and_puts_the_direct_path_first():
    speech = _speech()
    # A RIR with a delayed peak: the cut must move the peak to time zero.
    rir = np.zeros(1600, dtype=np.float32)
    rir[400] = 1.0
    rir[600] = 0.3

    out = noisy._reverb(speech, rir)

    assert len(out) == len(speech)
    # With only the direct path left at t=0, the mix starts with the speech
    # scaled by the normalised peak, not with silence.
    assert np.abs(out[0]) > 0


def _fake_material(path):
    """The audio a fake material path stands for: noise for n, tones otherwise."""
    if "n.wav" in str(path):
        return Audio(samples=_noise(seconds=0.5), sample_rate=SAMPLE_RATE)
    return Audio(samples=_speech(seconds=0.5, amplitude=0.8), sample_rate=SAMPLE_RATE)


def test_peak_is_scaled_down_to_under_one():
    speech = _speech(amplitude=0.95)
    noise = _noise()
    clip = make_clip("src:one")

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(noisy, "load_audio", _fake_material)
        mix, _ = noisy._mix(
            "src:one",
            "noise-snr0",
            Audio(samples=speech, sample_rate=SAMPLE_RATE),
            [(Path("n.wav"), "public domain")],
            [Path("r.wav")],
            [([(Path("b.wav"), "src:g1")], "CC-BY-4.0")],
            clip,
        )

    assert np.abs(mix).max() <= 0.99


# --- the deterministic choices --------------------------------------------------


def test_conditions_spread_evenly_over_many_ids():
    counts = {condition: 0 for condition in noisy.CONDITIONS}
    for index in range(600):
        clip_id = f"fake-src:clip-{index:04d}"
        condition = noisy.CONDITIONS[noisy._h("noisy-condition", clip_id) % len(noisy.CONDITIONS)]
        counts[condition] += 1

    for condition, count in counts.items():
        assert 70 <= count <= 130, f"{condition}: {count} of 600"


def test_babble_uses_four_different_groups_and_alternates_pools():
    """Four speakers, four different groups; with two pools they alternate pools."""
    loaded: list[Path] = []
    fr_pool = [(Path(f"fr/g{k}/c{i}.wav"), f"fr:g{k}") for k in range(5) for i in range(2)]
    en_pool = [(Path(f"en/g{k}/c{i}.wav"), f"en:g{k}") for k in range(5) for i in range(2)]

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(noisy, "load_audio", lambda source, sample_rate=SAMPLE_RATE: (
            loaded.append(Path(source)) or Audio(samples=_speech(seconds=0.5), sample_rate=SAMPLE_RATE)
        ))
        mixed, license_ = noisy._babble_segment(
            "src:one", int(1.0 * SAMPLE_RATE), [(fr_pool, "CC-BY-4.0"), (en_pool, "CC-BY-4.0")]
        )

    assert len(loaded) == 4
    groups = [group for pool in (fr_pool, en_pool) for path, group in pool if path in loaded]
    assert len(set(groups)) == 4
    assert [str(p).split("/")[0] for p in loaded] == ["fr", "en", "fr", "en"]
    assert license_ == "CC-BY-4.0"
    assert len(mixed) == int(1.0 * SAMPLE_RATE)


def test_babble_stops_when_a_pool_has_too_few_groups():
    pool = [(Path(f"g{k}/c0.wav"), f"g{k}") for k in range(3)]
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(noisy, "load_audio", lambda source, sample_rate=SAMPLE_RATE: Audio(
            samples=_speech(seconds=0.5), sample_rate=SAMPLE_RATE
        ))

        with pytest.raises(DownloadError, match="fewer than four free groups"):
            noisy._babble_segment("src:one", int(0.5 * SAMPLE_RATE), [(pool, "CC-BY-4.0")])


# --- what a mixed clip carries --------------------------------------------------


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("noise-snr10", "CC-BY-4.0 (speech) + public domain (noise)"),
        ("babble-snr5", "CC-BY-4.0 (speech) + CC-BY-4.0 (babble)"),
        ("reverb", f"CC-BY-4.0 (speech) + {noisy.RIRS_LICENSE} (reverb)"),
        (
            "reverb-noise-snr10",
            f"CC-BY-4.0 (speech) + {noisy.RIRS_LICENSE} (reverb) + public domain (noise)",
        ),
    ],
)
def test_the_license_names_the_material(condition, expected):
    speech = _speech(seconds=0.5)
    clip = make_clip("src:one", license="CC-BY-4.0")
    babble_pool = [(Path(f"b{k}.wav"), f"src:g{k}") for k in range(4)]

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(noisy, "load_audio", _fake_material)
        _, license_ = noisy._mix(
            "src:one",
            condition,
            Audio(samples=speech, sample_rate=SAMPLE_RATE),
            [(Path("n.wav"), "public domain")],
            [Path("r.wav")],
            [(babble_pool, "CC-BY-4.0")],
            clip,
        )

    assert license_ == expected


# --- the derived panels ---------------------------------------------------------


def _source_clips(prefix: str) -> list:
    return [make_clip(f"{prefix}:g{g}c{i}", group=f"{prefix}:g{g}") for g in (1, 2) for i in range(3)]


def test_derived_panels_map_the_source_ids_and_keep_sealing(tmp_path):
    source = "fleurs-fake"
    derived = f"noisy-{source}"
    source_clips = _source_clips(source)
    derived_clips = [
        make_clip(
            f"{derived}:{clip.id.partition(':')[2]}",
            group=f"{derived}:{clip.group.partition(':')[2]}",
            sha256=clip.sha256,
        )
        for clip in source_clips
    ]
    build_panels(source, source_clips, panels_dir=tmp_path)

    panels = build_panels(derived, derived_clips, panels_dir=tmp_path)

    for panel in panels:
        source_panel = read_panel(panel_path(f"{source}-{panel.name}", tmp_path))
        assert panel.ids == sorted(
            f"{derived}:{clip_id.partition(':')[2]}" for clip_id in source_panel.ids
        )
        # The sealed group of the source is the sealed group of the derived set.
        source_sealed = set(read_panel(panel_path(f"{source}-sealed", tmp_path)).ids)
        derived_sealed = set(read_panel(panel_path(f"{derived}-sealed", tmp_path)).ids)
        sealed_source_groups = {clip.group for clip in source_clips if clip.id in source_sealed}
        sealed_derived_groups = {clip.group for clip in derived_clips if clip.id in derived_sealed}
        assert sealed_derived_groups == {
            f"{derived}:{group.partition(':')[2]}" for group in sealed_source_groups
        }
        # The derived panel checks clean against its own manifest.
        assert check_panel(panel_path(derived + f"-{panel.name}", tmp_path), derived_clips) == []


def test_derived_panels_refuse_a_missing_source_and_an_overwrite(tmp_path):
    source = "fleurs-fake"
    derived = f"noisy-{source}"
    derived_clips = [
        make_clip(
            f"{derived}:{clip.id.partition(':')[2]}",
            group=f"{derived}:{clip.group.partition(':')[2]}",
            sha256=clip.sha256,
        )
        for clip in _source_clips(source)
    ]
    # No source panels yet: the build is refused, naming the missing file.
    with pytest.raises(ValueError, match="fleurs-fake-fast.txt is missing"):
        build_panels(derived, derived_clips, panels_dir=tmp_path)
    # With the source panels: the build runs, and a second build is refused.
    build_panels(source, _source_clips(source), panels_dir=tmp_path)
    build_panels(derived, derived_clips, panels_dir=tmp_path)
    with pytest.raises(FileExistsError):
        build_panels(derived, derived_clips, panels_dir=tmp_path)
