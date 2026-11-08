import hashlib

import numpy as np
import pytest
import soundfile as sf

from audiolab.core.audio import (
    SAMPLE_RATE,
    Audio,
    load_audio,
    pcm_sha256,
    to_int16,
    write_wav,
)
from audiolab.core.paths import cache_dir, data_dir, runs_dir


def _tone(seconds: float, rate: int, channels: int = 1) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    mono = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return np.stack([mono] * channels, axis=1) if channels > 1 else mono


def test_load_audio_resamples_to_16k(tmp_path):
    source = tmp_path / "48k.wav"
    sf.write(source, _tone(0.5, 48_000), 48_000, subtype="PCM_16")
    loaded = load_audio(source)
    assert loaded.sample_rate == SAMPLE_RATE
    assert loaded.samples.dtype == np.float32
    assert loaded.duration_s == pytest.approx(0.5, abs=1e-3)


def test_load_audio_averages_stereo_to_mono(tmp_path):
    source = tmp_path / "stereo.wav"
    stereo = np.zeros((16_000, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    stereo[:, 1] = -0.5
    sf.write(source, stereo, SAMPLE_RATE, subtype="PCM_16")
    loaded = load_audio(source)
    assert loaded.samples.ndim == 1
    assert np.abs(loaded.samples).max() < 1e-2  # the two channels cancel out


def test_write_wav_is_16k_mono_16bit(tmp_path):
    path = tmp_path / "out" / "clip.wav"
    write_wav(path, Audio(samples=_tone(0.1, SAMPLE_RATE), sample_rate=SAMPLE_RATE))
    info = sf.info(str(path))
    assert (info.samplerate, info.channels, info.subtype) == (SAMPLE_RATE, 1, "PCM_16")


def test_pcm_sha256_describes_the_written_file(tmp_path):
    path = tmp_path / "clip.wav"
    write_wav(path, Audio(samples=_tone(0.05, SAMPLE_RATE), sample_rate=SAMPLE_RATE))
    stored, rate = sf.read(str(path), dtype="int16")
    assert rate == SAMPLE_RATE
    assert pcm_sha256(path) == hashlib.sha256(stored.tobytes()).hexdigest()


def test_pcm_sha256_changes_with_the_audio(tmp_path):
    loud = tmp_path / "loud.wav"
    quiet = tmp_path / "quiet.wav"
    write_wav(loud, Audio(samples=_tone(0.05, SAMPLE_RATE), sample_rate=SAMPLE_RATE))
    write_wav(quiet, Audio(samples=0.5 * _tone(0.05, SAMPLE_RATE), sample_rate=SAMPLE_RATE))
    assert pcm_sha256(loud) != pcm_sha256(quiet)


def test_cut_takes_the_frames_between_two_times():
    audio = Audio(samples=_tone(1.0, 1000), sample_rate=1000)
    clip = audio.cut(0.25, 0.5)
    assert clip.sample_rate == 1000
    assert len(clip.samples) == 250
    assert np.array_equal(clip.samples, audio.samples[250:500])


def test_cut_clamps_a_time_that_runs_past_the_audio():
    audio = Audio(samples=_tone(1.0, 1000), sample_rate=1000)
    assert len(audio.cut(-0.5, 0.1).samples) == 100
    assert len(audio.cut(0.9, 3.0).samples) == 100
    # A time past the end of the audio gives no frames rather than an error: the
    # length filter of a preparation drops such a clip.
    assert len(audio.cut(1.5, 2.0).samples) == 0


def test_to_int16_clips_out_of_range_samples():
    assert to_int16(np.array([2.0, -2.0], dtype=np.float32)).tolist() == [32767, -32767]


def test_paths_come_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AUDIOLAB_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("AUDIOLAB_RUNS", str(tmp_path / "r"))
    monkeypatch.setenv("AUDIOLAB_CACHE", str(tmp_path / "c"))
    assert data_dir() == tmp_path / "d"
    assert runs_dir() == tmp_path / "r"
    assert cache_dir() == tmp_path / "c"
