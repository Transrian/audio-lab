"""The noise banks: the MUSAN split rule, its licence file, and the RIRS lists."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from helpers import write_tone

from audiolab.data import noise_bank
from audiolab.data.download import DownloadError
from audiolab.data.store import read_manifest


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Data and cache directories of the test, with the environment pointing at them."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    monkeypatch.setenv("AUDIOLAB_DATA", str(data))
    monkeypatch.setenv("AUDIOLAB_CACHE", str(cache))
    return data, cache


# --- the split rule -----------------------------------------------------------


def test_musan_split_rule_is_stable():
    """The halves come from a constant rule; named files pin both of them."""
    assert noise_bank.musan_split("noise-free-sound-0000.wav") == "eval"
    assert noise_bank.musan_split("noise-sound-bible-0000.wav") == "eval"
    assert noise_bank.musan_split("noise-free-sound-0001.wav") == "train"
    assert noise_bank.musan_split("noise-free-sound-0000.wav") == "eval"


SOUND_BIBLE_LICENCE = """\
noise-sound-bible-0000
Public Domain
==============================================================================
noise-sound-bible-0001
CC BY 3.0
http://freesound.org/sound/1/
==============================================================================
noise-sound-bible-0002
CC BY-NC 3.0
==============================================================================
"""


def _fake_musan_archive(directory: Path, *, with_licence: bool = True) -> Path:
    """A MUSAN-shaped archive: noise, music and speech files, and the licence."""
    names = [
        "musan/noise/free-sound/noise-free-sound-0000.wav",
        "musan/noise/free-sound/noise-free-sound-0001.wav",
        "musan/noise/sound-bible/noise-sound-bible-0000.wav",
        "musan/noise/sound-bible/noise-sound-bible-0001.wav",
        "musan/noise/sound-bible/noise-sound-bible-0002.wav",
        "musan/music/fma/music-fma-0001.wav",
        "musan/speech/speech-0001.wav",
    ]
    for name in names:
        write_tone(directory / name, seconds=0.3, frequency=200)
    if with_licence:
        (directory / "musan/noise/sound-bible/LICENSE").write_text(
            SOUND_BIBLE_LICENCE, encoding="utf-8"
        )
    members = sorted(
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    )
    archive = directory / "musan.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name in members:
            handle.add(str(directory / name), arcname=name)
    return archive


def test_sound_bible_licence_reads_a_block_per_block(tmp_path):
    """A block names its files and its licence; a file's licence is its block's."""
    archive = _fake_musan_archive(tmp_path / "fixture")

    licences = noise_bank._sound_bible_licenses(archive)

    assert licences == {
        "noise-sound-bible-0000.wav": "Public Domain",
        "noise-sound-bible-0001.wav": "CC BY 3.0",
        "noise-sound-bible-0002.wav": "CC BY-NC 3.0",
    }


SOUND_BIBLE_LICENCE_REAL = """\
noise-sound-bible-0046
Title: MP5 SMG 9mm
License: Attribution 3.0
Recorded by gunguru
http://soundbible.com/2091-MP5-SMG-9mm.html
========================================================================
noise-sound-bible-0045
Title: Annoying Speaker Pulsing
License: Public Domain
Recorded by Ethan Buck
http://soundbible.com/2085-Annoying-Speaker-Pulsing.html
========================================================================
"""


def test_sound_bible_licence_reads_the_real_licence_lines():
    """The archive writes 'License: <value>' inside each per-file block."""
    licences = noise_bank.parse_licence_blocks(SOUND_BIBLE_LICENCE_REAL, "noise-")
    assert licences == {
        "noise-sound-bible-0046.wav": "Attribution 3.0",
        "noise-sound-bible-0045.wav": "Public Domain",
    }


def test_musan_noise_preps_only_the_noise_part_and_drops_a_bad_block(dirs, monkeypatch):
    """Music and speech stay unread, and a sound-bible file outside the licences is dropped."""
    data, cache = dirs
    fixture = _fake_musan_archive(cache / "fixture")
    monkeypatch.setattr(noise_bank, "MUSAN_URL", str(fixture))
    monkeypatch.setattr(noise_bank, "MUSAN_SHA256", "")
    monkeypatch.setattr(noise_bank, "MUSAN_EXPECTED_CLIPS", 4, raising=True)

    count = noise_bank.fetch_musan_noise("musan-noise", cache, data / "musan-noise", verbose=False)

    clips = {clip.id: clip for clip in read_manifest(data / "musan-noise")}
    # The four files whose licence the archive states; the NC one is in the
    # archive, so it is the dropped one, not a file the archive lacks.
    assert count == 4
    assert set(clips) == {
        "musan-noise:noise-free-sound-0000",
        "musan-noise:noise-free-sound-0001",
        "musan-noise:noise-sound-bible-0000",
        "musan-noise:noise-sound-bible-0001",
    }
    free = clips["musan-noise:noise-free-sound-0000"]
    assert free.license == "public domain"
    assert free.category == "free-sound"
    assert free.text == ""
    assert free.language == "none"
    assert free.group == "musan-noise:noise-free-sound-0000"
    assert free.group_kind == "recording"
    bible = clips["musan-noise:noise-sound-bible-0001"]
    assert bible.license == "CC BY 3.0"
    assert bible.category == "sound-bible"
    # The halves follow the rule, and the music and speech parts of the archive
    # never become clips.
    assert clips["musan-noise:noise-free-sound-0000"].source_split == "eval"
    assert clips["musan-noise:noise-free-sound-0001"].source_split == "train"
    assert not any("music" in clip.id or "speech" in clip.id for clip in clips.values())


def test_musan_noise_stops_when_no_sound_bible_licence_is_readable(dirs, monkeypatch):
    """A sub-collection without its licence file is a changed source, not a guess."""
    data, cache = dirs
    fixture = _fake_musan_archive(cache / "fixture", with_licence=False)
    monkeypatch.setattr(noise_bank, "MUSAN_URL", str(fixture))
    monkeypatch.setattr(noise_bank, "MUSAN_SHA256", "")
    monkeypatch.setattr(noise_bank, "MUSAN_EXPECTED_CLIPS", 1, raising=True)

    with pytest.raises(DownloadError, match="no musan/noise/sound-bible/LICENSE"):
        noise_bank.fetch_musan_noise("musan-noise", cache, data / "musan-noise", verbose=False)


# --- RIRS ---------------------------------------------------------------------


def _fake_rirs_archive(directory: Path) -> Path:
    """A RIRS-shaped zip: real and simulated RIRs with their lists, and the parts
    that are never read: the isotropic noises and the point-source MUSAN copies."""
    real = directory / "RIRS_NOISES/real_rirs_isotropic_noises"
    real.mkdir(parents=True)
    # One real RIR is an 8-channel measurement, the other mono.
    t = np.zeros((160, 8), dtype=np.float32)
    t[:, 0] = 0.5 * np.sin(np.arange(160) / 3.0)
    sf.write(str(real / "RVB_a.wav"), t, 16000, subtype="PCM_16")
    write_tone(real / "RVB_b.wav", seconds=0.01, frequency=200)
    write_tone(real / "RVB_noise.wav", seconds=0.01, frequency=150)
    (real / "rir_list").write_text(
        "--rir-id 00001 --room-id RVB_room RIRS_NOISES/real_rirs_isotropic_noises/RVB_a.wav\n"
        "--rir-id 00002 --room-id RVB_room RIRS_NOISES/real_rirs_isotropic_noises/RVB_b.wav\n"
    )
    (real / "noise_list").write_text(
        "--noise-id 00001 --noise-type isotropic "
        "RIRS_NOISES/real_rirs_isotropic_noises/RVB_noise.wav\n"
    )
    for room in ("smallroom", "mediumroom", "largeroom"):
        sim = directory / f"RIRS_NOISES/simulated_rirs/{room}"
        sim.mkdir(parents=True)
        write_tone(sim / "Room001/Room001-00001.wav", seconds=0.01, frequency=200)
        (sim / "rir_list").write_text(
            f"--rir-id {room}-00001 --room-id Room001 "
            f"RIRS_NOISES/simulated_rirs/{room}/Room001/Room001-00001.wav\n"
        )
    point = directory / "RIRS_NOISES/pointsource_noises"
    point.mkdir(parents=True)
    write_tone(point / "noise-free-sound-0000.wav", seconds=0.01, frequency=150)
    (point / "noise_list").write_text(
        "--noise-id noise-free-sound-0000 "
        "RIRS_NOISES/pointsource_noises/noise-free-sound-0000.wav\n"
    )
    members = sorted(
        p.relative_to(directory).as_posix()
        for p in directory.rglob("*")
        if p.is_file() and "rirs_noises.zip" not in p.name
    )
    archive = directory / "rirs_noises.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name in members:
            handle.write(directory / name, name)
    return archive


def test_rirs_preps_the_rirs_of_the_lists_and_never_the_noises(dirs, monkeypatch):
    """The lists, not the names, name the RIRs; point-source copies stay unread."""
    data, cache = dirs
    fixture = _fake_rirs_archive(cache / "fixture")
    monkeypatch.setattr(noise_bank, "RIRS_URL", str(fixture))
    monkeypatch.setattr(noise_bank, "RIRS_SHA256", "", raising=True)
    monkeypatch.setattr(noise_bank, "RIRS_EXPECTED_CLIPS", 5, raising=True)

    count = noise_bank.fetch_rirs_noises("rirs-noises", cache, data / "rirs-noises", verbose=False)

    clips = {clip.id: clip for clip in read_manifest(data / "rirs-noises")}
    assert count == 5
    assert set(clips) == {
        "rirs-noises:RVB_a",
        "rirs-noises:RVB_b",
        "rirs-noises:simulated/smallroom/Room001/Room001-00001",
        "rirs-noises:simulated/mediumroom/Room001/Room001-00001",
        "rirs-noises:simulated/largeroom/Room001/Room001-00001",
    }
    # The same file name in three rooms is three clips; the noises of the archive
    # (isotropic and point-source) are no clips at all.
    stems = {clip.id.split(":", 1)[1] for clip in clips.values()}
    assert not any("noise" in stem for stem in stems)
    assert clips["rirs-noises:RVB_a"].source_split == "eval"
    assert clips["rirs-noises:RVB_a"].category == "real-rir"
    assert clips["rirs-noises:simulated/smallroom/Room001/Room001-00001"].source_split == "train"
    assert (
        clips["rirs-noises:simulated/smallroom/Room001/Room001-00001"].category == "simulated-rir"
    )
    # The 8-channel measurement is stored as channel 0, mono, 16 kHz.
    a, rate = sf.read(str(data / "rirs-noises" / clips["rirs-noises:RVB_a"].audio), always_2d=True)
    assert a.shape == (160, 1) and rate == 16000
    assert clips["rirs-noises:RVB_a"].license == "Apache-2.0"
