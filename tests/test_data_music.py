"""nonspeech-music: the licence and vocal filters, the excerpt, the family."""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest
import soundfile as sf

from helpers import write_tone
from audiolab.data import music
from audiolab.data.panels import (
    NONSPEECH_FAST,
    NONSPEECH_SEALED,
    build_panels,
    check_panel,
    panel_path,
    select_panels,
)
from audiolab.data.store import read_manifest
from audiolab.core.audio import SAMPLE_RATE

FMA_LICENCE = """\
music-fma-0001
"Old Tune"
Public Domain
==============================================================================
music-fma-0002
"New Tune" (by Somebody)
CC BY 4.0
http://example.org/tune
==============================================================================
music-fma-0003
"Shared Tune" (by Otherbody)
CC BY-SA 4.0
==============================================================================
"""
FMA_ANNOTATIONS = """\
music-fma-0001 blues N Instrumental_Ensemble
music-fma-0002 pop Y Somebody
music-fma-0003 jazz N Otherbody
"""
JAMENDO_LICENCE = """\
music-jamendo-0001
Public Domain
==============================================================================
"""
JAMENDO_ANNOTATIONS = "music-jamendo-0001 electronic N\n"


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Data and cache directories of the test, with the environment pointing at them."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    monkeypatch.setenv("AUDIOLAB_DATA", str(data))
    monkeypatch.setenv("AUDIOLAB_CACHE", str(cache))
    return data, cache


# --- the filters ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("licence", "ok"),
    [
        ("Public Domain", True),
        ("CC BY 4.0", True),
        ("CC BY 3.0", True),
        ("CC BY 2.5", True),
        ("CC BY-SA 4.0", False),
        ("CC BY-NC 3.0", False),
        ("CC BY-ND 4.0", False),
        (None, False),
    ],
)
def test_music_licence_ok(licence, ok):
    assert music.music_licence_ok(licence) is ok


@pytest.mark.parametrize(
    ("licence", "ok"),
    [
        # The wordings the archive's own licence files actually use.
        ("Stormy Blues by Arne Huseby is licensed under a Attribution 3.0 International License.", True),
        ("Live At Blues Alley by U.S. Army Blues is licensed under a Public Domain Mark 1.0 License.", True),
        ("Mussorgsky's Pictures at an Exhibition by Skidmore College Orchestra is licensed under a Public Domain License.", True),
        ("wcfsymphony In Concert by Jason Weinberger & wcfsymphony is licensed under a Creative Commons Attribution License.", True),
        ("Audionautix: Acoustic by Jason Shaw is licensed under a Attribution 3.0 United States License.", True),
        ("Attribution 3.0 Unported (CC BY 3.0)", True),
        ("Paris, Lisboa by Jahzzar is licensed under a Attribution-ShareAlike 3.0 International License.", False),
        ("Selections from the November 2006 Concert by Advent Chamber Orchestra is licensed under a Attribution-Share Alike 3.0 United States License.", False),
        ("Singles by bueno.sair.es is licensed under a Attribution-ShareAlike 2.5 Portugal License.", False),
        ("Go On by Morgan Sadler is licensed under a Attribution-NoDerivatives 3.0 International License.", False),
        ("Attribution-NoDerfff.0 Unported (CC BY-ND 3.0)", False),
        ("CC BY-SA 3.0 DE", False),
    ],
)
def test_the_archives_long_licence_wordings(licence, ok):
    assert music.music_licence_ok(licence) is ok


RFM_LICENCE = """\
CC BY 3.0
http://incompetech.com/music/royalty-free/index.html?genre=Blues
==============================================================================
music-rfm-0058
"Hustle" (by Kevin MacLeod)
music-rfm-0144
"Whiskey on the Mississippi" (by Kevin MacLeod)
==============================================================================
"""


def test_a_licence_header_above_the_file_list_applies_to_the_list():
    """The rfm part writes its CC BY 3.0 once, above the files it licences."""
    out = music.parse_licence_blocks(RFM_LICENCE, "music-")
    assert out == {"music-rfm-0058.wav": "CC BY 3.0", "music-rfm-0144.wav": "CC BY 3.0"}


def test_annotations_parse_vocals_and_artist():
    out = music._annotations(FMA_ANNOTATIONS)
    assert out == {
        "music-fma-0001": (False, "Instrumental_Ensemble"),
        "music-fma-0002": (True, "Somebody"),
        "music-fma-0003": (False, "Otherbody"),
    }


def _fake_music_archive(directory: Path) -> Path:
    """Two sub-collections with one eligible file each, the rest filtered out."""
    names = {
        "musan/music/fma/LICENSE": FMA_LICENCE,
        "musan/music/fma/ANNOTATIONS": FMA_ANNOTATIONS,
        "musan/music/jamendo/LICENSE": JAMENDO_LICENCE,
        "musan/music/jamendo/ANNOTATIONS": JAMENDO_ANNOTATIONS,
    }
    for name, text in names.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    # The long file exercises the 10 s excerpt, the short one stays whole.
    write_tone(directory / "musan/music/fma/music-fma-0001.wav", seconds=15, frequency=180)
    write_tone(directory / "musan/music/fma/music-fma-0002.wav", seconds=5, frequency=180)
    write_tone(directory / "musan/music/fma/music-fma-0003.wav", seconds=5, frequency=180)
    write_tone(directory / "musan/music/jamendo/music-jamendo-0001.wav", seconds=5, frequency=120)
    members = sorted(
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    )
    archive = directory / "musan.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name in members:
            handle.add(str(directory / name), arcname=name)
    return archive


def test_music_fetch_keeps_only_licensed_instrumental_files(dirs, monkeypatch, capsys):
    """Sung files and share-alike files stay out; the report comes before the write."""
    data, cache = dirs
    fixture = _fake_music_archive(cache / "fixture")
    monkeypatch.setattr(music, "MUSAN_URL", str(fixture))
    monkeypatch.setattr(music, "MUSAN_SHA256", "")
    monkeypatch.setattr(music, "MUSIC_EXPECTED_CLIPS", 2, raising=True)

    count = music.fetch_nonspeech_music("nonspeech-music", cache, data / "nonspeech-music", verbose=True)

    clips = {clip.id: clip for clip in read_manifest(data / "nonspeech-music")}
    assert count == 2
    assert set(clips) == {"nonspeech-music:music-fma-0001", "nonspeech-music:music-jamendo-0001"}
    long_file = clips["nonspeech-music:music-fma-0001"]
    assert long_file.license == "Public Domain"
    assert long_file.group == "nonspeech-music:Instrumental_Ensemble"
    assert long_file.group_kind == "artist"
    assert long_file.source_split == "fma"
    assert long_file.category == "music"
    assert long_file.text == ""
    assert long_file.language == "none"
    short_file = clips["nonspeech-music:music-jamendo-0001"]
    # No artist in the annotation: the file is its own group.
    assert short_file.group == "nonspeech-music:music-jamendo-0001"
    assert short_file.group_kind == "recording"
    # The 15 s file is cut to 10 s, the 5 s file stays whole.
    a, _ = sf.read(str(data / "nonspeech-music" / long_file.audio), always_2d=True)
    b, _ = sf.read(str(data / "nonspeech-music" / short_file.audio), always_2d=True)
    assert a.shape[0] == 10 * SAMPLE_RATE
    assert b.shape[0] == 5 * SAMPLE_RATE
    # The per-sub-collection report is printed before anything is written.
    out = capsys.readouterr().out
    report_end = out.index("eligible files before")
    assert "fma: Public Domain 1" in out[:report_end]
    assert "jamendo: Public Domain 1" in out[:report_end]


def test_spread_spreads_over_sub_collections_and_caps():
    kept = [(f"m/{c}/f{i}", c, f"c{i}", "Public Domain") for c in "abc" for i in range(5)]

    selected = music._spread(kept, 8)

    assert [entry[1] for entry in selected] == ["a", "b", "c", "a", "b", "c", "a", "b"]
    assert len(music._spread(kept, 100)) == 15


def test_the_music_family_keeps_the_old_family_numbers():
    """The frozen nonspeech panels must still check: only music may grow."""
    assert NONSPEECH_SEALED["silence"] == 10
    assert NONSPEECH_SEALED["noise"] == 10
    assert NONSPEECH_SEALED["esc50"] == 0
    assert NONSPEECH_FAST["silence"] == 10
    assert NONSPEECH_FAST["noise"] == 10
    assert NONSPEECH_FAST["esc50"] == 40
    assert NONSPEECH_SEALED["music"] == 40
    assert NONSPEECH_FAST["music"] == 30


def test_music_panels_draw_the_family(tmp_path):
    from helpers import make_clip

    # 60 groups: the first forty hold two clips each, the rest one.
    clips = [make_clip(f"nonspeech-music:music-x{i:03d}", language="none", group=f"nonspeech-music:g{i % 60}") for i in range(100)]
    build_panels("nonspeech-music", clips, panels_dir=tmp_path)

    by_id = {clip.id: clip for clip in clips}
    ids_of = {}
    for name in "sealed full fast".split():
        path = panel_path(f"nonspeech-music-{name}", tmp_path)
        ids_of[name] = {l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")}
        assert check_panel(path, clips) == []
    # The sealed part holds whole artists back: every clip of a sealed artist
    # is sealed, and a sealed artist never appears in the full panel.
    sealed_artists = {by_id[i].group for i in ids_of["sealed"]}
    full_artists = {by_id[i].group for i in ids_of["full"]}
    assert not sealed_artists & full_artists
    assert len(ids_of["sealed"]) >= 40  # whole artists, so the budget may overshoot
    assert len(ids_of["full"]) == 100 - len(ids_of["sealed"])
    assert ids_of["fast"] <= ids_of["full"]
    assert len(ids_of["fast"]) == 30


def test_a_big_artist_is_sealed_whole(tmp_path):
    """When the first sealed artist is large, the whole artist is sealed, not a cut of it."""
    from helpers import make_clip

    clips = (
        [make_clip(f"nonspeech-music:music-big{i:03d}", language="none", group="nonspeech-music:BigArtist") for i in range(80)]
        + [make_clip(f"nonspeech-music:music-s{i:03d}", language="none", group=f"nonspeech-music:small{i}") for i in range(20)]
    )
    chosen = select_panels(clips, 1234)

    big = {f"nonspeech-music:music-big{i:03d}" for i in range(80)}
    # Whatever the seed, the big artist is either fully sealed or fully unsealed:
    in_sealed = big & set(chosen["sealed"])
    in_full = big & set(chosen["full"])
    assert not (in_sealed and in_full)
    assert in_sealed in (set(), big)
    assert not set(chosen["sealed"]) & set(chosen["full"])
