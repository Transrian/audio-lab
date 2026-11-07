"""Dataset preparation, exercised on tiny local fixtures instead of the network.

Each fetcher reads everything from the cache directory (``download`` copies a
local path instead of fetching), so these tests run the real preparation code
on files that look like the upstream ones.
"""

import csv
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import write_tone  # noqa: E402
from test_data_source import make_archive  # noqa: E402

from audiolab.core.audio import pcm_sha256  # noqa: E402
from audiolab.data.datasets import DATASETS  # noqa: E402
from audiolab.data.download import DownloadError  # noqa: E402
from audiolab.data.esc50 import selected_clips  # noqa: E402
from audiolab.data.fetch import FETCHERS, fetch_dataset  # noqa: E402
from audiolab.data.fleurs import fetch_fleurs  # noqa: E402
from audiolab.data.librispeech import fetch_librispeech_en  # noqa: E402
from audiolab.data.mls import fetch_mls_fr  # noqa: E402
from audiolab.data.nonspeech import (  # noqa: E402
    ESC50_GROUPS,
    digital_silence,
    esc50_group,
    fetch_nonspeech,
    low_level_noise,
)
from audiolab.data.store import Clip, read_manifest, verify_clips, write_manifest  # noqa: E402


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Data and cache directories of the test, with the environment pointing at them."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    monkeypatch.setenv("AUDIOLAB_DATA", str(data))
    monkeypatch.setenv("AUDIOLAB_CACHE", str(cache))
    return data, cache


def fake_fleurs_cache(cache: Path, name: str, rows: int = 60, readers: int = 3) -> None:
    """A cache that looks like a downloaded FLEURS language: a TSV and a tar archive.

    The real table has no header line, and one sentence is read by several
    readers, so a sentence id repeats while a file name does not.
    """
    directory = cache / name
    directory.mkdir(parents=True, exist_ok=True)
    table = []
    members = {}
    for index in range(rows):
        sentence = 900000 + index // readers
        filename = f"clip_{index}"
        table.append(f"{sentence}\t{filename}.wav\tBonjour numéro {index}.\t16000\tFEMALE")
        members[f"audio/{filename}.wav"] = write_tone(directory / f"{filename}.wav", seconds=0.4)
    (directory / "test.tsv").write_text("\n".join(table) + "\n", encoding="utf-8")
    make_archive(directory / "test.tar.gz", members)


def fake_mls_cache(cache: Path, name: str = "mls-fr") -> None:
    """A cache that looks like a downloaded MLS test split."""
    directory = cache / name
    directory.mkdir(parents=True, exist_ok=True)
    audio_name = "1234_5678_000"
    tone = write_tone(directory / "1.flac", seconds=0.3)
    make_archive(directory / f"{audio_name}.tar.gz", {f"{audio_name}/1.flac": tone})
    # The real name list already carries the .tar.gz suffix.
    (directory / "audio_filenames.txt").write_text(f"{audio_name}.tar.gz\n", encoding="utf-8")
    (directory / "segments.txt").write_text("1 1234 0.0 0.3\n", encoding="utf-8")
    (directory / "transcripts.txt").write_text("1 le petit chat dort\n", encoding="utf-8")


def fake_librispeech_cache(cache: Path, name: str = "librispeech-en") -> None:
    """A cache that looks like a downloaded LibriSpeech test-clean archive."""
    directory = cache / name
    directory.mkdir(parents=True, exist_ok=True)
    flac = write_tone(directory / "121-121726-0000.flac", seconds=0.3)
    transcript = directory / "121-121726.trans.txt"
    transcript.write_text("121-121726-0000 THE CAT SAT\n121-121726-0001 ON THE MAT\n", "utf-8")
    make_archive(
        directory / "test-clean.tar.gz",
        {
            "test-clean/121/121726/121-121726.trans.txt": transcript,
            "test-clean/121/121726/121-121726-0000.flac": flac,
        },
    )


def fake_esc50_cache(cache: Path, classes: int = 50, clips_per_class: int = 4) -> None:
    """A cache that looks like the ESC-50 CSV and its audio directory."""
    directory = cache / "nonspeech"
    (directory / "audio").mkdir(parents=True, exist_ok=True)
    rows = [["filename", "fold", "target", "category", "esc10", "src_file", "take"]]
    for group_index, group in enumerate(sorted(all_esc50_classes())[:classes]):
        for clip_index in range(clips_per_class):
            name = f"{(group_index % 5) + 1}-{group_index * 1000 + clip_index}-A-{group_index}.wav"
            rows.append([name, "1", str(group_index), group, "False", "1", "A"])
            write_tone(directory / "audio" / name, seconds=0.2)
    with (directory / "esc50.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def all_esc50_classes() -> set[str]:
    return {name for names in ESC50_GROUPS.values() for name in names}


def test_fleurs_fetch_prepares_a_manifest_with_references(dirs):
    data, cache = dirs
    fake_fleurs_cache(cache, "fleurs-fr")
    assert fetch_fleurs("fleurs-fr", cache, data / "fleurs-fr", verbose=False) == 60

    clips = read_manifest(data / "fleurs-fr")
    by_id = {clip.id: clip for clip in clips}
    first = by_id["fleurs-fr:clip_0"]
    assert first.audio == "audio/fleurs-fr_clip_0.wav"
    assert first.text == "Bonjour numéro 0."
    assert first.text_has_casing_punct is True
    # Three recordings of one sentence are one group and three distinct ids.
    assert first.group == "fleurs-fr:900000"
    assert by_id["fleurs-fr:clip_2"].group == "fleurs-fr:900000"
    assert len(by_id) == len(clips)
    assert clips[0].group_kind == "sentence"
    assert clips[0].language == "fr"
    assert clips[0].license == "CC-BY-4.0"
    assert len(clips[0].sha256) == 64
    assert verify_clips(data / "fleurs-fr", clips) == []


def test_fleurs_fetch_refuses_an_archive_that_does_not_match(dirs):
    data, cache = dirs
    fake_fleurs_cache(cache, "fleurs-fr", rows=3)
    with pytest.raises(DownloadError, match="do not match"):
        fetch_fleurs("fleurs-fr", cache, data / "fleurs-fr", verbose=False)


def test_mls_fetch_groups_by_speaker(dirs):
    data, cache = dirs
    fake_mls_cache(cache)
    assert fetch_mls_fr("mls-fr", cache, data / "mls-fr", verbose=False) == 1

    clip = read_manifest(data / "mls-fr")[0]
    assert clip.id == "mls-fr:1"
    assert clip.text == "le petit chat dort"
    assert clip.text_has_casing_punct is False
    assert clip.group == "mls-fr:1234"
    assert clip.group_kind == "speaker"


def test_mls_fetch_refuses_an_audio_file_without_a_segment(dirs):
    data, cache = dirs
    fake_mls_cache(cache)
    (cache / "mls-fr" / "segments.txt").write_text("2 1234 0.0 0.3\n", encoding="utf-8")
    with pytest.raises(DownloadError, match="not in segments.txt"):
        fetch_mls_fr("mls-fr", cache, data / "mls-fr", verbose=False)


def test_librispeech_fetch_reads_chapter_transcripts(dirs):
    data, cache = dirs
    fake_librispeech_cache(cache)
    count = fetch_librispeech_en("librispeech-en", cache, data / "librispeech-en", verbose=False)
    assert count == 1

    clip = read_manifest(data / "librispeech-en")[0]
    assert clip.id == "librispeech-en:121-121726-0000"
    assert clip.text == "THE CAT SAT"
    assert clip.group == "librispeech-en:121"
    assert clip.group_kind == "speaker"
    assert clip.text_has_casing_punct is False


def test_nonspeech_fetch_builds_the_battery(dirs):
    data, cache = dirs
    fake_esc50_cache(cache)
    assert fetch_nonspeech("nonspeech", cache, data / "nonspeech", verbose=False) == 300

    clips = read_manifest(data / "nonspeech")
    assert all(clip.language == "none" and clip.text == "" for clip in clips)
    assert len({clip.group for clip in clips}) == 300  # every clip is its own group
    assert sum(clip.id.startswith("nonspeech:silence") for clip in clips) == 50
    assert sum(clip.id.startswith("nonspeech:noise") for clip in clips) == 50
    # Every clip carries a category so the report can break the rate down:
    # the two generated kinds, then one group/class label per ESC-50 class.
    assert sum(clip.category == "silence" for clip in clips) == 50
    assert sum(clip.category == "noise" for clip in clips) == 50
    sounds = [clip.category for clip in clips if "/" in clip.category]
    assert len(sounds) == 200
    assert len({category.split("/")[1] for category in sounds}) == 50
    assert {category.split("/")[0] for category in sounds} <= set(ESC50_GROUPS)
    assert verify_clips(data / "nonspeech", clips) == []


def test_nonspeech_generation_is_deterministic():
    for index in range(3):
        assert (
            digital_silence("nonspeech", index).samples.tobytes()
            == digital_silence("nonspeech", index).samples.tobytes()
        )
        assert (
            low_level_noise("nonspeech", index).samples.tobytes()
            == low_level_noise("nonspeech", index).samples.tobytes()
        )


def test_nonspeech_levels_and_durations_stay_in_range():
    for index in range(50):
        silence = digital_silence("nonspeech", index)
        assert silence.samples.size and not silence.samples.any()
        assert 2.0 <= silence.duration_s <= 20.0

        noise = low_level_noise("nonspeech", index)
        level = 20 * math.log10(float(abs(noise.samples).max()))
        assert -60.5 <= level <= -29.5
        assert 2.0 <= noise.duration_s <= 20.0


def test_esc50_selection_is_spread_over_each_class(dirs):
    _, cache = dirs
    fake_esc50_cache(cache, clips_per_class=12)
    csv_path = cache / "nonspeech" / "esc50.csv"
    chosen = selected_clips(csv_path, per_class=4)
    assert len(chosen) == 200
    counts: dict[str, int] = {}
    for row in chosen:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert set(counts.values()) == {4}
    assert chosen == selected_clips(csv_path, per_class=4)  # stable


def test_esc50_selection_refuses_a_short_class_list(dirs):
    _, cache = dirs
    fake_esc50_cache(cache, classes=3)
    with pytest.raises(DownloadError, match="classes, expected 50"):
        selected_clips(cache / "nonspeech" / "esc50.csv")


def test_every_esc50_class_has_a_group():
    assert {esc50_group(name) for name in all_esc50_classes()} == set(ESC50_GROUPS)
    with pytest.raises(ValueError, match="not in ESC50_GROUPS"):
        esc50_group("a_new_class")


def test_every_dataset_has_a_fetcher():
    assert set(FETCHERS) == set(DATASETS)


def test_fetch_dataset_refuses_an_unknown_name(dirs):
    with pytest.raises(ValueError, match="unknown dataset"):
        fetch_dataset("nope")


def test_fetch_dataset_refuses_a_corrupted_preparation(dirs):
    data, _ = dirs
    write_manifest(data / "fleurs-fr", [make_manifest_clip("fleurs-fr:1", "audio/missing.wav")])
    with pytest.raises(RuntimeError, match="--force"):
        fetch_dataset("fleurs-fr")


def test_fetch_dataset_verifies_and_skips_a_prepared_dataset(dirs, capsys):
    data, _ = dirs
    tone = write_tone(data / "fleurs-fr" / "audio" / "clip.wav")
    write_manifest(
        data / "fleurs-fr",
        [make_manifest_clip("fleurs-fr:1", "audio/clip.wav", sha256=pcm_sha256(tone))],
    )
    fetch_dataset("fleurs-fr")
    assert "already prepared (1 clips), verified" in capsys.readouterr().out


def make_manifest_clip(clip_id: str, audio: str, sha256: str = "0" * 64) -> Clip:
    return Clip(
        id=clip_id,
        audio=audio,
        sha256=sha256,
        duration_s=1.0,
        language="fr",
        text="bonjour",
        text_has_casing_punct=True,
        group=clip_id,
        group_kind="sentence",
        source_split="test",
        license="CC-BY-4.0",
    )
