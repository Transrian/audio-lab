"""Dataset preparation, exercised on tiny local fixtures instead of the network.

Each fetcher reads everything from the cache directory (``download`` copies a
local path instead of fetching), so these tests run the real preparation code
on files that look like the upstream ones.
"""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import audio_cell, write_parquet, write_tone  # noqa: E402
from test_data_source import make_archive  # noqa: E402

from audiolab.core.audio import pcm_sha256  # noqa: E402
from audiolab.data.codeswitched import fetch_servicenow_codeswitched  # noqa: E402
from audiolab.data.csfleurs import fetch_cs_fleurs_fr_en  # noqa: E402
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
from audiolab.data.voxpopuli import fetch_voxpopuli_french_accent  # noqa: E402


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Data and cache directories of the test, with the environment pointing at them.

    The published sizes and the pinned sha256 of the real datasets are replaced
    by the fixture's own, so every test runs the real preparation code on small
    files (``expected_counts``); the pinning itself is tested separately.
    """
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    monkeypatch.setenv("AUDIOLAB_DATA", str(data))
    monkeypatch.setenv("AUDIOLAB_CACHE", str(cache))
    import audiolab.data.fleurs as fleurs
    import audiolab.data.librispeech as librispeech
    import audiolab.data.mls as mls
    import audiolab.data.nonspeech as nonspeech

    monkeypatch.setattr(fleurs, "EXPECTED_CLIPS", {"fleurs-fr": 60, "fleurs-en": 60}, raising=True)
    monkeypatch.setattr(fleurs, "PINNED_SHA256", {}, raising=True)
    monkeypatch.setattr(mls, "EXPECTED_CLIPS", 1, raising=True)
    monkeypatch.setattr(librispeech, "EXPECTED_CLIPS", 1, raising=True)
    monkeypatch.setattr(librispeech, "PINNED_SHA256", {}, raising=True)
    monkeypatch.setattr(nonspeech, "EXPECTED_CLIPS", 300, raising=True)
    import audiolab.data.codeswitched as codeswitched
    import audiolab.data.csfleurs as csfleurs
    import audiolab.data.esc50 as esc50
    import audiolab.data.voxpopuli as voxpopuli

    monkeypatch.setattr(voxpopuli, "EXPECTED_CLIPS", 2, raising=True)
    monkeypatch.setattr(voxpopuli, "PINNED_SHA256", {}, raising=True)
    monkeypatch.setattr(codeswitched, "EXPECTED_CLIPS", {"en_fr": 4, "en_fr_ca": 4}, raising=True)
    monkeypatch.setattr(codeswitched, "PINNED_SHA256", {}, raising=True)
    monkeypatch.setattr(
        csfleurs,
        "SUBSETS",
        (("read/test", "read", 2, ""), ("xtts/test1", "xtts", 2, "")),
        raising=True,
    )
    monkeypatch.setattr(esc50, "EXPECTED_CSV_SHA256", "", raising=True)
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


def fake_mls_cache(cache: Path, name: str = "mls-fr", urls: int = 1) -> None:
    """A cache that looks like a downloaded MLS test split.

    Utterance ids look like ``<speaker>_<book>_<number>``, the segments file
    names the source mp3 of each utterance, and one speaker can appear under
    several mp3 files: that is what the real split does.
    """
    directory = cache / name
    directory.mkdir(parents=True, exist_ok=True)
    audio_name = "1234_5678_000"
    tone = write_tone(directory / "1.flac", seconds=0.3)
    make_archive(
        directory / f"{audio_name}.tar.gz",
        {f"{audio_name}/1234_5678_{index:06d}.flac": tone for index in range(1, 4)},
    )
    # The real name list already carries the .tar.gz suffix.
    (directory / "audio_filenames.txt").write_text(f"{audio_name}.tar.gz\n", encoding="utf-8")
    segments = [
        f"1234_5678_{index:06d}\thttp://archive.org/download/book/book_{index % urls}.mp3\t0.0\t0.3"
        for index in range(1, 4)
    ]
    (directory / "segments.txt").write_text("\n".join(segments) + "\n", encoding="utf-8")
    (directory / "transcripts.txt").write_text(
        "\n".join(f"1234_5678_{index:06d} le petit chat dort" for index in range(1, 4)) + "\n",
        encoding="utf-8",
    )


def fake_librispeech_cache(
    cache: Path,
    name: str = "librispeech-en",
    transcript_text: str = "121-121726-0000 THE CAT SAT\n121-121726-0001 ON THE MAT\n",
) -> None:
    """A cache that looks like a downloaded LibriSpeech test-clean archive.

    The FLAC comes first in the archive, as it does upstream: a reader that
    needs the transcript before the audio has to look for it first.
    """
    directory = cache / name
    directory.mkdir(parents=True, exist_ok=True)
    flac = write_tone(directory / "121-121726-0000.flac", seconds=0.3)
    transcript = directory / "121-121726.trans.txt"
    transcript.write_text(transcript_text, encoding="utf-8")
    make_archive(
        directory / "test-clean.tar.gz",
        {
            "test-clean/121/121726/121-121726-0000.flac": flac,
            "test-clean/121/121726/121-121726.trans.txt": transcript,
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


def test_fleurs_fetch_refuses_a_split_that_is_not_the_published_size(dirs):
    """A short preparation must fail loudly, not give a number that looks usable."""
    data, cache = dirs
    fake_fleurs_cache(cache, "fleurs-fr", rows=3)
    with pytest.raises(DownloadError, match="prepared 3 clips but 60 were expected"):
        fetch_fleurs("fleurs-fr", cache, data / "fleurs-fr", verbose=False)


def test_fleurs_fetch_checks_the_pinned_sha256_of_its_files(dirs, monkeypatch):
    """Each downloaded file is asked for with the checksum recorded for it."""
    import audiolab.data.fleurs as fleurs

    data, cache = dirs
    fake_fleurs_cache(cache, "fleurs-fr")
    pinned = {"fleurs-fr": {"test.tsv": "0" * 64, "test.tar.gz": "1" * 64}}
    monkeypatch.setattr(fleurs, "PINNED_SHA256", pinned)
    asked: dict[str, str] = {}
    real_download = fleurs.download

    def spy(url, destination, **options):
        asked[destination.name] = options.get("sha256", "")
        return real_download(url, destination, **options)

    monkeypatch.setattr(fleurs, "download", spy)
    assert fetch_fleurs("fleurs-fr", cache, data / "fleurs-fr", verbose=False) == 60
    assert asked == {"test.tsv": "0" * 64, "test.tar.gz": "1" * 64}


def test_mls_groups_by_speaker_not_by_source_file(dirs, monkeypatch):
    """One reader is spread over several mp3 files: the group is the reader.

    Reading the speaker out of the segments file instead gave one group per
    source mp3 (95 of them on the real split) so the sealed panel shared
    readers with the full panel, which is the leak the grouping exists to stop.
    """
    data, cache = dirs
    fake_mls_cache(cache, urls=2)
    monkeypatch.setattr("audiolab.data.mls.EXPECTED_CLIPS", 3)
    assert fetch_mls_fr("mls-fr", cache, data / "mls-fr", verbose=False) == 3

    clips = read_manifest(data / "mls-fr")
    assert {clip.group for clip in clips} == {"mls-fr:1234"}
    assert len({clip.id for clip in clips}) == 3
    assert clips[0].id == "mls-fr:1234_5678_000001"
    assert clips[0].text == "le petit chat dort"
    assert clips[0].text_has_casing_punct is False
    assert clips[0].group_kind == "speaker"
    assert clips[0].source_split == "test"


def test_mls_fetch_refuses_an_audio_file_without_a_segment(dirs, monkeypatch):
    data, cache = dirs
    fake_mls_cache(cache)
    monkeypatch.setattr("audiolab.data.mls.EXPECTED_CLIPS", 3)
    (cache / "mls-fr" / "segments.txt").write_text(
        "1234_5678_000002\thttp://archive.org/x.mp3\t0.0\t0.3\n", encoding="utf-8"
    )
    with pytest.raises(DownloadError, match="not in segments.txt"):
        fetch_mls_fr("mls-fr", cache, data / "mls-fr", verbose=False)


def test_librispeech_reads_transcripts_before_the_audio(dirs):
    """A tar stream lists the FLAC before the chapter transcript; both are needed."""
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


def test_librispeech_refuses_audio_without_a_transcript(dirs):
    data, cache = dirs
    fake_librispeech_cache(cache, transcript_text="121-121726-0001 ON THE MAT\n")
    with pytest.raises(DownloadError, match="has no transcript"):
        fetch_librispeech_en("librispeech-en", cache, data / "librispeech-en", verbose=False)


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
    # The split names where a clip came from: generated here, or ESC-50.
    assert {clip.source_split for clip in clips if "/" not in clip.category} == {"generated"}
    assert {clip.source_split for clip in clips if "/" in clip.category} == {"esc50"}
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
        # The level is the RMS, which is what a level in dBFS means: the battery
        # describes "noise at -60 to -30 dBFS", and the peak of white noise is
        # three times its RMS, so a peak-based level would miss the range.
        samples = np.asarray(noise.samples, dtype=np.float64)
        level = 20 * math.log10(float(np.sqrt(np.mean(np.square(samples)))))
        assert -60.5 <= level <= -29.5
        assert float(np.abs(samples).max()) <= 1.0
        assert 2.0 <= noise.duration_s <= 20.0


# The generated clips are the only ones audiolab makes itself, so they are the
# only ones whose exact bytes can be pinned. These digests are the samples of
# clip 0 of each kind: they change when a level, a duration or a seed changes,
# which is what a re-prepared dataset must not do silently. The pin was taken
# from the clips prepared after the level fix (RMS, not peak), and the previous
# scaling produced ad047a28... for this same clip.
PINNED_GENERATED_SAMPLES = {
    ("noise", 0): "e59af511b8e20ef6edde6f0faaee3930e30b7621d77fb3251ecda821724973ba",
    ("silence", 0): "c4dde64d69280acc2d59fde16225be723d9b381d6ec472815efd1a2314c75f20",
}


def test_the_generated_clips_are_the_same_bytes_as_when_they_were_pinned():
    """The battery is generated here, so its exact bytes can be pinned.

    The clips were prepared once with the level applied to the peak instead of
    the RMS, and those files stayed on disk until the dataset was fetched again.
    A pinned digest of one clip per kind makes that drift impossible to miss.
    """
    for (kind, index), pinned in PINNED_GENERATED_SAMPLES.items():
        generator = digital_silence if kind == "silence" else low_level_noise
        samples = np.asarray(generator("nonspeech", index).samples, dtype=np.float32)
        assert hashlib.sha256(samples.tobytes()).hexdigest() == pinned, kind


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


def fake_voxpopuli_cache(cache: Path, name: str = "voxpopuli-en-french-accent") -> None:
    """A cache that looks like a downloaded VoxPopuli accent subset.

    The subset is one split in two Parquet shards holding the audio in a column
    and fifteen accents between them: only the French one is kept.
    """
    directory = cache / name
    tone = write_tone(directory / "tone.wav", seconds=0.3)
    rows = [
        ("PLENARY-1-en_1", "en_fr", "124746", "So this means the quite relevant change", True),
        ("PLENARY-2-en_1", "en_fr", "1472", "there has also been a remark on it", True),
        ("PLENARY-3-en_1", "en_de", "99999", "a German accented remark", True),
    ]
    for shard, part in zip(VOXPOPULI_SHARDS, (rows[:2], rows[2:]), strict=True):
        write_parquet(
            directory / shard,
            {
                "audio": [audio_cell(tone, f"{row[0]}.wav") for row in part],
                "accent": [row[1] for row in part],
                "speaker_id": [row[2] for row in part],
                "raw_text": [row[3] for row in part],
                "is_gold_transcript": [row[4] for row in part],
            },
        )


VOXPOPULI_SHARDS = ("test-00000-of-00002.parquet", "test-00001-of-00002.parquet")


def test_voxpopuli_keeps_only_the_french_accent(dirs):
    """Fifteen accents share the subset; the panel must hold only the French one."""
    data, cache = dirs
    fake_voxpopuli_cache(cache)
    assert (
        fetch_voxpopuli_french_accent(
            "voxpopuli-en-french-accent", cache, data / "voxpopuli-en-french-accent", verbose=False
        )
        == 2
    )

    clips = read_manifest(data / "voxpopuli-en-french-accent")
    first = {clip.id: clip for clip in clips}["voxpopuli-en-french-accent:PLENARY-1-en_1"]
    assert first.text == "So this means the quite relevant change"
    assert first.language == "en"
    # The speaker is the group: one accent is spoken by several people.
    assert first.group == "voxpopuli-en-french-accent:124746"
    assert first.group_kind == "speaker"
    # The corpus's transcripts carry no punctuation, so there is no formatted view.
    assert first.text_has_casing_punct is False
    assert first.license.startswith("CC0-1.0")
    assert first.source_split == "test"
    assert len({clip.group for clip in clips}) == 2
    assert verify_clips(data / "voxpopuli-en-french-accent", clips) == []


def test_voxpopuli_refuses_a_transcript_that_is_not_gold(dirs):
    """The point of this set is human-edited transcripts; the flag says which are."""
    data, cache = dirs
    fake_voxpopuli_cache(cache)
    shard = cache / "voxpopuli-en-french-accent" / VOXPOPULI_SHARDS[0]
    tone = cache / "voxpopuli-en-french-accent" / "tone.wav"
    write_parquet(
        shard,
        {
            "audio": [audio_cell(tone, "PLENARY-1-en_1.wav")],
            "accent": ["en_fr"],
            "speaker_id": ["124746"],
            "raw_text": ["not a gold transcript"],
            "is_gold_transcript": [False],
        },
    )
    with pytest.raises(DownloadError, match="not a gold transcript"):
        fetch_voxpopuli_french_accent(
            "voxpopuli-en-french-accent", cache, data / "voxpopuli-en-french-accent", verbose=False
        )


def test_voxpopuli_refuses_a_split_that_is_not_the_published_size(dirs, monkeypatch):
    data, cache = dirs
    fake_voxpopuli_cache(cache)
    monkeypatch.setattr("audiolab.data.voxpopuli.EXPECTED_CLIPS", 5)
    with pytest.raises(DownloadError, match="prepared 2 clips but 5 were expected"):
        fetch_voxpopuli_french_accent(
            "voxpopuli-en-french-accent", cache, data / "voxpopuli-en-french-accent", verbose=False
        )


def fake_codeswitched_cache(cache: Path, clips: int = 4) -> None:
    """A cache that looks like both downloaded ServiceNow configurations."""
    for name, config in CONFIGS.items():
        directory = cache / name
        tone = write_tone(directory / "tone.wav", seconds=0.3)
        write_parquet(
            directory / "test-00000-of-00001.parquet",
            {
                "ID": [f"{config}_codeswitched_{index:04d}" for index in range(clips)],
                "audio": [
                    audio_cell(tone, f"audio_response_id_{config}_codeswitched_{index:04d}.wav")
                    for index in range(clips)
                ],
                "utterance": [
                    f"Je viens de rebooter mon laptop après un update {index}."
                    for index in range(clips)
                ],
                "voice": [
                    "F-Marine" if index % 2 == 0 else "M-Denis Landrieu" for index in range(clips)
                ],
            },
        )


CONFIGS = {
    "servicenow-codeswitched-en-fr": "en_fr",
    "servicenow-codeswitched-en-fr-ca": "en_fr_ca",
}


def test_servicenow_is_prepared_as_two_french_framed_datasets(dirs):
    """Each configuration is its own dataset, so Québec French has its own numbers."""
    data, cache = dirs
    fake_codeswitched_cache(cache)
    for name in CONFIGS:
        assert fetch_servicenow_codeswitched(name, cache, data / name, verbose=False) == 4

    clips = read_manifest(data / "servicenow-codeswitched-en-fr-ca")
    first = {clip.id: clip for clip in clips}[
        "servicenow-codeswitched-en-fr-ca:en_fr_ca_codeswitched_0000"
    ]
    assert first.text == "Je viens de rebooter mon laptop après un update 0."
    # French frames every utterance of the corpus, in both configurations.
    assert first.language == "fr"
    assert first.text_has_casing_punct is True
    # Two synthetic voices are too few groups: the clip is the group, the voice
    # its category.
    assert first.group == "servicenow-codeswitched-en-fr-ca:en_fr_ca_codeswitched_0000"
    assert first.group_kind == "clip"
    assert first.category == "voice/F-Marine"
    assert len({clip.group for clip in clips}) == len(clips)
    assert "evaluation only" in first.license
    # The two configurations are two datasets: their clips never share an id.
    other = read_manifest(data / "servicenow-codeswitched-en-fr")
    assert {clip.id for clip in other} & {clip.id for clip in clips} == set()
    assert all(clip.id.startswith("servicenow-codeswitched-en-fr:") for clip in other)
    assert verify_clips(data / "servicenow-codeswitched-en-fr-ca", clips) == []


def test_servicenow_refuses_a_configuration_that_is_not_the_published_size(dirs, monkeypatch):
    data, cache = dirs
    fake_codeswitched_cache(cache, clips=3)
    with pytest.raises(DownloadError, match="prepared 3 clips but 4 were expected"):
        fetch_servicenow_codeswitched(
            "servicenow-codeswitched-en-fr",
            cache,
            data / "servicenow-codeswitched-en-fr",
            verbose=False,
        )


def fake_cs_fleurs_cache(cache: Path, name: str = "cs-fleurs-fr-en") -> None:
    """A cache that looks like the two downloaded CS-FLEURS subsets.

    Each subset ships one metadata file naming its clips and one WAV per clip;
    both subsets hold French-English rows next to pairs in other languages.
    """
    directory = cache / name
    tone = write_tone(directory / "tone.wav", seconds=0.3)
    subsets = {
        "read": [
            {
                "id": "fra_1660_MW",
                "file_name": "audio/fra/fra_1660_MW.wav",
                "text": "Je viens de rebooter mon laptop.",
                "language": "fra-eng",
            },
            {
                "id": "fra_1661_MW",
                "file_name": "audio/fra/fra_1661_MW.wav",
                "text": "Le ticket ID7824956 est fermé.",
                "language": "fra-eng",
            },
            {
                "id": "deu_1660_AB",
                "file_name": "audio/deu/deu_1660_AB.wav",
                "text": "Ich habe mein Laptop neu gestartet.",
                "language": "deu-eng",
            },
        ],
        "xtts": [
            {
                "id": "1660_fra-spk0_sample0",
                "file_name": "audio/0/1660_fra-spk0_sample0.resamp.wav",
                "text": "Je viens de rebooter mon laptop.",
                "language": "fra-eng",
            },
            {
                "id": "1661_fra-spk0_sample0",
                "file_name": "audio/0/1661_fra-spk0_sample0.resamp.wav",
                "text": "Le ticket ID7824956 est fermé.",
                "language": "fra-eng",
            },
        ],
    }
    for category, records in subsets.items():
        for record in records:
            audio = directory / category / PurePosixPath(record["file_name"]).name
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(tone.read_bytes())
        (directory / f"{category}-metadata.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )


def test_cs_fleurs_keeps_the_french_english_pairs_of_both_subsets(dirs):
    """Real audio and synthetic audio are both kept, told apart by category."""
    data, cache = dirs
    fake_cs_fleurs_cache(cache)
    assert (
        fetch_cs_fleurs_fr_en("cs-fleurs-fr-en", cache, data / "cs-fleurs-fr-en", verbose=False)
        == 4
    )

    clips = read_manifest(data / "cs-fleurs-fr-en")
    by_id = {clip.id: clip for clip in clips}
    read = by_id["cs-fleurs-fr-en:fra_1660_MW"]
    assert read.text == "Je viens de rebooter mon laptop."
    assert read.language == "fr"  # the corpus's own field is "fra-eng"
    assert read.category == "read"
    assert read.source_split == "read/test"
    assert read.text_has_casing_punct is True
    # Both subsets are the same FLEURS sentence read twice: one group each.
    assert read.group == "cs-fleurs-fr-en:1660"
    assert by_id["cs-fleurs-fr-en:1660_fra-spk0_sample0"].group == "cs-fleurs-fr-en:1660"
    assert by_id["cs-fleurs-fr-en:1660_fra-spk0_sample0"].category == "xtts"
    assert by_id["cs-fleurs-fr-en:1660_fra-spk0_sample0"].source_split == "xtts/test1"
    assert {clip.group_kind for clip in clips} == {"sentence"}
    assert len({clip.group for clip in clips}) == 2
    assert "decision 0009" in read.license
    assert verify_clips(data / "cs-fleurs-fr-en", clips) == []


def test_cs_fleurs_refuses_a_subset_that_is_not_the_published_size(dirs, monkeypatch):
    data, cache = dirs
    fake_cs_fleurs_cache(cache)
    monkeypatch.setattr(
        "audiolab.data.csfleurs.SUBSETS",
        (("read/test", "read", 7, ""), ("xtts/test1", "xtts", 2, "")),
    )
    with pytest.raises(DownloadError, match="prepared 2 clips but 7 were expected"):
        fetch_cs_fleurs_fr_en("cs-fleurs-fr-en", cache, data / "cs-fleurs-fr-en", verbose=False)


def test_cs_fleurs_refuses_an_audio_file_the_hub_publishes_no_checksum_for(tmp_path, monkeypatch):
    """A file that still has to be downloaded is checked against the Hub's listing."""
    import audiolab.data.csfleurs as csfleurs

    subset = csfleurs.Subset("read/test", "read", 1, "")
    wanted = [{"file_name": "audio/fra/fra_1660_MW.wav"}]
    monkeypatch.setattr(csfleurs, "huggingface_sha256_map", lambda *args, **kwargs: {})
    with pytest.raises(DownloadError, match="lists no checksum"):
        csfleurs._published_checksums("cs-fleurs-fr-en", subset, wanted, tmp_path)

    monkeypatch.setattr(
        csfleurs, "huggingface_sha256_map", lambda *args, **kwargs: {"fra_1660_MW.wav": "a" * 64}
    )
    assert csfleurs._published_checksums("cs-fleurs-fr-en", subset, wanted, tmp_path) == {
        "fra_1660_MW.wav": "a" * 64
    }


def test_cs_fleurs_refuses_an_id_that_does_not_name_its_sentence(dirs):
    import audiolab.data.csfleurs as csfleurs

    with pytest.raises(DownloadError, match="does not name its FLEURS sentence"):
        csfleurs._sentence("cs-fleurs-fr-en", "concat.17")
