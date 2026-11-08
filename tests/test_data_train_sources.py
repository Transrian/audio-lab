"""Preparing training sources, exercised on tiny local fixtures instead of the network.

A training source is prepared like an evaluation set — the same manifest, the
same 16 kHz mono WAV — so these tests check what each source does differently:
which rows it reads, how it names a clip and its group, what it leaves out, and
that it stops as soon as the cap is reached. Everything runs on files written by
the test, so no test needs the network or a prepared dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import (  # noqa: E402
    audio_cell,
    make_clip,  # noqa: E402
    write_parquet,
    write_tone,
)
from test_data_source import make_archive  # noqa: E402

from audiolab.data import african_french, granary, mls, voxpopuli_train  # noqa: E402
from audiolab.data.download import DownloadError  # noqa: E402
from audiolab.data.store import read_manifest, write_manifest  # noqa: E402


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """Data and cache directories of the test, with the environment pointing at them."""
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    monkeypatch.setenv("AUDIOLAB_DATA", str(data))
    monkeypatch.setenv("AUDIOLAB_CACHE", str(cache))
    return data, cache


def listing(published: dict[str, dict[str, str]]):
    """A stand-in for the Hub's file listing of one directory."""

    def fake(repo: str, revision: str, path: str, **options) -> dict[str, str]:
        assert revision == "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
        return published[path]

    return fake


def fake_voxpopuli_cache(
    cache: Path, rows: int = 3, accents: dict[str, str] | None = None, shards: int = 1
) -> None:
    """A cache that looks like a downloaded VoxPopuli French train shard.

    ``accents`` gives the accent of the ``en_accented`` evaluation subset; a
    speaker listed there recorded English as well and must not be trained on.
    The default names one French-accented speaker who has no row in the French
    shard, so nothing is excluded unless a test asks for it.
    """
    accents = accents if accents is not None else {"999": "en_fr"}
    directory = cache / "voxpopuli-fr-train"
    tone = write_tone(directory / "tone.wav", seconds=3.0)
    write_parquet(
        directory / f"train-00000-of-{shards:05d}.parquet",
        {
            "audio": [audio_cell(tone, f"rec_{index}.wav") for index in range(rows)],
            "raw_text": [f"Bonjour, numéro {index}." for index in range(rows)],
            "speaker_id": ["10", "10", "20"][:rows],
            "is_gold_transcript": [True] * rows,
        },
    )
    accented = cache / "voxpopuli-en-french-accent"
    write_parquet(
        accented / "test-00000-of-00001.parquet",
        {
            "speaker_id": list((accents or {}).keys()),
            "accent": list((accents or {}).values()),
        },
    )


def patch_listing(monkeypatch, shards: int = 1) -> None:
    monkeypatch.setattr(
        voxpopuli_train,
        "huggingface_sha256_map",
        listing(
            {
                "fr": {
                    f"train-{index:05d}-of-{shards:05d}.parquet": "a" * 64
                    for index in range(shards)
                },
                "en_accented": {"test-00000-of-00001.parquet": "b" * 64},
            }
        ),
    )


def test_voxpopuli_fr_train_keeps_the_cased_text_and_groups_by_speaker(dirs, monkeypatch):
    data, cache = dirs
    fake_voxpopuli_cache(cache)
    patch_listing(monkeypatch)

    count = voxpopuli_train.fetch_voxpopuli_fr_train(
        "voxpopuli-fr-train", cache, data / "voxpopuli-fr-train", verbose=False
    )
    assert count == 3

    clips = read_manifest(data / "voxpopuli-fr-train")
    first = clips[0]
    assert first.id == "voxpopuli-fr-train:rec_0"
    assert first.text == "Bonjour, numéro 0."
    # raw_text, not normalized_text: the corpus publishes both and only raw_text
    # has casing and punctuation, which the base model already writes.
    assert first.text_has_casing_punct is True
    assert first.group == "voxpopuli-fr-train:10"
    assert first.group_kind == "speaker"
    assert first.source_split == "train"
    assert first.language == "fr"
    assert first.license == "CC0-1.0 (data only)"
    assert {clip.group for clip in clips} == {
        "voxpopuli-fr-train:10",
        "voxpopuli-fr-train:20",
    }


def test_voxpopuli_fr_train_leaves_out_a_speaker_the_accented_set_scores(dirs, monkeypatch):
    """The same speaker recorded English for the evaluation set (decision 0002)."""
    data, cache = dirs
    fake_voxpopuli_cache(cache, accents={"10": "en_fr", "999": "en_fr"})
    patch_listing(monkeypatch)

    count = voxpopuli_train.fetch_voxpopuli_fr_train(
        "voxpopuli-fr-train", cache, data / "voxpopuli-fr-train", verbose=False
    )
    assert count == 1
    assert [clip.id for clip in read_manifest(data / "voxpopuli-fr-train")] == [
        "voxpopuli-fr-train:rec_2"
    ]


def test_voxpopuli_fr_train_keeps_a_speaker_labelled_with_another_accent(dirs, monkeypatch):
    data, cache = dirs
    fake_voxpopuli_cache(cache, accents={"10": "en_de", "999": "en_fr"})
    patch_listing(monkeypatch)

    count = voxpopuli_train.fetch_voxpopuli_fr_train(
        "voxpopuli-fr-train", cache, data / "voxpopuli-fr-train", verbose=False
    )
    assert count == 3


def test_voxpopuli_fr_train_leaves_out_a_row_without_a_gold_transcript(dirs, monkeypatch):
    data, cache = dirs
    fake_voxpopuli_cache(cache)
    patch_listing(monkeypatch)
    tone = write_tone(cache / "voxpopuli-fr-train" / "other.wav", seconds=3.0)
    path = cache / "voxpopuli-fr-train" / "train-00000-of-00001.parquet"
    write_parquet(
        path,
        {
            "audio": [audio_cell(tone, "a.wav"), audio_cell(tone, "b.wav")],
            "raw_text": ["Bonjour.", "Bonjour."],
            "speaker_id": ["10", "10"],
            "is_gold_transcript": [False, True],
        },
    )
    count = voxpopuli_train.fetch_voxpopuli_fr_train(
        "voxpopuli-fr-train", cache, data / "voxpopuli-fr-train", verbose=False
    )
    assert count == 1
    assert [clip.id for clip in read_manifest(data / "voxpopuli-fr-train")] == [
        "voxpopuli-fr-train:b"
    ]


def test_voxpopuli_fr_train_stops_before_the_shard_it_does_not_need(dirs, monkeypatch):
    """Downloading only what the cap needs is the point of reading shard by shard.

    The second shard is not in the cache and cannot be downloaded here, so the
    test fails unless the preparation stops after the first one.
    """
    data, cache = dirs
    fake_voxpopuli_cache(cache, rows=1, shards=2)
    patch_listing(monkeypatch, shards=2)
    monkeypatch.setattr(voxpopuli_train, "TRAIN_CLIP_CAP", 1, raising=True)
    real_download = voxpopuli_train.download

    def refuse(url, destination, **options):
        if not Path(destination).is_file():
            raise AssertionError(f"{url} would have been downloaded after the cap was reached")
        return real_download(url, destination, **options)

    monkeypatch.setattr(voxpopuli_train, "download", refuse)
    count = voxpopuli_train.fetch_voxpopuli_fr_train(
        "voxpopuli-fr-train", cache, data / "voxpopuli-fr-train", verbose=False
    )
    assert count == 1


def test_voxpopuli_fr_train_refuses_a_directory_without_train_shards(dirs, monkeypatch):
    _, cache = dirs
    monkeypatch.setattr(
        voxpopuli_train,
        "huggingface_sha256_map",
        listing({"fr": {"test-00000-of-00001.parquet": "a" * 64}, "en_accented": {}}),
    )
    with pytest.raises(DownloadError, match="lists no 'train-' shard"):
        voxpopuli_train.fetch_voxpopuli_fr_train(
            "voxpopuli-fr-train", cache, cache / "out", verbose=False
        )


def test_voxpopuli_fr_train_refuses_a_shard_the_hub_lists_no_checksum_for(dirs, monkeypatch):
    """A file that cannot be checked is not a frozen source."""
    _, cache = dirs
    monkeypatch.setattr(
        voxpopuli_train,
        "huggingface_sha256_map",
        listing(
            {
                "fr": {"train-00000-of-00001.parquet": ""},
                "en_accented": {"test-00000-of-00001.parquet": "b" * 64},
            }
        ),
    )
    with pytest.raises(DownloadError, match="no checksum for 1 shards"):
        voxpopuli_train.fetch_voxpopuli_fr_train(
            "voxpopuli-fr-train", cache, cache / "out", verbose=False
        )


def fake_mls_train_cache(cache: Path, name: str = "mls-fr-train") -> None:
    """A cache that looks like a downloaded MLS French train split.

    One archive per speaker, as the corpus publishes it, and the same three text
    files as the test split; the utterance id carries the speaker.
    """
    directory = cache / name
    tone = write_tone(directory / "tone.flac", seconds=4.0)
    archives = {"1234_5678_000": "1234_5678_000001", "9999_1111_000": "9999_1111_000001"}
    for archive, utterance in archives.items():
        make_archive(directory / f"{archive}.tar.gz", {f"{archive}/{utterance}.flac": tone})
    (directory / "audio_filenames.txt").write_text(
        "\n".join(f"{archive}.tar.gz" for archive in archives) + "\n", encoding="utf-8"
    )
    (directory / "segments.txt").write_text(
        "\n".join(
            f"{utterance}\thttp://archive.org/download/book/book.mp3\t0.0\t4.0"
            for utterance in archives.values()
        )
        + "\n",
        encoding="utf-8",
    )
    (directory / "transcripts.txt").write_text(
        "\n".join(f"{utterance} le petit chat dort" for utterance in archives.values()) + "\n",
        encoding="utf-8",
    )


def test_mls_fr_train_reads_the_train_split_and_groups_by_speaker(dirs):
    data, cache = dirs
    fake_mls_train_cache(cache)

    count = mls.fetch_mls_fr_train("mls-fr-train", cache, data / "mls-fr-train", verbose=False)
    assert count == 2

    clips = read_manifest(data / "mls-fr-train")
    first = clips[0]
    assert first.id == "mls-fr-train:1234_5678_000001"
    assert first.text == "le petit chat dort"
    # The corpus publishes lowercase, unpunctuated transcripts, so the base
    # model's own formatting is not reinforced by this source.
    assert first.text_has_casing_punct is False
    assert {clip.group for clip in clips} == {"mls-fr-train:1234", "mls-fr-train:9999"}
    assert first.source_split == "train"
    assert first.license == "CC-BY-4.0"


def test_mls_fr_train_leaves_out_a_speaker_the_test_split_scores(dirs):
    """The splits keep their speakers apart; a scored speaker is never trained on."""
    data, cache = dirs
    fake_mls_train_cache(cache)
    write_manifest(
        data / "mls-fr",
        [make_clip("mls-fr:1234_5678_000001", group="mls-fr:1234")],
    )

    count = mls.fetch_mls_fr_train("mls-fr-train", cache, data / "mls-fr-train", verbose=False)
    assert count == 1
    assert [clip.group for clip in read_manifest(data / "mls-fr-train")] == ["mls-fr-train:9999"]


def test_mls_fr_train_stops_before_the_archive_it_does_not_need(dirs, monkeypatch):
    data, cache = dirs
    fake_mls_train_cache(cache)
    monkeypatch.setattr(mls, "TRAIN_CLIP_CAP", 1, raising=True)
    real_download = mls.download

    def refuse(url, destination, **options):
        if not Path(destination).is_file():
            raise AssertionError(f"{url} would have been downloaded after the cap was reached")
        return real_download(url, destination, **options)

    monkeypatch.setattr(mls, "download", refuse)
    count = mls.fetch_mls_fr_train("mls-fr-train", cache, data / "mls-fr-train", verbose=False)
    assert count == 1


def fake_granary_cache(cache: Path) -> None:
    """A cache that looks like the French directory of the Granary shards."""
    directory = cache / "granary-fr-train"
    tone = write_tone(directory / "tone.wav", seconds=2.0)
    write_parquet(
        directory / "00000000.parquet",
        {
            "audio": [audio_cell(tone, "fr000_00000000_video_0.wav")],
            "utt_id": ["fr000_00000000_video_0"],
            "text": ["Voilà, c'est simple."],
            "original_audio_id": ["video"],
        },
    )
    write_parquet(
        directory / "00000001.parquet",
        {
            "audio": [audio_cell(tone, "fr000_00000000_other_1.wav")],
            "utt_id": ["fr000_00000000_other_1"],
            "text": ["Une autre phrase."],
            "original_audio_id": ["video"],
        },
    )


def patch_granary_listing(monkeypatch, shards: int = 2) -> None:
    monkeypatch.setattr(
        granary,
        "huggingface_sha256_map",
        lambda repo, revision, path, **options: {
            f"{index:08d}.parquet": chr(ord("a") + index) * 64 for index in range(shards)
        },
    )


def test_granary_fr_train_keeps_the_label_text_and_groups_by_video(dirs, monkeypatch):
    data, cache = dirs
    fake_granary_cache(cache)
    patch_granary_listing(monkeypatch)

    count = granary.fetch_granary_fr_train(
        "granary-fr-train", cache, data / "granary-fr-train", verbose=False
    )
    assert count == 2

    clips = {clip.id: clip for clip in read_manifest(data / "granary-fr-train")}
    first = clips["granary-fr-train:fr000_00000000_video_0"]
    assert first.text == "Voilà, c'est simple."
    assert first.text_has_casing_punct is True
    # The corpus publishes no speaker identity, so the video is the group.
    assert {clip.group for clip in clips.values()} == {"granary-fr-train:video"}
    assert first.group_kind == "recording"
    assert first.source_split == "asr_only"
    assert first.license == "audio CC-BY-3.0, labels CC-BY-4.0"


def test_granary_fr_train_refuses_a_directory_without_shards(dirs, monkeypatch):
    _, cache = dirs
    monkeypatch.setattr(
        granary, "huggingface_sha256_map", lambda repo, revision, path, **options: {}
    )
    with pytest.raises(DownloadError, match="lists no shard"):
        granary.fetch_granary_fr_train("granary-fr-train", cache, cache / "out", verbose=False)


def fake_african_french_cache(
    cache: Path, name: str = "african-accented-fr-train", conditioned: str = ""
) -> None:
    """A cache that looks like SLR57, with its three ways of naming a recording.

    The corpus names its files after their date, so only the extension can be
    removed: a test that used a generic stem would pass while the preparation
    dropped every ``ca16`` recording.
    """
    directory = cache / name
    tone = write_tone(directory / "tone.wav", seconds=2.0)
    root = "African_Accented_French"
    members = {
        f"{root}/transcripts/train/ca16_read/conditioned.txt": _text_file(
            directory,
            "conditioned.txt",
            conditioned
            or "afc-gabon_16.06.15_048_read_0344 d' accord mais je ne connais pas\n",
        ),
        f"{root}/transcripts/train/ca16_conv/transcripts.txt": _text_file(
            directory,
            "transcripts.txt",
            "afc-gabon_16.06.11_002_conv_0023.tdf bonjour je voudrais voir\n",
        ),
        f"{root}/transcripts/train/yaounde/fn_text.txt": _text_file(
            directory,
            "fn_text.txt",
            "/mnt/corpora/Yaounde/read/wavs/16000/ctell1-01/ctell1-01-0002.wav"
            " quand il n' y en a plus\n",
        ),
        f"{root}/speech/train/ca16/113/afc-gabon_16.06.15_048_read_0344.wav": tone,
        f"{root}/speech/train/ca16/113/afc-gabon_16.06.11_002_conv_0023.wav": tone,
        f"{root}/speech/train/yaounde/read/ctell1-01/ctell1-01-0002.wav": tone,
        f"{root}/speech/train/yaounde/answers/ctell1-01/ctell1-01-0003.wav": tone,
    }
    make_archive(directory / "African_Accented_French.tar.gz", members)


def _text_file(directory: Path, file_name: str, content: str) -> Path:
    path = directory / "text" / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_african_french_removes_only_a_known_extension_from_a_recording_name():
    """``afc-gabon_16.06.15_048_read_0344`` has dots a generic stem would cut."""
    assert (
        african_french._key(
            "African_Accented_French/speech/train/ca16/113/afc-gabon_16.06.15_048_read_0344.wav"
        )
        == "afc-gabon_16.06.15_048_read_0344"
    )
    assert (
        african_french._key("afc-gabon_16.06.11_002_conv_0023.tdf")
        == "afc-gabon_16.06.11_002_conv_0023"
    )
    assert (
        african_french._key("/mnt/corpora/Yaounde/read/wavs/16000/ctell1-01/ctell1-01-0002.wav")
        == "ctell1-01-0002"
    )


def test_african_french_reads_the_three_transcript_forms_and_the_train_part(dirs):
    data, cache = dirs
    fake_african_french_cache(cache)

    count = african_french.fetch_african_accented_fr_train(
        "african-accented-fr-train", cache, data / "african-accented-fr-train", verbose=False
    )
    assert count == 3

    clips = {clip.id: clip for clip in read_manifest(data / "african-accented-fr-train")}
    assert set(clips) == {
        "african-accented-fr-train:afc-gabon_16.06.15_048_read_0344",
        "african-accented-fr-train:afc-gabon_16.06.11_002_conv_0023",
        "african-accented-fr-train:ctell1-01-0002",
    }
    ca16 = clips["african-accented-fr-train:afc-gabon_16.06.15_048_read_0344"]
    assert ca16.text == "d' accord mais je ne connais pas"
    assert ca16.group == "african-accented-fr-train:113"
    assert ca16.group_kind == "speaker"
    assert ca16.source_split == "train"
    assert ca16.license == "Apache-2.0"
    # The corpus publishes lowercase, unpunctuated text: a source that has
    # neither teaches the model to drop what it already writes.
    assert ca16.text_has_casing_punct is False
    # A recording published in the archive without a transcript is left out
    # rather than prepared with an empty reference.
    assert "african-accented-fr-train:ctell1-01-0003" not in clips


def test_african_french_stops_at_the_cap(dirs, monkeypatch):
    data, cache = dirs
    fake_african_french_cache(cache)
    monkeypatch.setattr(african_french, "TRAIN_CLIP_CAP", 1, raising=True)

    count = african_french.fetch_african_accented_fr_train(
        "african-accented-fr-train", cache, data / "african-accented-fr-train", verbose=False
    )
    assert count == 1


def test_african_french_refuses_a_transcript_line_without_text(dirs):
    """A line that cannot give a recording and its text stops the preparation."""
    data, cache = dirs
    fake_african_french_cache(cache, conditioned="afc-gabon_16.06.15_048_read_0344\n")
    with pytest.raises(DownloadError, match="unexpected line"):
        african_french.fetch_african_accented_fr_train(
            "african-accented-fr-train", cache, data / "african-accented-fr-train", verbose=False
        )
