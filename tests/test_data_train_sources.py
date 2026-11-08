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

from audiolab.data import mls, voxpopuli_train  # noqa: E402
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


def test_voxpopuli_fr_train_refuses_a_shard_the_hub_lists_no_checksum_for(
    dirs, monkeypatch
):
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
