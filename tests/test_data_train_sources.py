"""Preparing training sources, exercised on tiny local fixtures instead of the network.

A training source is prepared like an evaluation set — the same manifest, the
same 16 kHz mono WAV — so these tests check what each source does differently:
which rows it reads, how it names a clip and its group, what it leaves out, and
that it stops as soon as the cap is reached. Everything runs on files written by
the test, so no test needs the network or a prepared dataset.
"""

from __future__ import annotations

import json
import sys
import zipfile
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

from audiolab.data import (  # noqa: E402
    african_french,
    ami,
    granary,
    librispeech,
    librispeech_pc,
    mls,
    pxcorpus,
    voxpopuli_train,
)
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
    cache: Path,
    rows: int = 3,
    accents: dict[str, str] | None = None,
    shards: int = 1,
    speakers: list[str] | None = None,
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
            "speaker_id": (speakers or ["10", "10", "20"])[:rows],
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


def fake_mls_train_cache(
    cache: Path, name: str = "mls-fr-train", extra_archive: str | None = None
) -> None:
    """A cache that looks like a downloaded MLS French train split.

    One archive per speaker, as the corpus publishes it, and the same three text
    files as the test split; the utterance id carries the speaker.
    """
    directory = cache / name
    tone = write_tone(directory / "tone.flac", seconds=4.0)
    archives = {"1234_5678_000": "1234_5678_000001", "9999_1111_000": "9999_1111_000001"}
    if extra_archive:
        # A second archive of a reader that already appeared: the same speaker,
        # another book, and the file list keeps it next to the first.
        archives = {
            "1234_5678_000": "1234_5678_000001",
            extra_archive: f"{extra_archive.split('_')[0]}_5678_001001",
            "9999_1111_000": "9999_1111_000001",
        }
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
            conditioned or "afc-gabon_16.06.15_048_read_0344 d' accord mais je ne connais pas\n",
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
            " quand il n' y en a plus  il y en a encore \n"
            "/mnt/corpora/Yaounde/read/wavs/16000/ctell1-01/ctell1-01-0003.wav"
            " etes-vous sûr que c' est lui / elle \n",
        ),
        f"{root}/speech/train/ca16/113/afc-gabon_16.06.15_048_read_0344.wav": tone,
        f"{root}/speech/train/ca16/113/afc-gabon_16.06.11_002_conv_0023.wav": tone,
        f"{root}/speech/train/yaounde/read/ctell1-01/ctell1-01-0002.wav": tone,
        f"{root}/speech/train/yaounde/read/ctell1-01/ctell1-01-0003.wav": tone,
        f"{root}/speech/train/yaounde/answers/ctell1-01/ctell1-01-0004.wav": tone,
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
    # The corpus writes a space after the apostrophe; the model learns it and
    # the scorer counts two words, so the training text is cleaned.
    assert ca16.text == "d'accord mais je ne connais pas"
    assert ca16.group == "african-accented-fr-train:113"
    assert ca16.group_kind == "speaker"
    assert ca16.source_split == "train"
    assert ca16.license == "Apache-2.0"
    # The corpus publishes lowercase, unpunctuated text: a source that has
    # neither teaches the model to drop what it already writes.
    assert ca16.text_has_casing_punct is False
    # A recording published in the archive without a transcript is left out
    # rather than prepared with an empty reference.
    assert "african-accented-fr-train:ctell1-01-0004" not in clips


def test_african_french_cleans_the_training_text(dirs):
    """Apostrophe spacing and the whitespace runs go; nothing else changes."""
    data, cache = dirs
    fake_african_french_cache(cache)

    african_french.fetch_african_accented_fr_train(
        "african-accented-fr-train", cache, data / "african-accented-fr-train", verbose=False
    )

    clips = {clip.id: clip for clip in read_manifest(data / "african-accented-fr-train")}
    yaounde = clips["african-accented-fr-train:ctell1-01-0002"]
    assert yaounde.text == "quand il n'y en a plus il y en a encore"
    assert "  " not in yaounde.text
    assert yaounde.text == yaounde.text.strip()


def test_african_french_leaves_out_the_prompting_lines(dirs):
    """``c' est lui / elle`` is the question read to the speaker, not an answer."""
    data, cache = dirs
    fake_african_french_cache(cache)

    count = african_french.fetch_african_accented_fr_train(
        "african-accented-fr-train", cache, data / "african-accented-fr-train", verbose=False
    )
    assert count == 3
    assert "african-accented-fr-train:ctell1-01-0003" not in {
        clip.id for clip in read_manifest(data / "african-accented-fr-train")
    }


def test_african_french_cleanup_is_deterministic():
    assert african_french.training_text("aujourd' hui c' est ça") == "aujourd'hui c'est ça"
    assert african_french.training_text("il y en a plus  il y en a encore ") == (
        "il y en a plus il y en a encore"
    )
    # Nothing else is touched: the text stays lowercase and unpunctuated.
    assert african_french.training_text("qu' il aille à paris") == "qu'il aille à paris"


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


def fake_librispeech_train_cache(cache: Path, name: str = "librispeech-en-train") -> None:
    """A cache that looks like LibriSpeech train-clean-100.

    ``speaker/chapter/file.flac`` next to a ``.trans.txt`` per chapter, as the
    corpus publishes it, with the uppercase, unpunctuated transcripts. The third
    utterance is the one a second source (LibriSpeech-PC) can swap the text of.
    """
    directory = cache / name
    tone = write_tone(directory / "tone.flac", seconds=4.0)
    utterances = {
        "103/1240/103-1240-0001": "CHAPTER ONE OF A BOOK",
        "108/1254/108-1254-0001": "ANOTHER CHAPTER",
        "108/1254/108-1254-0002": "A SECOND PARAGRAPH",
    }
    members = {}
    for path, text in utterances.items():
        utterance = path.rsplit("/", 1)[-1]
        members[f"LibriSpeech/train-clean-100/{path}.flac"] = tone
        members[f"LibriSpeech/train-clean-100/{path}.trans.txt"] = _text_file(
            directory, f"{utterance}.trans.txt", f"{utterance} {text}\n"
        )
    make_archive(directory / "train-clean-100.tar.gz", members)


def test_librispeech_train_reads_the_train_archive_and_groups_by_speaker(dirs, monkeypatch):
    data, cache = dirs
    fake_librispeech_train_cache(cache)
    monkeypatch.setattr(librispeech, "PINNED_SHA256", {}, raising=True)

    count = librispeech.fetch_librispeech_en_train(
        "librispeech-en-train", cache, data / "librispeech-en-train", verbose=False
    )
    assert count == 3

    clips = {clip.id: clip for clip in read_manifest(data / "librispeech-en-train")}
    first = clips["librispeech-en-train:103-1240-0001"]
    assert first.text == "CHAPTER ONE OF A BOOK"
    # LibriSpeech publishes uppercase, unpunctuated transcripts.
    assert first.text_has_casing_punct is False
    assert first.group == "librispeech-en-train:103"
    assert first.group_kind == "speaker"
    assert first.source_split == "train-clean-100"
    assert first.license == "CC-BY-4.0"


def test_librispeech_train_leaves_out_a_speaker_test_clean_scores(dirs, monkeypatch):
    """The splits keep their speakers apart; a scored speaker is never trained on."""
    data, cache = dirs
    fake_librispeech_train_cache(cache)
    monkeypatch.setattr(librispeech, "PINNED_SHA256", {}, raising=True)
    write_manifest(
        data / "librispeech-en",
        [make_clip("librispeech-en:103-1240-0001", group="librispeech-en:103")],
    )

    count = librispeech.fetch_librispeech_en_train(
        "librispeech-en-train", cache, data / "librispeech-en-train", verbose=False
    )
    assert count == 2
    assert {clip.group for clip in read_manifest(data / "librispeech-en-train")} == {
        "librispeech-en-train:108"
    }


def test_librispeech_train_stops_at_the_cap(dirs, monkeypatch):
    data, cache = dirs
    fake_librispeech_train_cache(cache)
    monkeypatch.setattr(librispeech, "PINNED_SHA256", {}, raising=True)
    monkeypatch.setattr(librispeech, "TRAIN_CLIP_CAP", 1, raising=True)

    count = librispeech.fetch_librispeech_en_train(
        "librispeech-en-train", cache, data / "librispeech-en-train", verbose=False
    )
    assert count == 1


def fake_librispeech_pc_cache(cache: Path, name: str = "librispeech-pc-en-train") -> None:
    """A cache holding the LibriSpeech audio and the LibriSpeech-PC manifests.

    The audio is the archive ``librispeech-en-train`` downloads, so a test that
    prepares both sources says what a shared download means: the PC fetcher
    writes the audio under that source's cache entry.
    """
    fake_librispeech_train_cache(cache)
    directory = cache / name
    manifests = directory / "manifests.tar.gz"
    texts = directory / "train-clean-100.json"
    texts.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        _manifest_row("103/1240/103-1240-0001", "Hello, world. It is a test."),
        # A chapter title in capitals: three all-caps words in a row.
        _manifest_row("108/1254/108-1254-0001", "THE LONG ROAD She walked home."),
        # Two in a row are not a heading.
        _manifest_row("108/1254/108-1254-0002", "The MR JONES report is ready."),
    ]
    texts.write_text("\n".join(rows) + "\n", encoding="utf-8")
    make_archive(manifests, {"train-clean-100.json": texts})


def _manifest_row(path: str, text: str) -> str:
    return json.dumps(
        {
            "duration": 4.0,
            "audio_filepath": f"train-clean-100/{path}.flac",
            "text": text,
            "text_raw": text.upper(),
        }
    )


def test_librispeech_pc_takes_the_cased_text_and_leaves_out_a_heading(dirs, monkeypatch):
    data, cache = dirs
    fake_librispeech_pc_cache(cache)
    monkeypatch.setattr(librispeech_pc, "PINNED_SHA256", {}, raising=True)
    monkeypatch.setattr(librispeech_pc, "PINNED_AUDIO_SHA256", "", raising=True)

    count = librispeech_pc.fetch_librispeech_pc_en_train(
        "librispeech-pc-en-train", cache, data / "librispeech-pc-en-train", verbose=False
    )
    # The third utterance of the archive (103-1240-0001, 108-1254-0001,
    # 108-1254-0002) is the one whose text is a heading.
    assert count == 2

    clips = {clip.id: clip for clip in read_manifest(data / "librispeech-pc-en-train")}
    assert set(clips) == {
        "librispeech-pc-en-train:103-1240-0001",
        "librispeech-pc-en-train:108-1254-0002",
    }
    first = clips["librispeech-pc-en-train:103-1240-0001"]
    assert first.text == "Hello, world. It is a test."
    assert first.text_has_casing_punct is True
    assert first.group == "librispeech-pc-en-train:103"
    assert first.group_kind == "speaker"
    assert first.source_split == "train-clean-100"
    assert first.license == "CC-BY-4.0"
    assert clips["librispeech-pc-en-train:108-1254-0002"].text == "The MR JONES report is ready."


def test_librispeech_pc_stops_at_the_cap(dirs, monkeypatch):
    data, cache = dirs
    fake_librispeech_pc_cache(cache)
    monkeypatch.setattr(librispeech_pc, "PINNED_SHA256", {}, raising=True)
    monkeypatch.setattr(librispeech_pc, "PINNED_AUDIO_SHA256", "", raising=True)
    monkeypatch.setattr(librispeech, "TRAIN_CLIP_CAP", 1, raising=True)

    count = librispeech_pc.fetch_librispeech_pc_en_train(
        "librispeech-pc-en-train", cache, data / "librispeech-pc-en-train", verbose=False
    )
    assert count == 1


def test_librispeech_pc_counts_the_longest_run_of_capitalised_words():
    assert librispeech_pc.title_run("Hello, world. It is a test.") == 0
    assert librispeech_pc.title_run("The MR JONES report is ready.") == 2
    assert librispeech_pc.title_run("THE LONG ROAD She walked home.") == 3
    assert librispeech_pc.title_run("THE LONG ROAD She went to AMI HQ TODAY.") == 3


def test_librispeech_pc_refuses_a_manifest_archive_without_its_split(dirs, monkeypatch):
    _, cache = dirs
    directory = cache / "librispeech-pc-en-train"
    empty = directory / "other.json"
    empty.parent.mkdir(parents=True, exist_ok=True)
    empty.write_text("{}\n", encoding="utf-8")
    make_archive(directory / "manifests.tar.gz", {"test-clean.json": empty})
    with pytest.raises(DownloadError, match="holds no train-clean-100.json"):
        librispeech_pc.read_texts("librispeech-pc-en-train", directory / "manifests.tar.gz")


def test_librispeech_pc_refuses_an_unreadable_manifest_line(dirs):
    _, cache = dirs
    directory = cache / "librispeech-pc-en-train"
    bad = directory / "train-clean-100.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text('{"audio_filepath": "train-clean-100/1/2/1-2-0001.flac"}\n', encoding="utf-8")
    make_archive(directory / "manifests.tar.gz", {"train-clean-100.json": bad})
    with pytest.raises(DownloadError, match="unreadable line"):
        librispeech_pc.read_texts("librispeech-pc-en-train", directory / "manifests.tar.gz")


def fake_pxcorpus_cache(
    cache: Path, name: str = "pxcorpus-fr-train", non_audio: bool = False
) -> None:
    """A cache that looks like the zip PxCorpus publishes.

    One recording per utterance, and the three parallel files the corpus lists
    them in: the path, the participant and the transcript.
    """
    directory = cache / name
    tone = write_tone(directory / "tone.wav", seconds=3.0)
    root = "PxCorpus/PxSLU"
    # Three recordings and three participants: the corpus's files are parallel.
    paths = "00MYcyVK0t/recording_0.wav\n02Qp6ICj9Q/recording_1.wav\n02Qp6ICj9Q/recording_2.wav\n"
    if non_audio:
        paths = paths.replace("recording_2.wav", "readme.md")
    members = {
        f"{root}/paths.txt": _text_file(directory, "paths.txt", paths),
        f"{root}/metadata.txt": _text_file(
            directory,
            "metadata.txt",
            "24,18-28,F,doctor\n48,60+,F,non-expert\n48,60+,F,non-expert\n",
        ),
        f"{root}/seq.in": _text_file(
            directory,
            "seq.in",
            "flagyl 500 milligrammes euh qu/ en cachet pendant 8 jours\n"
            "3 comprimés par jour matin midi et soir pendant 10 jours\n"
            "and this line has no audio\n",
        ),
        f"{root}/recordings/00MYcyVK0t/recording_0.wav": tone,
        f"{root}/recordings/02Qp6ICj9Q/recording_1.wav": tone,
        f"{root}/recordings/02Qp6ICj9Q/recording_2.wav": tone,
        f"{root}/readme.md": _text_file(directory, "readme.md", "PxCorpus\n"),
    }
    make_zip(directory / "PxCorpus.zip", members)


def make_zip(path: Path, members: dict[str, Path]) -> Path:
    """A zip holding the given files, built the way PxCorpus publishes its own."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as handle:
        for name, source in members.items():
            handle.write(str(source), arcname=name)
    return path


def test_pxcorpus_takes_every_recording_and_groups_by_participant(dirs):
    data, cache = dirs
    fake_pxcorpus_cache(cache)

    count = pxcorpus.fetch_pxcorpus_fr_train(
        "pxcorpus-fr-train", cache, data / "pxcorpus-fr-train", verbose=False
    )
    assert count == 3

    clips = {clip.id: clip for clip in read_manifest(data / "pxcorpus-fr-train")}
    assert set(clips) == {
        "pxcorpus-fr-train:00MYcyVK0t/recording_0",
        "pxcorpus-fr-train:02Qp6ICj9Q/recording_1",
        "pxcorpus-fr-train:02Qp6ICj9Q/recording_2",
    }
    first = clips["pxcorpus-fr-train:00MYcyVK0t/recording_0"]
    # The transcript is kept as the corpus published it, cut-off word included.
    assert first.text == "flagyl 500 milligrammes euh qu/ en cachet pendant 8 jours"
    assert first.text_has_casing_punct is False
    assert first.group == "pxcorpus-fr-train:24"
    assert first.group_kind == "speaker"
    assert first.source_split == "recordings"
    assert first.license == "CC-BY-4.0"


def test_pxcorpus_refuses_parallel_files_that_do_not_line_up(dirs):
    """The three files describe the same recordings, so a short one is a defect."""
    _, cache = dirs
    directory = cache / "pxcorpus-fr-train"
    tone = write_tone(directory / "tone.wav", seconds=3.0)
    root = "PxCorpus/PxSLU"
    make_zip(
        directory / "PxCorpus.zip",
        {
            f"{root}/paths.txt": _text_file(directory, "paths.txt", "s/recording_0.wav\n"),
            f"{root}/metadata.txt": _text_file(directory, "metadata.txt", "24,18-28,F,doctor\n"),
            f"{root}/seq.in": _text_file(directory, "seq.in", "une phrase\nune autre phrase\n"),
            f"{root}/recordings/s/recording_0.wav": tone,
        },
    )
    with pytest.raises(DownloadError, match="the corpus's own files disagree"):
        pxcorpus.read_rows("pxcorpus-fr-train", directory / "PxCorpus.zip")


def test_pxcorpus_refuses_a_paths_file_naming_something_that_is_not_audio(dirs):
    data, cache = dirs
    fake_pxcorpus_cache(cache, non_audio=True)
    with pytest.raises(DownloadError, match="names a file that is not audio"):
        pxcorpus.fetch_pxcorpus_fr_train(
            "pxcorpus-fr-train", cache, data / "pxcorpus-fr-train", verbose=False
        )


def test_pxcorpus_refuses_an_archive_without_its_files(dirs):
    _, cache = dirs
    directory = cache / "pxcorpus-fr-train"
    make_zip(directory / "PxCorpus.zip", {})
    with pytest.raises(DownloadError, match="holds no PxCorpus/PxSLU/paths.txt"):
        pxcorpus.read_rows("pxcorpus-fr-train", directory / "PxCorpus.zip")


def test_voxpopuli_fr_train_leaves_out_a_row_whose_reader_is_not_named(dirs, monkeypatch):
    """A row the corpus does not name a reader for cannot be grouped or checked.

    VoxPopuli writes the string "None" in ``speaker_id`` for 12 % of its French
    train rows; such a clip can be neither resampled by reader nor checked
    against the evaluation sets, so the preparation passes over it.
    """
    data, cache = dirs
    fake_voxpopuli_cache(cache, rows=3, speakers=["10", "None", "20"])
    patch_listing(monkeypatch)

    count = voxpopuli_train.fetch_voxpopuli_fr_train(
        "voxpopuli-fr-train", cache, data / "voxpopuli-fr-train", verbose=False
    )
    assert count == 2
    assert {clip.group for clip in read_manifest(data / "voxpopuli-fr-train")} == {
        "voxpopuli-fr-train:10",
        "voxpopuli-fr-train:20",
    }


def test_mls_fr_train_skips_the_archives_of_a_reader_it_already_has(dirs, monkeypatch):
    """The file list is ordered by book, so one reader can own many archives.

    Reading them all would spend the whole cap on the first readers (19 speakers
    for 12 000 clips); a reader that already holds its share is not downloaded.
    """
    data, cache = dirs
    fake_mls_train_cache(cache, extra_archive="1234_5678_001")
    monkeypatch.setattr(mls, "CLIP_CAP_PER_GROUP", 1, raising=True)
    real_download = mls.download

    def refuse(url, destination, **options):
        if not Path(destination).is_file():
            raise AssertionError(f"{url} would have been downloaded for a full reader")
        return real_download(url, destination, **options)

    monkeypatch.setattr(mls, "download", refuse)
    count = mls.fetch_mls_fr_train("mls-fr-train", cache, data / "mls-fr-train", verbose=False)

    assert count == 2
    assert {clip.group for clip in read_manifest(data / "mls-fr-train")} == {
        "mls-fr-train:1234",
        "mls-fr-train:9999",
    }


AMI_PARTICIPANTS = """<?xml version="1.0" encoding="ISO-8859-1"?>
<root xmlns:nite="http://nite.sourceforge.net/">
   <participant nite:id="FIO017" nxt_agent="A" native_language="French"/>
   <participant nite:id="MIO023" nxt_agent="D" native_language="French"/>
   <participant nite:id="MIO042" nxt_agent="B" native_language="English"/>
</root>
"""

AMI_MEETINGS = """<?xml version="1.0" encoding="ISO-8859-1"?>
<root xmlns:nite="http://nite.sourceforge.net/">
  <meeting observation="IS1003a">
    <speaker nite:id="IS1003a.A" nxt_agent="A" global_name="FIO017" role="PM"/>
    <speaker nite:id="IS1003a.B" nxt_agent="B" global_name="MIO042" role="ME"/>
    <speaker nite:id="IS1003a.D" nxt_agent="D" global_name="MIO023" role="ID"/>
  </meeting>
  <meeting observation="ES2001a">
    <speaker nite:id="ES2001a.B" nxt_agent="B" global_name="MIO042" role="PM"/>
  </meeting>
</root>
"""

# One speaker's manual words: two sentences, a laugh and a disfluency marker in
# between, and a fragment that is too short to be a clip.
AMI_WORDS = """<?xml version="1.0" encoding="ISO-8859-1"?>
<root xmlns:nite="http://nite.sourceforge.net/">
  <w nite:id="w1" starttime="0.100" endtime="0.500">Okay</w>
  <w nite:id="w2" starttime="0.500" endtime="1.400">so</w>
  <w nite:id="w3" starttime="1.400" endtime="2.000" punc="true">,</w>
  <vocalsound nite:id="v1" starttime="2.000" endtime="2.500" type="laugh"/>
  <w nite:id="w4" starttime="2.600" endtime="3.000">I</w>
  <w nite:id="w5" starttime="3.000" endtime="3.500">think</w>
  <w nite:id="w6" starttime="3.500" endtime="4.200" punc="true">.</w>
  <disfmarker nite:id="d1" starttime="4.200" endtime="4.500" type="disfluency"/>
  <w nite:id="w7" starttime="6.000" endtime="6.800">Right</w>
  <w nite:id="w8" starttime="6.800" endtime="7.100" punc="true">?</w>
  <w nite:id="w9" starttime="8.000" endtime="8.200">tiny</w>
</root>
"""


def _ami_resource(directory: Path, file_name: str, content: str) -> Path:
    path = directory / "ami" / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def fake_ami_cache(
    cache: Path,
    name: str = "ami-fr-speakers-en-train",
    *,
    participants: str = AMI_PARTICIPANTS,
    meetings: str = AMI_MEETINGS,
    channels: dict[str, str] | None = None,
    headsets: bool = True,
) -> None:
    """A cache that looks like the AMI annotations archive plus the headsets it names.

    The headset recordings are written where the fetcher looks for them, which
    is how it takes them without the network. Each one is long enough that every
    timestamp of an annotation falls inside it.
    """
    directory = cache / name
    channels = channels or {
        "IS1003a.A": AMI_WORDS,
        "IS1003a.D": AMI_WORDS,
    }
    members = {
        ami.PARTICIPANTS: _ami_resource(directory, "participants.xml", participants),
        ami.MEETINGS: _ami_resource(directory, "meetings.xml", meetings),
    }
    for channel, content in channels.items():
        members[f"{ami.WORDS_DIRECTORY}/{channel}.words.xml"] = _ami_resource(
            directory, f"{channel}.words.xml", content
        )
    make_zip(directory / ami.ANNOTATIONS_NAME, members)
    if not headsets:
        return
    for channel in channels:
        meeting, _, agent = channel.partition(".")
        write_tone(
            directory / "audio" / f"{meeting}.Headset-{ami.head_number(agent)}.wav", seconds=10.0
        )


def test_ami_takes_the_words_of_a_french_speaker_and_groups_by_participant(dirs):
    data, cache = dirs
    fake_ami_cache(cache)

    count = ami.fetch_ami_fr_speakers_en_train(
        "ami-fr-speakers-en-train", cache, data / "ami-fr-speakers-en-train", verbose=False
    )
    # Two sentences and one fragment too short to be a clip, for each of the two
    # French speakers of the meeting.
    assert count == 4

    clips = {clip.id: clip for clip in read_manifest(data / "ami-fr-speakers-en-train")}
    assert set(clips) == {
        "ami-fr-speakers-en-train:IS1003a.A.0000",
        "ami-fr-speakers-en-train:IS1003a.A.0001",
        "ami-fr-speakers-en-train:IS1003a.D.0000",
        "ami-fr-speakers-en-train:IS1003a.D.0001",
    }
    first = clips["ami-fr-speakers-en-train:IS1003a.A.0000"]
    # Words and punctuation join the way the annotator wrote them, in cased text.
    assert first.text == "Okay so, I think."
    assert first.text_has_casing_punct is True
    assert first.group == "ami-fr-speakers-en-train:FIO017"
    assert first.group_kind == "speaker"
    assert first.source_split == "manual"
    assert first.license == "CC-BY-4.0"
    # The clip is the part of the meeting between its first and last word.
    assert first.duration_s == pytest.approx(4.1, abs=0.01)


def test_ami_keeps_one_speaker_of_a_meeting_out_of_another(dirs):
    """Each participant is a group of its own, even inside one meeting."""
    data, cache = dirs
    fake_ami_cache(cache)

    ami.fetch_ami_fr_speakers_en_train(
        "ami-fr-speakers-en-train", cache, data / "ami-fr-speakers-en-train", verbose=False
    )
    clips = read_manifest(data / "ami-fr-speakers-en-train")
    assert {clip.group for clip in clips} == {
        "ami-fr-speakers-en-train:FIO017",
        "ami-fr-speakers-en-train:MIO023",
    }
    assert sum(1 for clip in clips if clip.group.endswith("FIO017")) == 2


def test_ami_leaves_out_a_speaker_whose_first_language_is_not_french(dirs):
    """The source is the French-L1 participants of an English corpus, nobody else."""
    data, cache = dirs
    fake_ami_cache(cache)

    ami.fetch_ami_fr_speakers_en_train(
        "ami-fr-speakers-en-train", cache, data / "ami-fr-speakers-en-train", verbose=False
    )
    groups = {clip.group for clip in read_manifest(data / "ami-fr-speakers-en-train")}
    assert "ami-fr-speakers-en-train:MIO042" not in groups


def test_ami_cuts_at_a_sentence_end_and_at_a_silence(dirs):
    """A clip is a sentence; a silence of more than a second ends it just as well."""
    _, cache = dirs
    fake_ami_cache(cache)
    channel = ami.Channel(meeting="IS1003a", agent="A", speaker="FIO017")
    with zipfile.ZipFile(cache / "ami-fr-speakers-en-train" / ami.ANNOTATIONS_NAME) as archive:
        utterances = ami.utterances_of(archive, channel)

    assert [(u.start, u.end) for u in utterances] == [(0.1, 4.2), (6.0, 7.1), (8.0, 8.2)]
    assert [u.text for u in utterances] == ["Okay so, I think.", "Right?", "tiny"]
    # The laugh and the disfluency marker carry no text and separate nothing.
    assert [u.index for u in utterances] == [0, 1, 2]


def test_ami_does_not_let_a_clip_run_past_twenty_seconds(dirs):
    """A speaker who talks on without a pause still gets clips a recipe can batch."""
    _, cache = dirs
    words = "".join(
        f'  <w nite:id="w{i}" starttime="{i * 2.5:.2f}" endtime="{(i + 1) * 2.5:.2f}">word</w>\n'
        for i in range(12)
    )
    root = f'<root xmlns:nite="http://nite.sourceforge.net/">\n{words}</root>'
    fake_ami_cache(cache, channels={"IS1003a.A": root})
    channel = ami.Channel(meeting="IS1003a", agent="A", speaker="FIO017")
    with zipfile.ZipFile(cache / "ami-fr-speakers-en-train" / ami.ANNOTATIONS_NAME) as archive:
        utterances = ami.utterances_of(archive, channel)

    assert [round(u.end - u.start, 2) for u in utterances] == [20.0, 10.0]


def test_ami_copies_a_headset_recording_the_cache_does_not_have(dirs, tmp_path, monkeypatch):
    """The audio comes from the corpus's own mirror, in the letter's own channel."""
    data, cache = dirs
    fake_ami_cache(cache, headsets=False)
    mirror = tmp_path / "mirror" / "IS1003a" / "audio"
    write_tone(mirror / "IS1003a.Headset-0.wav", seconds=10.0)
    write_tone(mirror / "IS1003a.Headset-3.wav", seconds=10.0)
    monkeypatch.setattr(ami, "AUDIO_URL", str(tmp_path / "mirror"))

    count = ami.fetch_ami_fr_speakers_en_train(
        "ami-fr-speakers-en-train", cache, data / "ami-fr-speakers-en-train", verbose=False
    )
    assert count == 4
    assert (cache / "ami-fr-speakers-en-train" / "audio" / "IS1003a.Headset-0.wav").is_file()


def test_ami_refuses_an_archive_without_the_participants_resource(dirs):
    _, cache = dirs
    directory = cache / "ami-fr-speakers-en-train"
    make_zip(
        directory / ami.ANNOTATIONS_NAME,
        {ami.MEETINGS: _ami_resource(directory, "meetings.xml", AMI_MEETINGS)},
    )
    with pytest.raises(DownloadError, match="holds no corpusResources/participants.xml"):
        ami.read_channels(directory / ami.ANNOTATIONS_NAME)


def test_ami_refuses_a_speaker_the_annotations_do_not_describe(dirs):
    _, cache = dirs
    directory = cache / "ami-fr-speakers-en-train"
    fake_ami_cache(cache, channels={"IS1003a.B": AMI_WORDS})
    with pytest.raises(DownloadError, match="holds no words/IS1003a.A.words.xml"):
        with zipfile.ZipFile(directory / ami.ANNOTATIONS_NAME) as archive:
            ami.read_words(archive, ami.Channel(meeting="IS1003a", agent="A", speaker="FIO017"))


def test_ami_refuses_an_archive_of_speakers_without_a_french_one(dirs):
    _, cache = dirs
    directory = cache / "ami-fr-speakers-en-train"
    dutch = AMI_PARTICIPANTS.replace('native_language="French"', 'native_language="Dutch"')
    fake_ami_cache(cache, participants=dutch)
    with pytest.raises(DownloadError, match="no speaker of native language French"):
        ami.read_channels(directory / ami.ANNOTATIONS_NAME)


def test_ami_headsets_are_lettered_from_a(dirs):
    assert [ami.head_number(agent) for agent in "ABCD"] == [0, 1, 2, 3]
    with pytest.raises(DownloadError, match="no headset for speaker letter 'E'"):
        ami.head_number("E")
