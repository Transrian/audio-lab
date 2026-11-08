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
    granary,
    librispeech,
    librispeech_pc,
    mls,
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
