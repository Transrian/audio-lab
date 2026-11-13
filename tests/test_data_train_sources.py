"""Preparing training sources, exercised on tiny local fixtures instead of the network.

A training source is prepared like an evaluation set — the same manifest, the
same 16 kHz mono WAV — so these tests check what each source does differently:
which rows it reads, how it names a clip and its group, what it leaves out, and
that it stops as soon as the cap is reached. Everything runs on files written by
the test, so no test needs the network or a prepared dataset.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from helpers import (  # noqa: E402
    audio_cell,
    make_clip,  # noqa: E402
    write_parquet,
    write_tone,
)
from test_data_source import make_archive  # noqa: E402

from audiolab.core.stop import Stopped, request_stop  # noqa: E402
from audiolab.data import (  # noqa: E402
    african_french,
    ami,
    granary,
    icsi,
    librispeech,
    librispeech_pc,
    mls,
    notsofar,
    pxcorpus,
    voxpopuli_train,
)
from audiolab.data.download import (  # noqa: E402
    DownloadError,
    git_blob_sha1_file,
    sha256_file,
)
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
    The default names one speaker the corpus labels ``en_fr`` who has no row in
    the French shard, so nothing is excluded unless a test asks for it.
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
    accented = cache / "voxpopuli-en-nonnative"
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


def test_african_french_joins_an_apostrophe_only_after_a_letter():
    """A space before an apostrophe comes apart only when a word ends there."""
    assert african_french.training_text("grand ' mère et grand ' chose") == (
        "grand'mère et grand'chose"
    )
    # An apostrophe that opens a quotation has no letter before it.
    assert african_french.training_text("il a dit « ' bonjour ' » hier") == (
        "il a dit « ' bonjour ' » hier"
    )
    # Clean text is left exactly as it is.
    assert african_french.training_text("d'accord, aujourd'hui") == "d'accord, aujourd'hui"


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


def fake_african_french_test_cache(cache: Path) -> None:
    """The test part of SLR57: one transcript, one directory per speaker.

    Eleven speakers with a transcript between them and ``cameroon_m_006`` as the
    one a test may put in the prepared train manifest: a single overlap of
    eleven is under the tenth that makes the preparation stop. ``gabon_f_024``
    holds a recording with no transcript line and a ``devtest`` member is in the
    archive to prove it is not read.
    """
    directory = cache / "african-accented-fr-train"
    tone = write_tone(directory / "tone.wav", seconds=2.0)
    root = "African_Accented_French"
    speakers = (
        "gabon_f_026",
        "gabon_f_026",
        "gabon_m_060",
        "cameroon_m_006",
        "gabon_m_003",
        "gabon_m_058",
        "gabon_f_024",
        "gabon_m_072",
        "gabon_m_074",
        "chad_m_001",
        "congo_m_013",
    )
    members = {
        f"{root}/transcripts/test/ca16/prompts.txt": _text_file(
            directory,
            "test_prompts.txt",
            "".join(
                f"{speaker}_{index:03d} phrase {index} de {speaker}.\n"
                for index, speaker in enumerate(speakers, start=1)
            ),
        ),
        f"{root}/speech/test/ca16/gabon_f_024/gabon_f_024_999.wav": tone,
        f"{root}/transcripts/devtest/ca16_read/conditioned.txt": _text_file(
            directory, "devtest_conditioned.txt", "004_001 autre chose\n"
        ),
        f"{root}/speech/devtest/ca16/004/004_001.wav": tone,
    }
    for index, speaker in enumerate(speakers, start=1):
        key = f"{speaker}_{index:03d}"
        members[f"{root}/speech/test/ca16/{speaker}/{key}.wav"] = tone
    make_archive(directory / "African_Accented_French.tar.gz", members)


def test_african_french_test_reads_only_the_test_part_and_drops_an_overlap(dirs, monkeypatch):
    """The test part is prepared in full; a train speaker in it is dropped."""
    data, cache = dirs
    fake_african_french_test_cache(cache)
    monkeypatch.setattr(african_french, "TEST_EXPECTED_CLIPS", 11, raising=True)
    write_manifest(
        data / "african-accented-fr-train",
        [
            make_clip("african-accented-fr-train:113", group="african-accented-fr-train:113"),
            make_clip(
                "african-accented-fr-train:cameroon_m_006",
                group="african-accented-fr-train:cameroon_m_006",
            ),
        ],
    )

    count = african_french.fetch_african_accented_fr_test(
        "african-accented-fr-test", cache, data / "african-accented-fr-test", verbose=False
    )
    clips = {clip.id: clip for clip in read_manifest(data / "african-accented-fr-test")}
    assert count == 10
    # The overlapping speaker is out; the recording without a transcript is out;
    # the devtest member of the archive is not read.
    assert "african-accented-fr-test:cameroon_m_006_004" not in clips
    assert "african-accented-fr-test:gabon_f_024_999" not in clips
    assert "african-accented-fr-test:004_001" not in clips
    first = clips["african-accented-fr-test:gabon_f_026_001"]
    # The corpus's own text is kept — casing and punctuation included.
    assert first.text == "phrase 1 de gabon_f_026."
    assert first.text_has_casing_punct is True
    assert first.group == "african-accented-fr-test:gabon_f_026"
    assert first.group_kind == "speaker"
    assert first.source_split == "test"
    assert first.license == "Apache-2.0"
    assert first.category == "read"


def test_african_french_test_stops_when_most_speakers_overlap(dirs, monkeypatch):
    """A test part that shares most of its speakers with the train set is refused."""
    data, cache = dirs
    fake_african_french_test_cache(cache)
    monkeypatch.setattr(african_french, "TEST_EXPECTED_CLIPS", 11, raising=True)
    write_manifest(
        data / "african-accented-fr-train",
        [
            make_clip(
                f"african-accented-fr-train:{speaker}",
                group=f"african-accented-fr-train:{speaker}",
            )
            for speaker in (
                "gabon_f_026",
                "gabon_m_060",
                "cameroon_m_006",
                "gabon_m_003",
                "gabon_m_058",
                "gabon_f_024",
                "gabon_m_072",
                "gabon_m_074",
                "chad_m_001",
                "congo_m_013",
            )
        ],
    )
    with pytest.raises(DownloadError, match="overlap the prepared train set"):
        african_french.fetch_african_accented_fr_test(
            "african-accented-fr-test", cache, data / "african-accented-fr-test", verbose=False
        )


def test_african_french_train_leaves_out_the_test_speakers(dirs):
    """The train fetcher takes its exclusions from the prepared test manifest."""
    data, cache = dirs
    fake_african_french_cache(cache)
    write_manifest(
        data / "african-accented-fr-test",
        [make_clip("african-accented-fr-test:113", group="african-accented-fr-test:113")],
    )

    count = african_french.fetch_african_accented_fr_train(
        "african-accented-fr-train", cache, data / "african-accented-fr-train", verbose=False
    )
    # Both ca16 rows of speaker 113 are out; the yaoundé rows stand.
    assert count == 1
    assert {clip.id for clip in read_manifest(data / "african-accented-fr-train")} == {
        "african-accented-fr-train:ctell1-01-0002",
    }


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
    # A heading keeps its punctuation, and a capitalised word that is not a
    # heading does not become one.
    assert librispeech_pc.title_run("THE MARCH, THE MIXER'S AHEAD He went.") == 5
    assert librispeech_pc.title_run("The MR. JONES report is ready.") == 2
    assert librispeech_pc.title_run("THE MR JONES He went home.") == 3


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
    # The transcript is kept as the corpus published it, minus the slash that
    # marks the word the recording button cut off.
    assert first.text == "flagyl 500 milligrammes euh qu en cachet pendant 8 jours"
    assert first.text_has_casing_punct is False
    assert first.group == "pxcorpus-fr-train:24"
    assert first.group_kind == "speaker"
    assert first.source_split == "recordings"
    assert first.license == "CC-BY-4.0"


def test_pxcorpus_rejoins_a_word_the_transcript_marked_with_a_slash(dirs):
    """``comp/primés`` is the word "comprimés": the slash is a mark, not speech."""
    _, cache = dirs
    directory = cache / "pxcorpus-fr-train"
    tone = write_tone(directory / "tone.wav", seconds=3.0)
    root = "PxCorpus/PxSLU"
    make_zip(
        directory / "PxCorpus.zip",
        {
            f"{root}/paths.txt": _text_file(directory, "paths.txt", "s/recording_0.wav\n"),
            f"{root}/metadata.txt": _text_file(directory, "metadata.txt", "24,18-28,F,doctor\n"),
            f"{root}/seq.in": _text_file(
                directory, "seq.in", "2/ comprimés et /pone  matin et soir\n"
            ),
            f"{root}/recordings/s/recording_0.wav": tone,
        },
    )
    rows = pxcorpus.read_rows("pxcorpus-fr-train", directory / "PxCorpus.zip")

    assert [row.text for row in rows] == ["2 comprimés et pone matin et soir"]


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

# The headset channel of a speaker is what meetings.xml says, not what the
# speaker letter usually means: A is channel 1 here.
AMI_MEETINGS = """<?xml version="1.0" encoding="ISO-8859-1"?>
<root xmlns:nite="http://nite.sourceforge.net/">
  <meeting observation="IS1003a">
    <speaker nite:id="IS1003a.A" nxt_agent="A" channel="1" global_name="FIO017" role="PM"/>
    <speaker nite:id="IS1003a.B" nxt_agent="B" channel="0" global_name="MIO042" role="ME"/>
    <speaker nite:id="IS1003a.D" nxt_agent="D" channel="3" global_name="MIO023" role="ID"/>
  </meeting>
  <meeting observation="ES2001a">
    <speaker nite:id="ES2001a.B" nxt_agent="B" channel="0" global_name="MIO042" role="PM"/>
  </meeting>
</root>
"""

# One speaker's manual words: two sentences, a laugh, a disfluency marker and a
# gap in the middle of a sentence, then a fragment too short to be a clip.
AMI_WORDS = """<?xml version="1.0" encoding="ISO-8859-1"?>
<root xmlns:nite="http://nite.sourceforge.net/">
  <w nite:id="w1" starttime="0.100" endtime="0.500">Okay</w>
  <w nite:id="w2" starttime="0.500" endtime="1.400">so</w>
  <w nite:id="w3" starttime="1.400" endtime="2.000" punc="true">,</w>
  <vocalsound nite:id="v1" starttime="2.000" endtime="2.500" type="laugh"/>
  <w nite:id="w4" starttime="2.600" endtime="3.000">I</w>
  <w nite:id="w5" starttime="3.000" endtime="3.500">think</w>
  <w nite:id="w6" starttime="3.500" endtime="3.700">that</w>
  <gap nite:id="g1" starttime="3.700" endtime="4.600"/>
  <w nite:id="w7" starttime="4.700" endtime="5.200">this</w>
  <w nite:id="w8" starttime="5.200" endtime="5.900">works</w>
  <w nite:id="w9" starttime="5.900" endtime="6.200" punc="true">.</w>
  <disfmarker nite:id="d1" starttime="6.200" endtime="6.300" type="disfluency"/>
  <w nite:id="w10" starttime="6.400" endtime="7.000">Right</w>
  <w nite:id="w11" starttime="7.000" endtime="7.300" punc="true">?</w>
  <w nite:id="w12" starttime="8.000" endtime="8.200">tiny</w>
</root>
"""


def _ami_resource(directory: Path, file_name: str, content: str) -> Path:
    path = directory / "ami" / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _ami_channels(meetings: str) -> dict[str, int]:
    """The headset channel of every speaker of the fixture's meetings."""
    root = ET.fromstring(meetings)
    return {
        f"{meeting.attrib['observation']}.{speaker.attrib['nxt_agent']}": int(
            speaker.attrib["channel"]
        )
        for meeting in root
        for speaker in meeting
    }


def fake_ami_cache(
    cache: Path,
    monkeypatch=None,
    name: str = "ami-fr-speakers-en-train",
    *,
    participants: str = AMI_PARTICIPANTS,
    meetings: str = AMI_MEETINGS,
    channels: dict[str, str] | None = None,
    headsets: bool = True,
) -> None:
    """A cache that looks like the AMI annotations archive plus the headsets it names.

    The headset recordings are written where the fetcher looks for them — under
    the channel the meeting gives the speaker — which is how it takes them
    without the network. Each one is long enough that every timestamp of an
    annotation falls inside it. With ``monkeypatch``, the recordings of the
    fixture are pinned by the digest of the tone they are made of, the way the
    corpus's own recordings are pinned.
    """
    directory = cache / name
    channels = channels or {"IS1003a.A": AMI_WORDS, "IS1003a.D": AMI_WORDS}
    members = {
        ami.PARTICIPANTS: _ami_resource(directory, "participants.xml", participants),
        ami.MEETINGS: _ami_resource(directory, "meetings.xml", meetings),
    }
    for channel, content in channels.items():
        members[f"{ami.WORDS_DIRECTORY}/{channel}.words.xml"] = _ami_resource(
            directory, f"{channel}.words.xml", content
        )
    make_zip(directory / ami.ANNOTATIONS_NAME, members)
    heads = _ami_channels(meetings)
    recordings = [
        f"{channel.partition('.')[0]}.Headset-{heads[channel]}.wav" for channel in channels
    ]
    if monkeypatch is not None:
        # The recordings of the fixture are tones, so they are pinned by the
        # digest of a tone; the corpus's own recordings are pinned one by one.
        digest = sha256_file(write_tone(directory / "tone.wav", seconds=10.0))
        monkeypatch.setattr(ami, "PINNED_RECORDINGS", dict.fromkeys(recordings, digest))
    if not headsets:
        return
    for recording in recordings:
        write_tone(directory / "audio" / recording, seconds=10.0)


def test_ami_takes_the_words_of_a_french_speaker_and_groups_by_participant(dirs, monkeypatch):
    data, cache = dirs
    fake_ami_cache(cache, monkeypatch)

    count = ami.fetch_ami_fr_speakers_en_train(
        "ami-fr-speakers-en-train", cache, data / "ami-fr-speakers-en-train", verbose=False
    )
    # Three sentences and one fragment too short to be a clip, for each of the
    # two French speakers of the meeting.
    assert count == 6

    clips = {clip.id: clip for clip in read_manifest(data / "ami-fr-speakers-en-train")}
    assert set(clips) == {
        "ami-fr-speakers-en-train:IS1003a.A.0000",
        "ami-fr-speakers-en-train:IS1003a.A.0001",
        "ami-fr-speakers-en-train:IS1003a.A.0002",
        "ami-fr-speakers-en-train:IS1003a.D.0000",
        "ami-fr-speakers-en-train:IS1003a.D.0001",
        "ami-fr-speakers-en-train:IS1003a.D.0002",
    }
    first = clips["ami-fr-speakers-en-train:IS1003a.A.0000"]
    # Words and punctuation join the way the annotator wrote them, in cased text.
    assert first.text == "Okay so, I think that"
    assert first.text_has_casing_punct is True
    assert first.group == "ami-fr-speakers-en-train:FIO017"
    assert first.group_kind == "speaker"
    assert first.source_split == "manual"
    assert first.license == "CC-BY-4.0"
    # The clip is the part of the meeting between its first and last word.
    assert first.duration_s == pytest.approx(3.6, abs=0.01)


def test_ami_does_not_let_a_clip_span_a_gap(dirs):
    """The audio under a gap has no text, so a clip that spanned it would be mislabelled."""
    _, cache = dirs
    fake_ami_cache(cache)
    channel = ami.Channel(meeting="IS1003a", agent="A", speaker="FIO017", channel=1)
    with zipfile.ZipFile(cache / "ami-fr-speakers-en-train" / ami.ANNOTATIONS_NAME) as archive:
        utterances = ami.utterances_of(archive, channel)

    bounds = [(u.start, u.end) for u in utterances]
    assert bounds == [(0.1, 3.7), (4.7, 6.2), (6.4, 7.3), (8.0, 8.2)]
    # Without the gap closing the clip, the first one would run to 6.2 s and say
    # words the audio between 3.7 s and 4.6 s does not have.
    assert [u.text for u in utterances] == [
        "Okay so, I think that",
        "this works.",
        "Right?",
        "tiny",
    ]


def test_ami_writes_a_word_spelled_letter_by_letter_as_the_word_it_spells(dirs):
    """``T_V_`` is how the annotators wrote "TV", not something the speaker said."""
    _, cache = dirs
    words = """<root xmlns:nite="http://nite.sourceforge.net/">
  <w nite:id="w1" starttime="0.100" endtime="0.500">We</w>
  <w nite:id="w2" starttime="0.500" endtime="1.000">watched</w>
  <w nite:id="w3" starttime="1.000" endtime="1.400">T_V_</w>
  <w nite:id="w4" starttime="1.400" endtime="1.800">on</w>
  <w nite:id="w5" starttime="1.800" endtime="2.200">the</w>
  <w nite:id="w6" starttime="2.200" endtime="2.600">L_C_D_</w>
  <w nite:id="w7" starttime="2.600" endtime="2.900" punc="true">.</w>
</root>"""
    fake_ami_cache(cache, channels={"IS1003a.A": words})
    channel = ami.Channel(meeting="IS1003a", agent="A", speaker="FIO017", channel=1)
    with zipfile.ZipFile(cache / "ami-fr-speakers-en-train" / ami.ANNOTATIONS_NAME) as archive:
        utterances = ami.utterances_of(archive, channel)

    assert [u.text for u in utterances] == ["We watched TV on the LCD."]


def test_ami_keeps_one_speaker_of_a_meeting_out_of_another(dirs, monkeypatch):
    """Each participant is a group of its own, even inside one meeting."""
    data, cache = dirs
    fake_ami_cache(cache, monkeypatch)

    ami.fetch_ami_fr_speakers_en_train(
        "ami-fr-speakers-en-train", cache, data / "ami-fr-speakers-en-train", verbose=False
    )
    clips = read_manifest(data / "ami-fr-speakers-en-train")
    assert {clip.group for clip in clips} == {
        "ami-fr-speakers-en-train:FIO017",
        "ami-fr-speakers-en-train:MIO023",
    }
    assert sum(1 for clip in clips if clip.group.endswith("FIO017")) == 3


def test_ami_leaves_out_a_speaker_whose_first_language_is_not_french(dirs, monkeypatch):
    """The source is the French-L1 participants of an English corpus, nobody else."""
    data, cache = dirs
    fake_ami_cache(cache, monkeypatch)

    ami.fetch_ami_fr_speakers_en_train(
        "ami-fr-speakers-en-train", cache, data / "ami-fr-speakers-en-train", verbose=False
    )
    groups = {clip.group for clip in read_manifest(data / "ami-fr-speakers-en-train")}
    assert "ami-fr-speakers-en-train:MIO042" not in groups


def test_ami_reads_the_headset_channel_the_meeting_names(dirs):
    """``A`` is ``Headset-0`` in most meetings but not in all of them."""
    _, cache = dirs
    fake_ami_cache(cache)
    channels = ami.read_channels(cache / "ami-fr-speakers-en-train" / ami.ANNOTATIONS_NAME)

    assert [(c.agent, c.channel) for c in channels] == [("A", 1), ("D", 3)]


def test_ami_does_not_let_a_clip_run_past_twenty_seconds(dirs):
    """A speaker who talks on without a pause still gets clips a recipe can batch."""
    _, cache = dirs
    words = "".join(
        f'  <w nite:id="w{i}" starttime="{i * 2.5:.2f}" endtime="{(i + 1) * 2.5:.2f}">word</w>\n'
        for i in range(12)
    )
    root = f'<root xmlns:nite="http://nite.sourceforge.net/">\n{words}</root>'
    fake_ami_cache(cache, channels={"IS1003a.A": root})
    channel = ami.Channel(meeting="IS1003a", agent="A", speaker="FIO017", channel=1)
    with zipfile.ZipFile(cache / "ami-fr-speakers-en-train" / ami.ANNOTATIONS_NAME) as archive:
        utterances = ami.utterances_of(archive, channel)

    assert [round(u.end - u.start, 2) for u in utterances] == [20.0, 10.0]


def test_ami_copies_a_headset_recording_the_cache_does_not_have(dirs, tmp_path, monkeypatch):
    """The audio comes from the corpus's own mirror, in the channel it names."""
    data, cache = dirs
    fake_ami_cache(cache, monkeypatch, headsets=False)
    mirror = tmp_path / "mirror" / "IS1003a" / "audio"
    write_tone(mirror / "IS1003a.Headset-1.wav", seconds=10.0)
    write_tone(mirror / "IS1003a.Headset-3.wav", seconds=10.0)
    monkeypatch.setattr(ami, "AUDIO_URL", str(tmp_path / "mirror"))

    count = ami.fetch_ami_fr_speakers_en_train(
        "ami-fr-speakers-en-train", cache, data / "ami-fr-speakers-en-train", verbose=False
    )
    assert count == 6
    assert (cache / "ami-fr-speakers-en-train" / "audio" / "IS1003a.Headset-1.wav").is_file()


def test_ami_refuses_a_recording_it_has_no_pin_for(dirs):
    """A recording the source did not pin is not taken unverified."""
    data, cache = dirs
    fake_ami_cache(cache)
    with pytest.raises(DownloadError, match="has no pinned sha256"):
        ami.fetch_ami_fr_speakers_en_train(
            "ami-fr-speakers-en-train", cache, data / "ami-fr-speakers-en-train", verbose=False
        )


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
            ami.read_words(
                archive,
                ami.Channel(meeting="IS1003a", agent="A", speaker="FIO017", channel=1),
            )


def test_ami_refuses_a_word_the_annotations_give_no_time_for(dirs):
    """A word that cannot be placed in the recording cannot be a clip either."""
    _, cache = dirs
    directory = cache / "ami-fr-speakers-en-train"
    timeless = AMI_WORDS.replace(
        '<w nite:id="w1" starttime="0.100" endtime="0.500">Okay</w>', '<w nite:id="w1">Okay</w>'
    )
    fake_ami_cache(cache, channels={"IS1003a.A": timeless})
    with pytest.raises(DownloadError, match="holds a word without a time: 'Okay'"):
        with zipfile.ZipFile(directory / ami.ANNOTATIONS_NAME) as archive:
            ami.read_words(
                archive,
                ami.Channel(meeting="IS1003a", agent="A", speaker="FIO017", channel=1),
            )


def test_ami_refuses_an_archive_of_speakers_without_a_french_one(dirs):
    _, cache = dirs
    directory = cache / "ami-fr-speakers-en-train"
    dutch = AMI_PARTICIPANTS.replace('native_language="French"', 'native_language="Dutch"')
    fake_ami_cache(cache, participants=dutch)
    with pytest.raises(DownloadError, match="no speaker of native language French"):
        ami.read_channels(directory / ami.ANNOTATIONS_NAME)


# --- ICSI -------------------------------------------------------------------

# Two meetings in the shape the corpus publishes its MRT transcripts: one
# segment per utterance, the participant's own microphone in the preamble, and
# the annotation elements between the words.
ICSI_MEETINGS = {
    "Btr001": """<?xml version="1.0" encoding="iso-8859-1"?>
<Meeting Session="Btr001" DateTimeStamp="2000-01-10-1000" Version="2">
  <Preamble>
    <Participants>
      <Participant Name="fe068" Channel="chan0" Seat="3"/>
      <Participant Name="fe069" Channel="chan1" Seat="4"/>
      <Participant Name="me011"/>
    </Participants>
  </Preamble>
  <Transcript StartTime="0.0" EndTime="600.0">
    <Segment StartTime="0.000" EndTime="1.770" Participant="fe068">
      We have littler heads so.
    </Segment>
    <Segment StartTime="2.000" EndTime="3.000" Participant="fe069">
      Is - is M- So, is - is Marisa's <VocalSound Description="laugh"/> close enough?
    </Segment>
    <Segment StartTime="3.400" EndTime="4.400" Participant="fe068">
      O_K. <Comment Description="coughs"/> So <Emphasis>that</Emphasis> is fine.
    </Segment>
    <Segment StartTime="4.600" EndTime="9.800" Participant="fe069" CloseMic="false">
      Would you check the microphone adjustments, please?
    </Segment>
    <Segment StartTime="5.000" EndTime="5.300" Participant="fe069">
      Yeah.
    </Segment>
    <Segment StartTime="6.000" EndTime="6.400" Participant="fe069">
      <VocalSound Description="clears throat"/>
    </Segment>
    <Segment StartTime="6.500" EndTime="7.000" CloseMic="false">
      <NonVocalSound Description="door slams"/>
    </Segment>
  </Transcript>
</Meeting>
""",
    "Bnr002": """<?xml version="1.0" encoding="iso-8859-1"?>
<Meeting Session="Bnr002" DateTimeStamp="2000-02-10-1000" Version="2">
  <Preamble>
    <Participants>
      <Participant Name="fe070" Channel="chan0" Seat="2"/>
      <Participant Name="fe071" Channel="chan1" Seat="5"/>
      <Participant Name="me011"/>
    </Participants>
  </Preamble>
  <Transcript StartTime="0.0" EndTime="600.0">
    <Segment StartTime="0.500" EndTime="3.500" Participant="fe070">
      Right, let's start with the agenda.
    </Segment>
    <Segment StartTime="3.600" EndTime="8.000" Participant="fe071" CloseMic="false">
      I heard that on somebody else's microphone.
    </Segment>
    <Segment StartTime="4.000" EndTime="6.000" Participant="me011" CloseMic="false">
      So did I.
    </Segment>
  </Transcript>
</Meeting>
""",
}


def write_sphere(
    path: Path, seconds: float = 10.03, rate: int = 16000, frequency: int = 440
) -> Path:
    """A NIST SPHERE file of plain samples, the shape ICSI's own headers declare.

    The corpus's own files are shorten-compressed, which no reader here decodes;
    ``ffmpeg`` is what turns them into the samples this project uses, and this
    fixture is what those samples are made of when they are already plain, so
    the decode can be exercised without the corpus. The header carries the count
    and the checksum of the samples, as the corpus's headers do.
    """
    times = np.arange(int(seconds * rate)) / rate
    samples = (0.3 * np.sin(2 * np.pi * frequency * times) * 32767).astype("<i2")
    checksum = int(samples.astype("int64").sum() % icsi.CHECKSUM_MODULUS)
    fields = [
        "sample_n_bytes -i 2",
        "channel_count -i 1",
        f"sample_count -i {len(samples)}",
        f"sample_rate -i {rate}",
        "sample_coding -s4 pcm",
        "sample_byte_format -s2 01",
        f"sample_checksum -i {checksum}",
    ]
    header = "NIST_1A\n   1024\n" + "\n".join(fields) + "\nend_head\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        header.encode("latin-1").ljust(icsi.SPHERE_HEADER_BYTES, b"\n") + samples.tobytes()
    )
    return path


def fake_icsi_cache(cache: Path, monkeypatch, meetings: dict[str, str] | None = None) -> Path:
    """A cache that looks like the corpus's transcript archive and its signals.

    The archive is pinned by its own digest, the way the corpus's is, and the
    signals are files of the fixture rather than of the corpus: the fetcher
    downloads by URL, and a URL that is already a local file is copied, so the
    preparation runs without the network.
    """
    directory = cache / "icsi-en-train"
    members = {
        f"{icsi.TRANSCRIPTS_DIRECTORY}/{session}.mrt": _icsi_member(directory, session, content)
        for session, content in (meetings or ICSI_MEETINGS).items()
    }
    # The corpus also serves the preambles of every meeting in one file, which
    # holds no segment and must not be read as a meeting of its own.
    members[icsi.MEETINGS_FILE] = _icsi_member(directory, "preambles", "<Meetings/>")
    archive = make_zip(directory / icsi.TRANSCRIPTS_NAME, members)
    monkeypatch.setattr(icsi, "TRANSCRIPTS_URL", str(archive))
    monkeypatch.setattr(icsi, "PINNED_SHA256", {icsi.TRANSCRIPTS_NAME: sha256_file(archive)})
    monkeypatch.setattr(icsi, "AUDIO_URL", str(directory / "audio"))
    return directory


def _icsi_member(directory: Path, name: str, content: str) -> Path:
    path = directory / "members" / f"{name}.mrt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_an_ffmpeg_a_stop_interrupted_is_a_stop_not_a_file_it_cannot_decode(tmp_path, monkeypatch):
    stopping = [False]

    def interrupted_ffmpeg(command, **options):
        if stopping[0]:
            request_stop()  # the Ctrl-C reaches ffmpeg too, which exits with 255
        return subprocess.CompletedProcess(command, 255, b"", b"Exiting normally, signal 2.")

    monkeypatch.setattr(icsi.shutil, "which", lambda name: name)
    monkeypatch.setattr(icsi.subprocess, "run", interrupted_ffmpeg)
    source, destination = tmp_path / "Bmr001.sph", tmp_path / "Bmr001.wav"

    with pytest.raises(DownloadError, match="ffmpeg cannot decode Bmr001.sph"):
        icsi._decode(source, destination)
    stopping[0] = True
    with pytest.raises(Stopped, match="decoding of Bmr001.sph stopped"):
        icsi._decode(source, destination)


def fake_icsi_decode(monkeypatch) -> None:
    """Decode the fixture's signals without ffmpeg, checking them the same way.

    The corpus's own signals are shorten-compressed and ffmpeg is the only
    decoder on hand for them; the fixture's are plain, so a preparation can be
    exercised on a machine without it. What the fake writes is checked with the
    same function the real decode checks ffmpeg's output with, so nothing about
    the samples the corpus declares is lost by not running ffmpeg.
    """

    def decode(source: Path, destination: Path) -> None:
        samples, rate = sf.read(str(source), dtype="int16", always_2d=False)
        icsi.verify_samples(source, samples)
        staged = destination.with_name(f"{destination.name}.part")
        sf.write(str(staged), samples, rate, format="WAV", subtype="PCM_16")
        staged.replace(destination)

    monkeypatch.setattr(icsi, "_decode", decode)


def fake_icsi_signals(directory: Path, sources: list[str]) -> None:
    """The signals the fixture's segments are sliced from, one SPH file each.

    Only the channels the fixture's segments are sliced from are written, so a
    preparation that asked for a channel it does not need would fail here the way
    a wrong URL fails against the corpus.
    """
    for source in sources:
        write_sphere(directory / "audio" / f"{source}.sph")


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg is what decodes the shorten-compressed audio of ICSI",
)


def test_icsi_reads_the_mrt_segments_and_groups_by_participant(dirs, monkeypatch):
    data, cache = dirs
    directory = fake_icsi_cache(cache, monkeypatch)
    fake_icsi_decode(monkeypatch)
    fake_icsi_signals(directory, ["Btr001/chan0", "Btr001/chan1", "Bnr002/chan0"])

    count = icsi.fetch_icsi_en_train("icsi-en-train", cache, data / "icsi-en-train", verbose=False)
    # Three segments of the first meeting and one of the second; the segment the
    # close microphone did not carry, the one too short to be a clip, the one
    # with nothing but a vocal sound and the one without a participant are not
    # clips, and the participant without a microphone of their own is not read.
    assert count == 4

    clips = {clip.id: clip for clip in read_manifest(data / "icsi-en-train")}
    assert set(clips) == {
        "icsi-en-train:Btr001.fe068.0000",
        "icsi-en-train:Btr001.fe069.0001",
        "icsi-en-train:Btr001.fe068.0002",
        "icsi-en-train:Bnr002.fe070.0000",
    }
    first = clips["icsi-en-train:Btr001.fe068.0000"]
    # The text is the annotators' own, cased and punctuated, and the clip is the
    # part of the meeting between the times they gave the segment.
    assert first.text == "We have littler heads so."
    assert first.text_has_casing_punct is True
    assert first.duration_s == pytest.approx(1.77, abs=0.01)
    assert first.group == "icsi-en-train:fe068"
    assert first.group_kind == "speaker"
    assert first.source_split == "mrt"
    assert first.language == "en"
    assert first.license == "CC-BY-4.0"
    # A vocal sound is not a word, and an emphasised word is: the markup goes,
    # the words it was written around stay. The dash of a restart ("Is - is") is
    # a break, which the label writes as nothing (plain_label).
    assert clips["icsi-en-train:Btr001.fe069.0001"].text == (
        "Is is M- So, is is Marisa's close enough?"
    )
    # A token written letter by letter is the word it spells, as in AMI.
    assert clips["icsi-en-train:Btr001.fe068.0002"].text == "OK. So that is fine."
    assert {clip.group for clip in clips.values()} == {
        "icsi-en-train:fe068",
        "icsi-en-train:fe069",
        "icsi-en-train:fe070",
    }


def test_icsi_downloads_only_the_channels_a_clip_is_sliced_from(dirs, monkeypatch):
    """A preparation reads the channels its clips are on, not every microphone.

    The preambles of the corpus name 469 microphones across its 75 meetings; a
    preparation of the whole corpus reads the ones a planned clip is sliced from.
    """
    data, cache = dirs
    directory = fake_icsi_cache(cache, monkeypatch)
    fake_icsi_decode(monkeypatch)
    # No signal is written for Bnr002/chan1 or for the channel of a participant
    # the preamble does not give one to: asking for either would fail.
    fake_icsi_signals(directory, ["Btr001/chan0", "Btr001/chan1", "Bnr002/chan0"])

    icsi.fetch_icsi_en_train("icsi-en-train", cache, data / "icsi-en-train", verbose=False)

    assert sorted(path.name for path in (directory / "audio").glob("*/*.sph")) == [
        "chan0.sph",
        "chan0.sph",
        "chan1.sph",
    ]


@requires_ffmpeg
def test_icsi_decodes_the_samples_the_header_of_a_recording_declares(dirs, monkeypatch):
    _, cache = dirs
    directory = fake_icsi_cache(cache, monkeypatch)
    fake_icsi_signals(directory, ["Btr001/chan0"])

    audio = icsi.load_recording("icsi-en-train", cache, "Btr001", "chan0")

    assert audio.sample_rate == 16000
    assert audio.duration_s == pytest.approx(10.03, abs=0.01)
    assert (cache / "icsi-en-train" / "audio" / "Btr001" / "chan0.wav").is_file()


def test_icsi_refuses_samples_that_are_not_what_the_header_declares(tmp_path):
    """The header's count and sum are what says the samples are the recording.

    A decoder that quietly dropped or repeated samples would misalign every clip
    on the channel, and samples that do not add up to the sum the corpus wrote
    are not the recording the transcript times were made on. The check reads the
    header alone, so it runs without a decoder.
    """
    source = write_sphere(tmp_path / "chan0.sph")
    samples, _ = sf.read(str(source), dtype="int16", always_2d=False)
    icsi.verify_samples(source, samples)

    with pytest.raises(DownloadError, match="its own header says 160480"):
        icsi.verify_samples(source, samples[:-1])

    # One sample louder, the header left alone: the count still matches and the
    # sum does not.
    louder = samples.copy()
    louder[100] = louder[100] + 1
    with pytest.raises(DownloadError, match="do not match its own checksum"):
        icsi.verify_samples(source, louder)


@requires_ffmpeg
def test_icsi_refuses_a_recording_whose_samples_were_changed_under_its_header(tmp_path):
    """What the decode produced is checked before it is kept, not after."""
    source = write_sphere(tmp_path / "chan0.sph")
    changed = bytearray(source.read_bytes())
    # One sample of the audio flipped, the header and the file length untouched.
    changed[icsi.SPHERE_HEADER_BYTES + 2 * 100] ^= 0x01
    source.write_bytes(bytes(changed))

    with pytest.raises(DownloadError, match="do not match its own checksum"):
        icsi._decode(source, tmp_path / "chan0.wav")
    assert not (tmp_path / "chan0.wav").exists()
    assert not (tmp_path / "chan0.wav.part").exists()


def test_icsi_joins_the_segments_of_one_participant_into_a_clip(dirs, monkeypatch):
    """The annotators cut a speaker's turn at pauses; a clip is the sentence they said."""
    data, cache = dirs
    meeting = """<?xml version="1.0" encoding="iso-8859-1"?>
<Meeting Session="Ctr003">
  <Preamble>
    <Participants>
      <Participant Name="fe068" Channel="chan0" Seat="1"/>
    </Participants>
  </Preamble>
  <Transcript StartTime="0.0" EndTime="600.0">
    <Segment StartTime="1.000" EndTime="2.000" Participant="fe068">
      We have littler heads
    </Segment>
    <Segment StartTime="2.300" EndTime="4.000" Participant="fe068">
      so. So that is fine.
    </Segment>
    <Segment StartTime="4.500" EndTime="5.000" Participant="fe068">
      And another thing.
    </Segment>
  </Transcript>
</Meeting>
"""
    directory = fake_icsi_cache(cache, monkeypatch, meetings={"Ctr003": meeting})
    fake_icsi_decode(monkeypatch)
    fake_icsi_signals(directory, ["Ctr003/chan0"])

    icsi.fetch_icsi_en_train("icsi-en-train", cache, data / "icsi-en-train", verbose=False)

    clips = read_manifest(data / "icsi-en-train")
    # The first two segments are one sentence the annotators cut at a pause, so
    # they are one clip; the third follows a sentence the annotators closed and
    # is a clip of its own.
    assert [clip.id for clip in clips] == [
        "icsi-en-train:Ctr003.fe068.0000",
        "icsi-en-train:Ctr003.fe068.0002",
    ]
    assert clips[0].text == "We have littler heads so. So that is fine."
    assert clips[0].duration_s == pytest.approx(3.0, abs=0.01)
    assert clips[1].text == "And another thing."


def test_segment_text_keeps_the_words_of_markup_written_inside_markup():
    """The annotation nests, and a word inside a pronunciation is still a word."""
    element = ET.fromstring(
        "<Segment>the <Emphasis>real <Pronounce>places</Pronounce></Emphasis> "
        "are <Comment>coughs</Comment>here.</Segment>"
    )

    assert icsi.segment_text(element) == "the real places are here."


def test_icsi_leaves_out_what_it_cannot_write_a_reference_for(dirs, monkeypatch):
    """A digit task, a word nobody made out and a re-sliced channel are not clips."""
    data, cache = dirs
    meeting = """<?xml version="1.0" encoding="iso-8859-1"?>
<Meeting Session="Dtr004">
  <Preamble>
    <Participants>
      <Participant Name="fe068" Channel="chan0" Seat="1"/>
      <Participant Name="fe069" Channel="chan1" Seat="2"/>
    </Participants>
  </Preamble>
  <Transcript StartTime="0.0" EndTime="600.0">
    <Segment StartTime="1.000" EndTime="2.000" Participant="fe068">
      We have littler heads so.
    </Segment>
    <Segment StartTime="2.500" EndTime="3.000" Participant="fe068" DigitTask="true">
      Six three, three three.
    </Segment>
    <Segment StartTime="3.400" EndTime="4.000" Participant="fe068">
      And then <Uncertain> @@ </Uncertain> happened.
    </Segment>
    <Segment StartTime="4.400" EndTime="5.000" Participant="fe068">
      This one is fine.
    </Segment>
    <Segment StartTime="5.400" EndTime="6.000" Participant="fe068" Channel="chan8">
      And this is the same speech on another microphone.
    </Segment>
    <Segment StartTime="6.400" EndTime="7.200" Participant="fe068">
      And the last thing on this channel
    </Segment>
    <Segment StartTime="7.400" EndTime="7.700" Participant="fe068" DigitTask="true">
      Nine nine two.
    </Segment>
    <Segment StartTime="7.900" EndTime="8.600" Participant="fe068">
      was not a digit task.
    </Segment>
    <Segment StartTime="1.000" EndTime="2.500" Participant="fe069">
      The <Emphasis><Pronounce>places</Pronounce></Emphasis> are here.
    </Segment>
  </Transcript>
</Meeting>
"""
    directory = fake_icsi_cache(cache, monkeypatch, meetings={"Dtr004": meeting})
    fake_icsi_decode(monkeypatch)
    # chan8 is not among them: a preparation that kept the re-sliced segment
    # would ask the corpus for a file it has no clip to cut.
    fake_icsi_signals(directory, ["Dtr004/chan0", "Dtr004/chan1"])

    icsi.fetch_icsi_en_train("icsi-en-train", cache, data / "icsi-en-train", verbose=False)

    clips = read_manifest(data / "icsi-en-train")
    assert [clip.id for clip in clips] == [
        "icsi-en-train:Dtr004.fe068.0000",
        "icsi-en-train:Dtr004.fe068.0003",
        "icsi-en-train:Dtr004.fe068.0005",
        "icsi-en-train:Dtr004.fe068.0007",
        "icsi-en-train:Dtr004.fe069.0008",
    ]
    texts = [clip.text for clip in clips]
    # The row nobody made out is gone, and the digit task between the last two
    # ends what would otherwise have been one clip.
    assert texts == [
        "We have littler heads so.",
        "This one is fine.",
        "And the last thing on this channel",
        "was not a digit task.",
        "The places are here.",
    ]


def test_icsi_refuses_a_segment_of_a_speaker_whose_microphone_is_not_named(dirs, monkeypatch):
    """A clip is sliced from the speaker's own microphone: without one, there is no clip."""
    _, cache = dirs
    unamplified = ICSI_MEETINGS["Btr001"].replace(
        'Participant="fe068">\n      We have littler heads so.',
        'Participant="me011">\n      We have littler heads so.',
    )
    fake_icsi_cache(cache, monkeypatch, meetings={"Btr001": unamplified})
    with pytest.raises(DownloadError, match="which the preamble does not give them"):
        icsi.read_chunks(cache / "icsi-en-train" / icsi.TRANSCRIPTS_NAME)


def test_icsi_refuses_an_archive_without_a_transcript(dirs, monkeypatch):
    _, cache = dirs
    directory = cache / "icsi-en-train"
    empty = _icsi_member(directory, "nothing", "<Meeting Session='X'/>")
    archive = make_zip(directory / icsi.TRANSCRIPTS_NAME, {"transcripts/other.txt": empty})
    monkeypatch.setattr(icsi, "TRANSCRIPTS_URL", str(archive))
    with pytest.raises(DownloadError, match="holds no transcripts"):
        icsi.read_chunks(archive)


# --- NOTSOFAR-1 -------------------------------------------------------------

# Two recorded meetings in the shape the corpus publishes its transcripts: one
# row per utterance, with the speaker, the close-talking recording of that
# speaker and the times of the utterance, and the annotation markup inside the
# text.
NOTSOFAR_TRANSCRIPTS = {
    "MTG_00001": [
        {
            "speaker_id": "Peter",
            "ct_wav_file_name": "close_talk/CT_21.wav",
            "start_time": 1.0,
            "end_time": 3.5,
            "text": "I think schools are about preparing children for <PName> Harvard </PName>.",
        },
        {
            "speaker_id": "Sophie",
            "ct_wav_file_name": "close_talk/CT_22.wav",
            "start_time": 4.0,
            "end_time": 5.0,
            "text": "Mm-hmm. <ST/>",
        },
        {
            "speaker_id": "Peter",
            "ct_wav_file_name": "close_talk/CT_21.wav",
            "start_time": 5.5,
            "end_time": 5.9,
            "text": "Yeah.",
        },
        {
            "speaker_id": "Sophie",
            "ct_wav_file_name": "close_talk/CT_22.wav",
            "start_time": 6.0,
            "end_time": 7.5,
            "text": "<FILLlaugh/>",
        },
        {
            "speaker_id": "Peter",
            "ct_wav_file_name": "close_talk/CT_21.wav",
            "start_time": 8.0,
            "end_time": 9.0,
            "text": "But we should ask.",
        },
    ],
    "MTG_00002": [
        {
            "speaker_id": "Bert",
            "ct_wav_file_name": "close_talk/CT_21.wav",
            "start_time": 0.5,
            "end_time": 2.5,
            "text": "Right, let's start with the agenda.",
        }
    ],
}


def notsofar_listing(directory: Path, transcripts: dict, split: str | None = None) -> list[dict]:
    """The files of the fixture as the Hub lists them, with their own digests.

    A recording is a large file the Hub keeps through its LFS store and names by
    a sha256; a transcript is a small one it keeps as a git object and names by
    the sha1 of that object. Both are the digests the real listing carries, so a
    download of the fixture is checked the way a download of the corpus is.
    """
    split = split or notsofar.DIRECTORY
    paths = [f"{split}/{notsofar.MEETINGS_DIRECTORY}"]
    for meeting in transcripts:
        paths.append(f"{split}/{notsofar.MEETINGS_DIRECTORY}/{meeting}")
        paths.append(f"{split}/{notsofar.MEETINGS_DIRECTORY}/{meeting}/{notsofar.TRANSCRIPT_NAME}")
        for recording in sorted(_notsofar_recordings(transcripts, meeting)):
            paths.append(f"{split}/{notsofar.MEETINGS_DIRECTORY}/{meeting}/{recording}")
    entries = []
    for path in paths:
        file = directory / path
        if file.is_dir():
            entries.append({"type": "directory", "path": path})
        elif path.endswith(".wav"):
            entries.append(
                {
                    "type": "file",
                    "path": path,
                    "size": file.stat().st_size,
                    "lfs": {"oid": sha256_file(file), "size": file.stat().st_size},
                }
            )
        else:
            entries.append({"type": "file", "path": path, "oid": git_blob_sha1_file(file)})
    return entries


def _notsofar_recordings(transcripts: dict, meeting: str) -> set[str]:
    return {row["ct_wav_file_name"] for row in transcripts[meeting]}


def fake_notsofar_release(
    cache: Path,
    monkeypatch,
    transcripts: dict | None = None,
    listing=None,
    seconds: float = 10.0,
    split: str | None = None,
) -> Path:
    """A cache that looks like the train split of the corpus, without the Hub.

    The files are the fixture's own and the listing is built from them, so the
    checksums the preparation verifies are the checksums of the files it copies:
    a fixture file whose digest does not match the listing is refused exactly as
    a corpus file that changed under the pinned revision would be. The URL of a
    file is that file, which is what makes the download a copy, and the listing
    is asked for through the same call the corpus's listing is.
    """
    transcripts = NOTSOFAR_TRANSCRIPTS if transcripts is None else transcripts
    split = split or notsofar.DIRECTORY
    directory = cache / "fixture"
    for meeting, rows in transcripts.items():
        transcript = directory / split / notsofar.MEETINGS_DIRECTORY / meeting
        transcript.mkdir(parents=True, exist_ok=True)
        (transcript / notsofar.TRANSCRIPT_NAME).write_text(json.dumps(rows), encoding="utf-8")
        for recording in _notsofar_recordings(transcripts, meeting):
            write_tone(transcript / recording, seconds=seconds)
    entries = notsofar_listing(directory, transcripts, split)
    if listing is not None:
        entries = listing(entries)
    monkeypatch.setattr(
        notsofar, "huggingface_entries", lambda repo, revision, path, **options: entries
    )
    monkeypatch.setattr(notsofar.Release, "url", lambda self, path: str(directory / path))
    return directory


def test_notsofar_reads_the_train_meetings_and_groups_by_participant(dirs, monkeypatch):
    data, cache = dirs
    fake_notsofar_release(cache, monkeypatch)

    count = notsofar.fetch_notsofar_en_train(
        "notsofar-en-train", cache, data / "notsofar-en-train", verbose=False
    )
    # Two answers of Peter, one of Sophie and one of Bert; the utterance too
    # short to be a clip and the one that was nothing but a laugh are not clips.
    assert count == 4

    clips = {clip.id: clip for clip in read_manifest(data / "notsofar-en-train")}
    assert set(clips) == {
        "notsofar-en-train:MTG_00001.Peter.0000",
        "notsofar-en-train:MTG_00001.Sophie.0001",
        "notsofar-en-train:MTG_00001.Peter.0004",
        "notsofar-en-train:MTG_00002.Bert.0000",
    }
    first = clips["notsofar-en-train:MTG_00001.Peter.0000"]
    assert first.text == "I think schools are about preparing children for Harvard."
    assert first.text_has_casing_punct is True
    # The clip is the part of the recording the transcript gives it.
    assert first.duration_s == pytest.approx(2.5, abs=0.01)
    assert first.group == "notsofar-en-train:Peter"
    assert first.group_kind == "speaker"
    assert first.source_split == "240825.1_train"
    assert first.language == "en"
    assert first.license == "CC-BY-4.0"
    assert clips["notsofar-en-train:MTG_00001.Sophie.0001"].text == "Mm-hmm."
    assert clips["notsofar-en-train:MTG_00001.Peter.0004"].text == "But we should ask."
    # The same participant of two meetings is one group, which is what the
    # aliases of the corpus are for.
    assert {clip.group for clip in clips.values()} == {
        "notsofar-en-train:Peter",
        "notsofar-en-train:Sophie",
        "notsofar-en-train:Bert",
    }


def test_notsofar_leaves_out_the_rows_the_annotators_could_not_write_down(dirs, monkeypatch):
    """A word nobody caught, and a stretch they had a problem with, are not references."""
    data, cache = dirs
    transcripts = {
        "MTG_00001": [
            {
                "speaker_id": "Peter",
                "ct_wav_file_name": "close_talk/CT_21.wav",
                "start_time": 1.0,
                "end_time": 2.0,
                "text": "But <UNKNOWN/> we should ask.",
            },
            {
                "speaker_id": "Peter",
                "ct_wav_file_name": "close_talk/CT_21.wav",
                "start_time": 2.5,
                "end_time": 3.5,
                "text": "This one is written down.",
            },
            {
                "speaker_id": "Peter",
                "ct_wav_file_name": "close_talk/CT_21.wav",
                "start_time": 4.0,
                "end_time": 5.0,
                "text": "And <ISSUE/> this one is not.",
            },
            {
                "speaker_id": "Peter",
                "ct_wav_file_name": "close_talk/CT_21.wav",
                "start_time": 5.5,
                "end_time": 6.5,
                "text": "So is this.",
            },
        ]
    }
    fake_notsofar_release(cache, monkeypatch, transcripts=transcripts)

    notsofar.fetch_notsofar_en_train(
        "notsofar-en-train", cache, data / "notsofar-en-train", verbose=False
    )

    clips = read_manifest(data / "notsofar-en-train")
    # The rows with a mark in them break the run of one speaker's speech, so the
    # two that are readable are two clips and not one.
    assert [clip.id for clip in clips] == [
        "notsofar-en-train:MTG_00001.Peter.0001",
        "notsofar-en-train:MTG_00001.Peter.0003",
    ]
    assert [clip.text for clip in clips] == ["This one is written down.", "So is this."]


def test_notsofar_joins_the_utterances_of_one_participant_into_a_clip(dirs, monkeypatch):
    """The corpus cuts at every pause; a clip is what a participant said in one go."""
    data, cache = dirs
    transcripts = {
        "MTG_00001": [
            {
                "speaker_id": "Peter",
                "ct_wav_file_name": "close_talk/CT_21.wav",
                "start_time": 1.0,
                "end_time": 2.0,
                "text": "I think schools are about",
            },
            {
                "speaker_id": "Peter",
                "ct_wav_file_name": "close_talk/CT_21.wav",
                "start_time": 2.3,
                "end_time": 4.0,
                "text": "preparing children for Harvard.",
            },
            {
                "speaker_id": "Peter",
                "ct_wav_file_name": "close_talk/CT_21.wav",
                "start_time": 4.4,
                "end_time": 5.0,
                "text": "And the second one.",
            },
        ]
    }
    fake_notsofar_release(cache, monkeypatch, transcripts=transcripts)

    notsofar.fetch_notsofar_en_train(
        "notsofar-en-train", cache, data / "notsofar-en-train", verbose=False
    )

    clips = read_manifest(data / "notsofar-en-train")
    # The first two rows are one sentence the annotators cut at a pause, so they
    # are one clip; the third follows a sentence they closed and stands alone.
    assert [clip.id for clip in clips] == [
        "notsofar-en-train:MTG_00001.Peter.0000",
        "notsofar-en-train:MTG_00001.Peter.0002",
    ]
    assert clips[0].text == "I think schools are about preparing children for Harvard."
    assert clips[0].duration_s == pytest.approx(3.0, abs=0.01)
    assert clips[1].text == "And the second one."


def test_notsofar_cleans_the_annotation_markup_but_keeps_the_words():
    """A pause is not a word and a marked name is: the tags go, the words stay."""
    assert notsofar.clean_text("Mm-hmm. <ST/>") == "Mm-hmm."
    assert notsofar.clean_text("<PName> Peter </PName>") == "Peter"
    assert notsofar.clean_text("But <UNKNOWN/> we should ask.") == "But we should ask."
    assert notsofar.clean_text("<FILLlaugh/>") == ""
    assert notsofar.clean_text("So   <ST/>  <FILL/>  then.") == "So then."
    # Nothing is put in place of a tag, so a tag next to a word does not move it.
    assert notsofar.clean_text("the <PName> Rock </PName> said.") == "the Rock said."


def test_notsofar_stops_at_the_share_of_one_group(dirs, monkeypatch):
    """No more than its share of the cap comes from one participant."""
    data, cache = dirs
    fake_notsofar_release(cache, monkeypatch)
    monkeypatch.setattr(notsofar, "CLIP_CAP_PER_GROUP", 1)

    count = notsofar.fetch_notsofar_en_train(
        "notsofar-en-train", cache, data / "notsofar-en-train", verbose=False
    )

    clips = read_manifest(data / "notsofar-en-train")
    assert count == 3
    assert [clip.id for clip in clips if clip.group.endswith(":Peter")] == [
        "notsofar-en-train:MTG_00001.Peter.0000"
    ]


def test_notsofar_refuses_a_recording_the_hub_lists_no_checksum_for(dirs, monkeypatch):
    """A file nobody can pin is not downloaded: the pinned revision is the point."""
    _, cache = dirs

    def without_checksums(entries):
        return [
            {key: value for key, value in entry.items() if key != "lfs"}
            if entry["path"].endswith("CT_21.wav")
            else entry
            for entry in entries
        ]

    fake_notsofar_release(cache, monkeypatch, listing=without_checksums)
    with pytest.raises(DownloadError, match="lists no checksum"):
        notsofar.fetch_notsofar_en_train("notsofar-en-train", cache, cache / "out", verbose=False)


def test_notsofar_refuses_a_recording_of_another_meeting(dirs, monkeypatch):
    """A transcript names the microphone of its own meeting, not of the room next door."""
    _, cache = dirs
    elsewhere = json.loads(json.dumps(NOTSOFAR_TRANSCRIPTS))
    elsewhere["MTG_00001"][0]["ct_wav_file_name"] = "../MTG_00002/close_talk/CT_21.wav"
    fake_notsofar_release(cache, monkeypatch, transcripts=elsewhere)

    with pytest.raises(DownloadError, match="not a close_talk/<name> recording"):
        notsofar.fetch_notsofar_en_train("notsofar-en-train", cache, cache / "out", verbose=False)


def test_notsofar_refuses_a_clip_that_reaches_past_its_recording(dirs, monkeypatch):
    """Times that do not fit the recording are not times of that recording."""
    _, cache = dirs
    too_late = json.loads(json.dumps(NOTSOFAR_TRANSCRIPTS))
    too_late["MTG_00001"][0]["end_time"] = 6.5
    fake_notsofar_release(cache, monkeypatch, transcripts=too_late, seconds=5.0)

    with pytest.raises(DownloadError, match="the annotation is not of this recording"):
        notsofar.fetch_notsofar_en_train("notsofar-en-train", cache, cache / "out", verbose=False)


def test_notsofar_asks_the_hub_for_the_revision_it_pins(dirs):
    """The revision and the split are in every URL, so neither can drift silently."""
    _, cache = dirs
    release = notsofar.Release("notsofar-en-train", cache)
    path = (
        f"{notsofar.DIRECTORY}/{notsofar.MEETINGS_DIRECTORY}/MTG_00001/{notsofar.TRANSCRIPT_NAME}"
    )

    url = release.url(path)

    assert notsofar.REVISION in url
    assert path in url
    # The simulated corpus of the same release is another tree of the repository,
    # and the dev and evaluation subsets are other directories of this one.
    for other in ("css-datasets", "dev_set", "eval_set"):
        with pytest.raises(DownloadError, match="is not a file of"):
            release.url(other)


# The evaluation split of the fixture: the same shape, one alias (Peter) a test
# may put in the prepared train manifest to check the overlap drops.
NOTSOFAR_EVAL_TRANSCRIPTS = {
    "MTG_00010": [
        {
            "speaker_id": "Peter",
            "ct_wav_file_name": "close_talk/CT_21.wav",
            "start_time": 1.0,
            "end_time": 3.5,
            "text": "We should ship it.",
        },
        {
            "speaker_id": "Sophie",
            "ct_wav_file_name": "close_talk/CT_22.wav",
            "start_time": 4.0,
            "end_time": 5.0,
            "text": "I agree.",
        },
        {
            "speaker_id": "Peter",
            "ct_wav_file_name": "close_talk/CT_21.wav",
            "start_time": 5.5,
            "end_time": 6.0,
            "text": "Yeah.",
        },
        {
            "speaker_id": "Peter",
            "ct_wav_file_name": "close_talk/CT_21.wav",
            "start_time": 7.0,
            "end_time": 9.0,
            "text": "And we <PName> should </PName> log it.",
        },
    ],
    "MTG_00011": [
        {
            "speaker_id": "Peter",
            "ct_wav_file_name": "close_talk/CT_21.wav",
            "start_time": 0.5,
            "end_time": 2.5,
            "text": "Same here.",
        },
        {
            "speaker_id": "Anna",
            "ct_wav_file_name": "close_talk/CT_23.wav",
            "start_time": 3.0,
            "end_time": 5.0,
            "text": "Noted.",
        },
    ],
}


def test_notsofar_eval_reads_the_eval_split_and_drops_a_train_alias(dirs, monkeypatch):
    """The evaluation split is prepared in full; a train alias in it is dropped."""
    data, cache = dirs
    fake_notsofar_release(
        cache, monkeypatch, transcripts=NOTSOFAR_EVAL_TRANSCRIPTS, split=notsofar.EVAL_DIRECTORY
    )
    monkeypatch.setattr(notsofar, "EVAL_EXPECTED_CLIPS", 2, raising=True)
    write_manifest(
        data / "notsofar-en-train",
        [make_clip("notsofar-en-train:Peter", group="notsofar-en-train:Peter")],
    )

    count = notsofar.fetch_notsofar_en_eval(
        "notsofar-en-eval", cache, data / "notsofar-en-eval", verbose=False
    )
    clips = {clip.id: clip for clip in read_manifest(data / "notsofar-en-eval")}
    assert count == 2
    assert set(clips) == {
        "notsofar-en-eval:MTG_00010.Sophie.0001",
        "notsofar-en-eval:MTG_00011.Anna.0001",
    }
    sophie = clips["notsofar-en-eval:MTG_00010.Sophie.0001"]
    assert sophie.text == "I agree."
    assert sophie.text_has_casing_punct is True
    assert sophie.group == "notsofar-en-eval:Sophie"
    assert sophie.group_kind == "speaker"
    assert sophie.source_split == notsofar.EVAL_SPLIT
    assert sophie.license == "CC-BY-4.0"


def test_notsofar_eval_keeps_everyone_without_a_prepared_train_set(dirs, monkeypatch):
    """Without the prepared train manifest nothing is excluded."""
    data, cache = dirs
    fake_notsofar_release(
        cache, monkeypatch, transcripts=NOTSOFAR_EVAL_TRANSCRIPTS, split=notsofar.EVAL_DIRECTORY
    )
    monkeypatch.setattr(notsofar, "EVAL_EXPECTED_CLIPS", 6, raising=True)

    count = notsofar.fetch_notsofar_en_eval(
        "notsofar-en-eval", cache, data / "notsofar-en-eval", verbose=False
    )
    assert count == 6
    assert "notsofar-en-eval:MTG_00010.Peter.0000" in {
        clip.id for clip in read_manifest(data / "notsofar-en-eval")
    }


def test_notsofar_eval_keeps_the_words_inside_the_markup():
    """The PName tag goes, the word it marks stays, and the padding is joined."""
    row = NOTSOFAR_EVAL_TRANSCRIPTS["MTG_00010"][3]
    assert notsofar.clean_text(row["text"]) == "And we should log it."


def test_notsofar_train_leaves_out_the_eval_aliases(dirs, monkeypatch):
    """The train fetcher takes its exclusions from the prepared eval manifest."""
    data, cache = dirs
    fake_notsofar_release(cache, monkeypatch)
    write_manifest(
        data / "notsofar-en-eval",
        [make_clip("notsofar-en-eval:Peter", group="notsofar-en-eval:Peter")],
    )

    count = notsofar.fetch_notsofar_en_train(
        "notsofar-en-train", cache, data / "notsofar-en-train", verbose=False
    )
    # Peter's two clips are out; the rest of the fixture stands.
    assert count == 2
    assert {clip.id for clip in read_manifest(data / "notsofar-en-train")} == {
        "notsofar-en-train:MTG_00001.Sophie.0001",
        "notsofar-en-train:MTG_00002.Bert.0000",
    }
