"""MLS French (``mls-fr`` test, ``mls-fr-train``): read audiobook excerpts.

The corpus is stored as one tar archive of FLAC files per speaker plus three
text files (audio file names, segments, transcripts). Hugging Face also serves
auto-converted Parquet files, but those are a derived copy: the official files
are used here, so preparation matches the published corpus.

The train split is the same three text files and six hundred archives, of which
only the first archives are downloaded: a training source stops at the cap, so
it reads the published file list in order and takes archive after archive until
it has enough. The splits name their speakers, and MLS keeps them apart, so the
speakers scored on the test split cannot be trained on here; the preparation
also checks that against the prepared test manifest.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from audiolab.data.build import ManifestBuilder, require_clips
from audiolab.data.download import (
    DownloadError,
    audio_from_archive,
    download,
    huggingface_sha256_map,
    open_archive,
)
from audiolab.data.textfiles import read_records
from audiolab.data.train import CLIP_CAP_PER_GROUP, TRAIN_CLIP_CAP, Take, evaluation_groups

REPO = "facebook/multilingual_librispeech"
# Pinned so the files a panel was built from cannot change under it.
REVISION = "2e83e61823b4c47dcbcb1980bb88601274127609"
REPO_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data"
LICENSE = "CC-BY-4.0"
LANGUAGE = "fr"
SPLIT = "test"
TRAIN_SPLIT = "train"
# The published size of the test split; tests patch this.
EXPECTED_CLIPS = 2426
# The dataset prepared from the test split of this same corpus. MLS keeps the
# speakers of its splits apart, so nothing should be found in common here, but a
# speaker that is scored is never trained on (decision 0002).
EVALUATION_DATASET = "mls-fr"


def _source(file_name: str, split: str = SPLIT) -> str:
    return f"{REPO_URL}/mls_french/{split}/{file_name}"


def fetch_mls_fr(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare the MLS French test split, returning the number of clips written."""
    cache = cache_dir / name
    filenames = _read_names(download(_source("audio_filenames.txt"), cache / "audio_filenames.txt"))
    segments = read_records(download(_source("segments.txt"), cache / "segments.txt"), 3)
    transcripts = read_records(download(_source("transcripts.txt"), cache / "transcripts.txt"), 2)
    builder = ManifestBuilder(dataset_dir, verbose=verbose)

    pinned = _published_hashes(cache, filenames)
    for index, audio_name in enumerate(filenames, start=1):
        # The name list already includes the .tar.gz suffix.
        archive = download(
            _source(f"audio/{audio_name}"), cache / audio_name, sha256=pinned.get(audio_name, "")
        )
        with open_archive(archive) as handle:
            for member in handle:
                text = _clip_text(name, member.name, segments, transcripts)
                builder.add(
                    _clip_id(name, member.name),
                    partial(audio_from_archive, handle, member),
                    language=LANGUAGE,
                    text=text,
                    text_has_casing_punct=False,
                    group=f"{name}:{_speaker(name, member.name)}",
                    group_kind="speaker",
                    source_split=SPLIT,
                    license=LICENSE,
                )
        if verbose:
            print(f"  ... archive {index}/{len(filenames)}")

    require_clips(name, len(builder.clips), EXPECTED_CLIPS)
    builder.write()
    return len(builder.clips)


def fetch_mls_fr_train(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare the French train split up to the cap, returning the number of clips."""
    cache = cache_dir / name
    filenames = _read_names(
        download(_source("audio_filenames.txt", TRAIN_SPLIT), cache / "audio_filenames.txt")
    )
    segments = read_records(
        download(_source("segments.txt", TRAIN_SPLIT), cache / "segments.txt"), 3
    )
    transcripts = read_records(
        download(_source("transcripts.txt", TRAIN_SPLIT), cache / "transcripts.txt"), 2
    )
    builder = ManifestBuilder(dataset_dir, verbose=verbose, total=TRAIN_CLIP_CAP)
    take = Take(
        name=name,
        builder=builder,
        language=LANGUAGE,
        source_split=TRAIN_SPLIT,
        license=LICENSE,
        text_has_casing_punct=False,
        excluded=evaluation_groups(EVALUATION_DATASET),
        cap=TRAIN_CLIP_CAP,
        cap_per_group=CLIP_CAP_PER_GROUP,
    )

    pinned = _published_hashes(cache, filenames, TRAIN_SPLIT)
    skipped = 0
    for index, audio_name in enumerate(filenames, start=1):
        if take.full():
            break
        # The file list is ordered by book, so the first archives of one book
        # carry many readers; the archives of a reader that already holds its
        # share of the cap are not downloaded at all.
        if take.group_full(audio_name.split("_", 1)[0]):
            skipped += 1
            continue
        # The name list already includes the .tar.gz suffix.
        archive = download(
            _source(f"audio/{audio_name}", TRAIN_SPLIT),
            cache / audio_name,
            sha256=pinned.get(audio_name, ""),
        )
        with open_archive(archive) as handle:
            for member in handle:
                if take.full():
                    break
                take.offer(
                    Path(member.name).stem,
                    partial(audio_from_archive, handle, member),
                    text=_clip_text(name, member.name, segments, transcripts),
                    group=_speaker(name, member.name),
                )
        if verbose:
            print(f"  ... archive {index}/{len(filenames)}")

    if verbose:
        print(f"  {take.report()}; archives not read (reader already taken): {skipped}")
    builder.write()
    return take.kept


def _published_hashes(cache: Path, filenames: list[str], split: str = SPLIT) -> dict[str, str]:
    """sha256 of each archive, from the Hub listing at the pinned revision.

    The listing is only asked for when an archive still has to be downloaded: a
    file already in the cache was checked when it arrived.
    """
    if not any(not (cache / audio_name).is_file() for audio_name in filenames):
        return {}
    hashes = huggingface_sha256_map(REPO, REVISION, f"data/mls_french/{split}/audio")
    missing = [audio_name for audio_name in filenames if not hashes.get(audio_name)]
    if missing:
        raise DownloadError(
            f"the Hub lists no checksum for {len(missing)} archives (first {missing[0]})"
        )
    return hashes


def _read_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name]


def _clip_id(name: str, member_name: str) -> str:
    return f"{name}:{Path(member_name).stem}"


def _speaker(name: str, member_name: str) -> str:
    """The speaker of one utterance, read from its id.

    An MLS utterance id is ``<speaker>_<book>_<number>``. The speaker is what a
    bootstrap must resample: one reader's voice, accent and recording conditions
    fail together, and the segments file says nothing about speakers — its second
    field is the source mp3, and one reader is spread over many of those.
    """
    utterance_id = Path(member_name).stem
    speaker = utterance_id.split("_")[0]
    if not speaker or speaker == utterance_id:
        raise DownloadError(f"{name}: {utterance_id} does not look like <speaker>_<book>_<number>")
    return speaker


def _clip_text(name: str, member_name: str, segments: dict, transcripts: dict) -> str:
    """The transcript of one archive member, checked against the segment list."""
    transcript_id = Path(member_name).stem
    if transcript_id not in segments:
        raise DownloadError(f"{name}: {transcript_id} is in the archive but not in segments.txt")
    text = transcripts.get(transcript_id)
    if text is None:
        raise DownloadError(f"{name}: {transcript_id} has no transcript")
    return text
