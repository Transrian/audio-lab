"""MLS French (``mls-fr``): read audiobook excerpts.

The corpus is stored as one tar archive of FLAC files per speaker plus three
text files (audio file names, segments, transcripts). Hugging Face also serves
auto-converted Parquet files, but those are a derived copy: the official files
are used here, so preparation matches the published corpus.
"""

from __future__ import annotations

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

REPO = "facebook/multilingual_librispeech"
# Pinned so the files a panel was built from cannot change under it.
REVISION = "2e83e61823b4c47dcbcb1980bb88601274127609"
REPO_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data"
LICENSE = "CC-BY-4.0"
LANGUAGE = "fr"
SPLIT = "test"
# The published size of the test split; tests patch this.
EXPECTED_CLIPS = 2426


def _source(file_name: str) -> str:
    return f"{REPO_URL}/mls_french/{SPLIT}/{file_name}"


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
                    audio_from_archive(handle, member),
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


def _published_hashes(cache: Path, filenames: list[str]) -> dict[str, str]:
    """sha256 of each archive, from the Hub listing at the pinned revision.

    The listing is only asked for when an archive still has to be downloaded: a
    file already in the cache was checked when it arrived.
    """
    if not any(not (cache / audio_name).is_file() for audio_name in filenames):
        return {}
    hashes = huggingface_sha256_map(REPO, REVISION, f"data/mls_french/{SPLIT}/audio")
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
