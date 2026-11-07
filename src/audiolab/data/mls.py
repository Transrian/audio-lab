"""MLS French (``mls-fr``): read audiobook excerpts.

The corpus is stored as one tar archive of FLAC files per speaker plus three
text files (audio file names, segments, transcripts). Hugging Face also serves
auto-converted Parquet files, but those are a derived copy: the official files
are used here, so preparation matches the published corpus.
"""

from __future__ import annotations

from pathlib import Path

from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError, audio_from_archive, download, open_archive
from audiolab.data.textfiles import read_records

REPO_URL = "https://huggingface.co/datasets/facebook/multilingual_librispeech/resolve/main/data"
LICENSE = "CC-BY-4.0"
LANGUAGE = "fr"
SPLIT = "test"
MINIMUM_CLIPS = 1


def _source(file_name: str) -> str:
    return f"{REPO_URL}/mls_french/{SPLIT}/{file_name}"


def fetch_mls_fr(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare the MLS French test split, returning the number of clips written."""
    cache = cache_dir / name
    filenames = _read_names(download(_source("audio_filenames.txt"), cache / "audio_filenames.txt"))
    segments = read_records(download(_source("segments.txt"), cache / "segments.txt"), 3)
    transcripts = read_records(download(_source("transcripts.txt"), cache / "transcripts.txt"), 2)
    builder = ManifestBuilder(dataset_dir, verbose=verbose)

    for index, audio_name in enumerate(filenames, start=1):
        archive = download(_source(f"audio/{audio_name}.tar.gz"), cache / f"{audio_name}.tar.gz")
        with open_archive(archive) as handle:
            for member in handle:
                text = _clip_text(name, member.name, segments, transcripts)
                builder.add(
                    _clip_id(name, member.name),
                    audio_from_archive(handle, member),
                    language=LANGUAGE,
                    text=text,
                    text_has_casing_punct=False,
                    group=f"{name}:{segments[Path(member.name).stem].split()[0]}",
                    group_kind="speaker",
                    source_split=SPLIT,
                    license=LICENSE,
                )
        if verbose:
            print(f"  ... archive {index}/{len(filenames)}")

    if len(builder.clips) < MINIMUM_CLIPS:
        raise DownloadError(f"{name}: only {len(builder.clips)} clips prepared")
    builder.write()
    return len(builder.clips)


def _read_names(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [name for name in names if name]


def _clip_id(name: str, member_name: str) -> str:
    return f"{name}:{Path(member_name).stem}"


def _clip_text(name: str, member_name: str, segments: dict, transcripts: dict) -> str:
    """The transcript of one archive member, checked against the segment list."""
    transcript_id = Path(member_name).stem
    if transcript_id not in segments:
        raise DownloadError(f"{name}: {transcript_id} is in the archive but not in segments.txt")
    text = transcripts.get(transcript_id)
    if text is None:
        raise DownloadError(f"{name}: {transcript_id} has no transcript")
    return text
