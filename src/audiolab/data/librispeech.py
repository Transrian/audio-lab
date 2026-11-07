"""LibriSpeech test-clean (``librispeech-en``): read audiobook excerpts.

Only the test-clean archive is downloaded: test-other is harder speech and is
left for a later phase. The archive holds FLAC files named
``speaker/chapter/file.flac`` next to a ``.trans.txt`` per chapter, and a tar
stream cannot be read backwards, so the transcripts are read in a first pass
and the audio in a second one.
"""

from __future__ import annotations

from pathlib import Path

from audiolab.data.build import ManifestBuilder, require_clips
from audiolab.data.download import DownloadError, audio_from_archive, download, open_archive

ARCHIVE_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
ARCHIVE_NAME = "test-clean.tar.gz"
LICENSE = "CC-BY-4.0"
LANGUAGE = "en"
SPLIT = "test-clean"
# OpenSLR publishes this file without a checksum, so the sha256 saw when the
# panels were built is recorded here and checked on every fresh download.
PINNED_SHA256 = {
    "test-clean.tar.gz": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
}
# The published size of the archive; tests patch this to a small number.
EXPECTED_CLIPS = 2620


def fetch_librispeech_en(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare LibriSpeech test-clean, returning the number of clips written."""
    archive_path = download(
        ARCHIVE_URL,
        cache_dir / name / ARCHIVE_NAME,
        sha256=PINNED_SHA256.get(ARCHIVE_NAME, ""),
    )
    transcripts = _read_transcripts(name, archive_path)
    builder = ManifestBuilder(dataset_dir, verbose=verbose)

    with open_archive(archive_path) as handle:
        for member in handle:
            if member.name.endswith(".flac"):
                text = transcripts.get(Path(member.name).stem)
                if text is None:
                    raise DownloadError(f"{name}: {member.name} has no transcript in the archive")
                add_librispeech_clip(builder, name, handle, member, text)

    require_clips(name, len(builder.clips), EXPECTED_CLIPS)
    builder.write()
    return len(builder.clips)


def _read_transcripts(name: str, archive_path: Path) -> dict[str, str]:
    """First pass: every ``.trans.txt`` of the archive, keyed by utterance id."""
    transcripts: dict[str, str] = {}
    with open_archive(archive_path) as handle:
        for member in handle:
            if member.name.endswith(".trans.txt"):
                transcripts.update(_parse_transcripts(_read_member(name, handle, member)))
    if not transcripts:
        raise DownloadError(f"{name}: the archive holds no .trans.txt file")
    return transcripts


def _read_member(name: str, handle, member) -> bytes:
    data = handle.extractfile(member)
    if data is None:
        raise DownloadError(f"{name}: {member.name} is not a readable file")
    return data.read()


def add_librispeech_clip(builder: ManifestBuilder, name: str, handle, member, text: str) -> None:
    """Add one archive member; its path gives the speaker (``speaker/chapter/file``)."""
    parts = Path(member.name).parts
    if len(parts) < 3:
        raise DownloadError(f"{name}: unexpected path in the archive: {member.name}")
    builder.add(
        f"{name}:{Path(member.name).stem}",
        audio_from_archive(handle, member),
        language=LANGUAGE,
        text=text,
        text_has_casing_punct=False,
        group=f"{name}:{parts[-3]}",
        group_kind="speaker",
        source_split=SPLIT,
        license=LICENSE,
    )


def _parse_transcripts(data: bytes) -> dict[str, str]:
    """Read a ``.trans.txt`` file: one ``<utterance id> <text>`` per line."""
    lines = data.decode("utf-8").splitlines()
    return {line.split(" ", 1)[0]: line.split(" ", 1)[1] for line in lines if " " in line}
