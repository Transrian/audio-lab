"""LibriSpeech test-clean (``librispeech-en``): read audiobook excerpts.

Only the test-clean archive is downloaded: test-other is harder speech and is
left for a later phase. The archive holds FLAC files named
``speaker/chapter/file.flac`` next to a ``.trans.txt`` per chapter.
"""

from __future__ import annotations

from pathlib import Path

from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError, audio_from_archive, download, open_archive

ARCHIVE_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
LICENSE = "CC-BY-4.0"
LANGUAGE = "en"
SPLIT = "test-clean"
MINIMUM_CLIPS = 1  # the real test-clean archive holds 2620 clips


def fetch_librispeech_en(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare LibriSpeech test-clean, returning the number of clips written."""
    archive_path = download(ARCHIVE_URL, cache_dir / name / "test-clean.tar.gz")
    builder = ManifestBuilder(dataset_dir, verbose=verbose)
    transcripts: dict[str, str] = {}

    with open_archive(archive_path) as handle:
        for member in handle:
            if member.name.endswith(".trans.txt"):
                transcripts.update(_parse_transcripts(handle.extractfile(member).read()))
            elif member.name.endswith(".flac") and Path(member.name).stem in transcripts:
                add_librispeech_clip(
                    builder, name, handle, member, transcripts[Path(member.name).stem]
                )

    if len(builder.clips) < MINIMUM_CLIPS:
        raise DownloadError(f"{name}: only {len(builder.clips)} clips prepared")
    builder.write()
    return len(builder.clips)


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
