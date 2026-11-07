"""FLEURS (``fleurs-fr``, ``fleurs-en``): read Wikipedia sentences.

The corpus is distributed as one TSV per split plus a tar archive of WAV
files, both under the dataset's ``data/<language>`` directory on Hugging Face.
"""

from __future__ import annotations

from pathlib import Path

from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError, audio_from_archive, download, open_archive
from audiolab.data.textfiles import read_fields

REPO_URL = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
LICENSE = "CC-BY-4.0"
MINIMUM_CLIPS = 50


def fetch_fleurs(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare one FLEURS language, returning the number of clips written."""
    language = {"fleurs-fr": "fr_fr", "fleurs-en": "en_us"}[name]
    source = f"{REPO_URL}/{language}"
    table_path = download(f"{source}/test.tsv", cache_dir / name / "test.tsv")
    archive_path = download(f"{source}/audio/test.tar.gz", cache_dir / name / "test.tar.gz")
    # FLEURS test.tsv has no header line: sentence id, file name, raw text,
    # normalized text, character tokens, duration, gender. The raw text is the
    # reference. One sentence is read by several readers, so the sentence id
    # repeats and only the file name identifies a recording; the archive names
    # files without their extension, the table with it.
    rows = {
        Path(file_name).stem: row
        for file_name, row in read_fields(
            table_path, {0: "sentence", 2: "transcription"}, key_index=1, min_fields=3
        ).items()
    }
    builder = ManifestBuilder(dataset_dir, verbose=verbose)

    with open_archive(archive_path) as archive:
        for member in archive:
            row = rows.get(Path(member.name).stem)
            if row is None:
                continue
            # The recording is the clip; the sentence it reads is the group,
            # since FLEURS publishes no speaker identity.
            stem = Path(member.name).stem
            builder.add(
                f"{name}:{stem}",
                audio_from_archive(archive, member),
                language=language[:2],
                text=row["transcription"],
                text_has_casing_punct=True,
                group=f"{name}:{row['sentence']}",
                group_kind="sentence",
                source_split="test",
                license=LICENSE,
            )

    if len(builder.clips) < MINIMUM_CLIPS:
        raise DownloadError(
            f"{name}: the archive and the test.tsv do not match "
            f"({len(builder.clips)} clips, expected at least {MINIMUM_CLIPS})"
        )
    builder.write()
    return len(builder.clips)
