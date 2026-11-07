"""FLEURS (``fleurs-fr``, ``fleurs-en``): read Wikipedia sentences.

The corpus is distributed as one TSV per split plus a tar archive of WAV
files, both under the dataset's ``data/<language>`` directory on Hugging Face.
"""

from __future__ import annotations

from pathlib import Path

from audiolab.data.build import ManifestBuilder, require_clips
from audiolab.data.download import audio_from_archive, download, open_archive
from audiolab.data.textfiles import read_fields

# Pinned so the files a panel was built from cannot change under it.
REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
REPO_URL = f"https://huggingface.co/datasets/google/fleurs/resolve/{REVISION}/data"
LICENSE = "CC-BY-4.0"
# sha256 of the two files of each language at that revision, recorded from the
# download that built the panels (the Hub publishes no checksum for these).
PINNED_SHA256 = {
    "fleurs-fr": {
        "test.tsv": "5d06d338b242e00786fcf12c4c92008b9f399d5a5c872c91dca90572e7869c0d",
        "test.tar.gz": "d23690e102f373554d1b544cd2ff1e76e4fedeb04953c0b72751a1b7c518cfdd",
    },
    "fleurs-en": {
        "test.tsv": "74c046239374deeb60fa63f258f907388093a32bcaa3140965f70ef05c79f7ca",
        "test.tar.gz": "d9c2e37b41aacd41bc283554a0a82b5476b36887049774ecb2819dcaaa55a356",
    },
}
# The published size of each language's test split; tests patch this.
EXPECTED_CLIPS = {"fleurs-fr": 676, "fleurs-en": 647}


def fetch_fleurs(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare one FLEURS language, returning the number of clips written."""
    language = {"fleurs-fr": "fr_fr", "fleurs-en": "en_us"}[name]
    source = f"{REPO_URL}/{language}"
    pinned = PINNED_SHA256.get(name, {})
    table_path = download(
        f"{source}/test.tsv", cache_dir / name / "test.tsv", sha256=pinned.get("test.tsv", "")
    )
    archive_path = download(
        f"{source}/audio/test.tar.gz",
        cache_dir / name / "test.tar.gz",
        sha256=pinned.get("test.tar.gz", ""),
    )
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

    require_clips(name, len(builder.clips), EXPECTED_CLIPS[name])
    builder.write()
    return len(builder.clips)
