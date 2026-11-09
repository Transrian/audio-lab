"""VoxPopuli accented English (``voxpopuli-en-nonnative``): non-native English.

VoxPopuli is European Parliament speech. Besides the eighteen languages it also
holds ``en_accented``, English read by speakers whose first language is not
English, with the speaker's accent in a column. Only the rows whose ``accent``
is ``en_fr`` are prepared here, which the corpus labels French; matching those
speakers to the Parliament's member list showed that only about 26 of the 737
clips come from French-speaking members (largest: Latvia 208, Flanders 171,
Portugal 107), so the dataset is named and read as English spoken by non-native
speakers in the European Parliament. That is the fourth case of the owner's
target use (D-009). The transcripts are the corpus's own and every kept clip is
flagged as a gold transcript.

The subset has one split (``test``), published as two Parquet shards with the
audio inside them, so the shards are downloaded whole and filtered by accent;
the kept clips are a fifth of the shard's rows.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from audiolab.data.build import ManifestBuilder, require_clips
from audiolab.data.download import DownloadError, download
from audiolab.data.parquet import audio_of, rows

REPO = "facebook/voxpopuli"
# Pinned so the files a panel was built from cannot change under it.
REVISION = "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
REPO_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/en_accented"
# The data is CC0; the repository also publishes code and models under
# CC-BY-NC, which this project does not use.
LICENSE = "CC0-1.0 (data only)"
LANGUAGE = "en"
# The accent column labels the speaker's first language; the corpus calls this
# value French, and the label is kept whatever the speaker's real first language.
ACCENT = "en_fr"
SPLIT = "test"
SHARDS = ("test-00000-of-00002.parquet", "test-00001-of-00002.parquet")
COLUMNS = ["audio", "accent", "speaker_id", "raw_text", "is_gold_transcript"]
# sha256 of each shard at that revision, recorded from the download that built
# the panels; it is the value the Hub publishes for the file (mls.py reads the
# same value from the file listing instead, for a directory that changes shape).
PINNED_SHA256 = {
    "test-00000-of-00002.parquet":
        "202d71bde2680f30aa7ae1050a98cdf85972458f94e6493db2aa7b8dd8ff46ec",
    "test-00001-of-00002.parquet":
        "7d61b4c7a2e5c565a382543236442a7a8dd081c985e52dd1084905e4becc3901",
}  # fmt: skip
# The published size of the accent subset; tests patch this.
EXPECTED_CLIPS = 737


def fetch_voxpopuli_nonnative(
    name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool
) -> int:
    """Prepare the rows the corpus labels ``en_fr``, returning the number of clips written."""
    builder = ManifestBuilder(dataset_dir, verbose=verbose)
    for shard in SHARDS:
        path = download(
            f"{REPO_URL}/{shard}",
            cache_dir / name / shard,
            sha256=PINNED_SHA256.get(shard, ""),
        )
        if verbose:
            print(f"  ... reading {shard}")
        for row in rows(path, COLUMNS):
            if row["accent"] != ACCENT:
                continue
            if not row["is_gold_transcript"]:
                raise DownloadError(f"{name}: {row['audio']['path']} is not a gold transcript")
            # The recording is the clip; the speaker is the group, because one
            # corpus accent is spoken by several people and their errors are not
            # independent. The reference has no punctuation, so the formatted
            # view is not computed for these clips.
            builder.add(
                f"{name}:{Path(row['audio']['path']).stem}",
                partial(audio_of, row["audio"]),
                language=LANGUAGE,
                text=row["raw_text"],
                text_has_casing_punct=False,
                group=f"{name}:{row['speaker_id']}",
                group_kind="speaker",
                source_split=SPLIT,
                license=LICENSE,
            )
    require_clips(name, len(builder.clips), EXPECTED_CLIPS)
    builder.write()
    return len(builder.clips)
