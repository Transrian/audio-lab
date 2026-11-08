"""VoxPopuli French (``voxpopuli-fr-train``): European Parliament speech, train split.

VoxPopuli is European Parliament speech with human-edited transcripts, and it is
CC0. Its French train split is published as thirteen Parquet shards with the
audio inside them, so reading a row means downloading the shard that holds it:
shards are taken in the corpus's own order until the cap is reached, which is
why a preparation downloads a few gigabytes instead of the whole split.

The transcripts kept here are the corpus's ``raw_text``: cased and punctuated,
unlike the ``normalized_text`` the corpus also publishes. The base model this
work fine-tunes writes casing and punctuation, so a source that has none teaches
it to drop them; where a source is lowercase, its card says so.

The same speakers also recorded the ``en_accented`` subset that is an evaluation
set, so any speaker id found there is left out here (decision 0002). Knowing
those ids means reading two Parquet shards' metadata, and those are the shards
the evaluation preparation downloads, in its own cache directory: the two share
one download instead of fetching 4.7 GiB twice.
"""

from __future__ import annotations

from pathlib import Path

from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError, download, huggingface_sha256_map
from audiolab.data.parquet import audio_of, rows
from audiolab.data.train import CLIP_CAP_PER_GROUP, TRAIN_CLIP_CAP, Take

REPO = "facebook/voxpopuli"
# Pinned so the clips a preparation takes cannot change under it.
REVISION = "42f01879c780b4a2e90ec0b4f616c2ece526e4f1"
REPO_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
# The data is CC0; the repository also publishes code and models under
# CC-BY-NC, which this project does not use.
LICENSE = "CC0-1.0 (data only)"
LANGUAGE = "fr"
SPLIT = "train"
SPLIT_PATH = "fr"
SHARD_PREFIX = "train-"
COLUMNS = ["audio", "raw_text", "speaker_id", "is_gold_transcript"]

# The French-accented English evaluation subset: same corpus, same recordings,
# and a speaker id can appear in both languages.
ACCENTED_DATASET = "voxpopuli-en-french-accent"
ACCENTED_PATH = "en_accented"
ACCENTED_ACCENT = "en_fr"
# VoxPopuli publishes no speaker id for part of its rows: 12 % of the French
# train split carries the string "None". A row whose reader is unknown can
# neither be grouped by reader nor be checked against the evaluation sets, which
# is what a group is for, so such a row is left out and counted.
UNNAMED_SPEAKERS = frozenset({"", "none", "nan", "null"})
ACCENTED_COLUMNS = ["speaker_id", "accent"]


def fetch_voxpopuli_fr_train(
    name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool
) -> int:
    """Prepare the French train split up to the cap, returning the number of clips."""
    cache = cache_dir / name
    shards = _shards(SPLIT_PATH, SHARD_PREFIX)
    builder = ManifestBuilder(dataset_dir, verbose=verbose)
    take = Take(
        name=name,
        builder=builder,
        language=LANGUAGE,
        source_split=SPLIT,
        license=LICENSE,
        text_has_casing_punct=True,
        excluded=_accented_speakers(cache_dir),
        cap=TRAIN_CLIP_CAP,
        cap_per_group=CLIP_CAP_PER_GROUP,
    )

    not_gold = 0
    unnamed = 0
    for shard, sha256 in shards.items():
        if take.full():
            break
        path = download(f"{REPO_URL}/{SPLIT_PATH}/{shard}", cache / shard, sha256=sha256)
        if verbose:
            print(f"  ... reading {shard}")
        for row in rows(path, COLUMNS):
            if take.full():
                break
            if not row["is_gold_transcript"]:
                not_gold += 1
                continue
            if str(row["speaker_id"]).strip().lower() in UNNAMED_SPEAKERS:
                unnamed += 1
                continue
            # The recording is the clip; the speaker is the group, because the
            # segments of one speaker share a voice, an accent and a room.
            take.offer(
                Path(row["audio"]["path"]).stem,
                audio_of(row["audio"]),
                text=row["raw_text"] or "",
                group=row["speaker_id"],
            )
    if verbose:
        print(
            f"  {take.report()}; without a gold transcript: {not_gold}; "
            f"without a named speaker: {unnamed}"
        )
    builder.write()
    return take.kept


def _shards(path: str, prefix: str) -> dict[str, str]:
    """The shards of one Hub directory with their published sha256, in name order.

    A pinned revision makes the repository immutable, so the file names and the
    checksum the Hub stores for each of them describe the source exactly; the
    listing is read once and only the shards this preparation wants are kept.
    """
    published = huggingface_sha256_map(REPO, REVISION, path)
    wanted = {name: sha for name, sha in published.items() if name.startswith(prefix)}
    if not wanted:
        raise DownloadError(f"{REPO} lists no {prefix!r} shard under {path}")
    unverified = [name for name, sha in wanted.items() if not sha]
    if unverified:
        raise DownloadError(f"the Hub lists no checksum for {len(unverified)} shards")
    return dict(sorted(wanted.items()))


def _accented_speakers(cache_dir: Path) -> frozenset[str]:
    """The speaker ids of the ``en_accented`` evaluation subset.

    Its two test shards are downloaded into the cache directory of the
    evaluation dataset, whose preparation pins the same revision, and only the
    columns naming the speaker and the accent are decoded: the audio of those
    rows is never read.
    """
    cache = cache_dir / ACCENTED_DATASET
    speakers: set[str] = set()
    for shard, sha256 in _shards(ACCENTED_PATH, "test-").items():
        path = download(f"{REPO_URL}/{ACCENTED_PATH}/{shard}", cache / shard, sha256=sha256)
        for row in rows(path, ACCENTED_COLUMNS):
            if row["accent"] == ACCENTED_ACCENT:
                speakers.add(row["speaker_id"])
    if not speakers:
        raise DownloadError(f"{ACCENTED_DATASET} lists no {ACCENTED_ACCENT} speaker")
    return frozenset(speakers)
