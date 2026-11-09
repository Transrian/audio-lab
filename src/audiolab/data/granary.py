"""Granary French (``granary-fr-train``): YouTube speech with pseudo-labels.

Granary is a corpus of pseudo-labelled web speech: each row is a passage of a
YouTube video with the text a large model heard in it. The audio is CC-BY-3.0
and the labels CC-BY-4.0. ESPnet publishes the French part under ``data/fr000``
(a second pass of the corpus lives in ``fr100`` to ``fr103``) as hundreds of
small Parquet shards of about seventy rows each, with the encoded audio inside
them.

There is no train and test split and no speaker identity: the corpus calls its
split ``asr_only``, and a row belongs to the video it was cut from, which is the
group here. Shards are taken in the Hub's own (numeric) order until the cap is
reached, which means a preparation downloads a few gigabytes of the corpus and
leaves the rest alone.

The text is the label's own, with casing and punctuation, but it is model output
rather than a human transcript: a row may be a fragment rather than a sentence
(``une définition``), and the labels were not verified. That is part of what the
screening measures.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError, download, huggingface_sha256_map
from audiolab.data.parquet import audio_of, rows
from audiolab.data.train import CLIP_CAP_PER_GROUP, TRAIN_CLIP_CAP, Take

REPO = "espnet/yodas-granary"
# Pinned so the clips a preparation takes cannot change under it.
REVISION = "969944574ea3f37890beaf67ea651e160cfaf043"
REPO_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
# The audio and the labels carry different licenses; both allow reuse and
# modification with attribution, so the source may train a published model.
LICENSE = "audio CC-BY-3.0, labels CC-BY-4.0"
LANGUAGE = "fr"
# The corpus's only split for speech recognition, and the French directory of it.
SPLIT = "asr_only"
LANGUAGE_DIR = "fr000"
DIRECTORY = f"data/{LANGUAGE_DIR}/{SPLIT}"
COLUMNS = ["audio", "utt_id", "text", "original_audio_id"]

# Every video of a row is a YouTube video, so a group is a recording: the rows of
# one video share a speaker, a microphone and a topic, and their errors are not
# independent. The corpus publishes no speaker identity to group by instead.
GROUP_KIND = "recording"


def fetch_granary_fr_train(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare the French pseudo-labelled speech up to the cap, returning the clip count."""
    cache = cache_dir / name
    shards = _shards()
    builder = ManifestBuilder(dataset_dir, verbose=verbose)
    take = Take(
        name=name,
        builder=builder,
        language=LANGUAGE,
        source_split=SPLIT,
        license=LICENSE,
        text_has_casing_punct=True,
        group_kind=GROUP_KIND,
        cap=TRAIN_CLIP_CAP,
        cap_per_group=CLIP_CAP_PER_GROUP,
    )

    for index, (shard, sha256) in enumerate(shards.items(), start=1):
        if take.full():
            break
        path = download(f"{REPO_URL}/{DIRECTORY}/{shard}", cache / shard, sha256=sha256)
        if verbose and index % 25 == 0:
            print(f"  ... shard {index}/{len(shards)}")
        for row in rows(path, COLUMNS):
            if take.full():
                break
            take.offer(
                row["utt_id"],
                partial(audio_of, row["audio"]),
                text=row["text"] or "",
                group=row["original_audio_id"],
            )
    if verbose:
        print(f"  {take.report()}")
    builder.write()
    return take.kept


def _shards() -> dict[str, str]:
    """Every shard of the French directory with its published sha256, in name order."""
    published = huggingface_sha256_map(REPO, REVISION, DIRECTORY)
    wanted = {name: sha for name, sha in published.items() if name.endswith(".parquet")}
    if not wanted:
        raise DownloadError(f"{REPO} lists no shard under {DIRECTORY}")
    unverified = [name for name, sha in wanted.items() if not sha]
    if unverified:
        raise DownloadError(f"the Hub lists no checksum for {len(unverified)} shards")
    return dict(sorted(wanted.items()))
