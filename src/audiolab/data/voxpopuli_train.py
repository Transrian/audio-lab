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

The plenary sessions of the evaluation set are shared rooms and microphones,
not only voices: a train clip of one of those sittings would teach a model the
acoustics the evaluation set measures (decision 0014, option 2), so every clip
whose session is one of the set's is left out here. The sessions are a
committed pin, :data:`EVALUATION_SESSIONS`, not a re-read of the evaluation
manifest: the exclusion is the same on every machine, and an emptied or
broken pin is refused, so it cannot silently stop excluding.
"""

from __future__ import annotations

import json
import re
from functools import partial
from pathlib import Path

from audiolab.data.build import ManifestBuilder
from audiolab.data.datasets import base_dataset
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

# The non-native English evaluation subset: same corpus, same recordings, and
# a speaker id can appear in both languages.
NONNATIVE_DATASET = "voxpopuli-en-nonnative"
ACCENTED_PATH = "en_accented"
ACCENTED_ACCENT = "en_fr"
# VoxPopuli publishes no speaker id for part of its rows: 12 % of the French
# train split carries the string "None". A row whose reader is unknown can
# neither be grouped by reader nor be checked against the evaluation sets, which
# is what a group is for, so such a row is left out and counted.
UNNAMED_SPEAKERS = frozenset({"", "none", "nan", "null"})
ACCENTED_COLUMNS = ["speaker_id", "accent"]

# A session is the prefix of a VoxPopuli clip id before the language tag (for
# example 20090310-0900-PLENARY-7 in 20090310-0900-PLENARY-7-fr_20090310-11:10:07_2):
# the clips of one sitting share the room, the microphones and the background.
# The pattern is the one of experiments/leakage-audit/sessions.py, which
# defined the session for the overlap audit of the two sets.
SESSION_RE = re.compile(r"^([0-9]{8}-[0-9]{4}-.+?)-(en|fr|de|es|fi|it|nl)_")
# The shape of a session itself, once the language tag is off: the sitting
# name after its date and time.
SESSION_SHAPE = re.compile(r"^[0-9]{8}-[0-9]{4}-\S+$")
EVAL_SESSIONS_FILE = Path(__file__).parent / "voxpopuli_eval_sessions.json"


def session_of(clip_id: str) -> str:
    """The session of one VoxPopuli id: its prefix before the language tag.

    Accepts a raw source stem (``20090310-0900-PLENARY-7-fr_20090310-11:10:07_2``)
    and a prepared manifest id alike: a ``<dataset>:`` prefix is stripped before
    matching, as in experiments/leakage-audit/sessions.py, and the strip is
    attempted only when the id does not already start with a session. An id
    that holds no session before a language tag is a data error and is refused,
    not skipped: silently treating it as "no session" would exclude nothing,
    and a silent empty exclusion is the bug this check exists to prevent.
    """
    match = SESSION_RE.match(clip_id)
    if match is None and ":" in clip_id:
        match = SESSION_RE.match(clip_id.split(":", 1)[1])
    if match is None:
        raise ValueError(f"{clip_id!r} does not hold a session before a language tag")
    return match.group(1)


def evaluation_sessions(path: Path | None = None) -> frozenset[str]:
    """The plenary sessions of the evaluation set, from the committed pin.

    ``voxpopuli_eval_sessions.json`` holds the 307 sessions of the prepared
    ``voxpopuli-en-nonnative`` manifest, sorted. They were read from that
    manifest on 2026-11-15 with :func:`session_of`, and the evaluation set and
    its panels are frozen, so the list is fixed. A preparation reads the pin,
    not the evaluation manifest, so the exclusion does not depend on the
    machine's data. The pin is validated on load: an empty, unsorted,
    duplicated or malformed list is refused, so an emptied pin cannot silently
    disable the exclusion.
    """
    pin = path if path is not None else EVAL_SESSIONS_FILE
    sessions = json.loads(pin.read_text(encoding="utf-8"))
    if not sessions:
        raise ValueError(f"{pin.name} is empty: the evaluation sessions cannot be an empty pin")
    for session in sessions:
        if not isinstance(session, str) or SESSION_SHAPE.match(session) is None:
            raise ValueError(f"{pin.name} holds a malformed session: {session!r}")
    if sessions != sorted(sessions):
        raise ValueError(f"{pin.name} is not sorted")
    if len(set(sessions)) != len(sessions):
        raise ValueError(f"{pin.name} holds a session twice")
    return frozenset(sessions)


#: The pin, loaded once: a preparation made with another pin would exclude
#: other clips, so the constant is part of the recipe, not a machine state.
EVALUATION_SESSIONS = evaluation_sessions()


def fetch_voxpopuli_fr_train(
    name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool, clips: int | None = None
) -> int:
    """Prepare the French train split up to the cap, returning the number of clips.

    ``clips`` is the cap of a larger preparation (``voxpopuli-fr-train-40k``):
    the clips are recorded under ``name``, the directory they land in, and the
    shards are read from the source's own cache, so the larger preparation
    downloads nothing the usual one already read. Every preparation of this
    source, 12 000 and ``-Nk`` alike, leaves out the evaluation set's sessions
    (:data:`EVALUATION_SESSIONS`) as well as its speakers.
    """
    cap = clips if clips is not None else TRAIN_CLIP_CAP
    cache = cache_dir / base_dataset(name)
    shards = _shards(SPLIT_PATH, SHARD_PREFIX)
    with ManifestBuilder(dataset_dir, verbose=verbose, total=cap) as builder:
        take = Take(
            name=name,
            builder=builder,
            language=LANGUAGE,
            source_split=SPLIT,
            license=LICENSE,
            text_has_casing_punct=True,
            excluded=_accented_speakers(cache_dir),
            cap=cap,
            cap_per_group=CLIP_CAP_PER_GROUP,
        )

        not_gold = 0
        unnamed = 0
        shared = 0
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
                clip_id = Path(row["audio"]["path"]).stem
                # The sitting of a clip is a room and a microphone, not only a
                # voice: the evaluation set was recorded in its 307 sessions,
                # and a train set must not hear their acoustics (decision 0014).
                if session_of(clip_id) in EVALUATION_SESSIONS:
                    shared += 1
                    continue
                # The recording is the clip; the speaker is the group, because the
                # segments of one speaker share a voice, an accent and a room.
                take.offer(
                    clip_id,
                    partial(audio_of, row["audio"]),
                    text=row["raw_text"] or "",
                    group=row["speaker_id"],
                )
        if verbose:
            print(
                f"  {take.report()}; without a gold transcript: {not_gold}; "
                f"without a named speaker: {unnamed}; "
                f"in the evaluation set's sessions: {shared}"
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
    cache = cache_dir / NONNATIVE_DATASET
    speakers: set[str] = set()
    for shard, sha256 in _shards(ACCENTED_PATH, "test-").items():
        path = download(f"{REPO_URL}/{ACCENTED_PATH}/{shard}", cache / shard, sha256=sha256)
        for row in rows(path, ACCENTED_COLUMNS):
            if row["accent"] == ACCENTED_ACCENT:
                speakers.add(row["speaker_id"])
    if not speakers:
        raise DownloadError(f"{NONNATIVE_DATASET} lists no {ACCENTED_ACCENT} speaker")
    return frozenset(speakers)
