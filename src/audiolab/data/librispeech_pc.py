"""LibriSpeech-PC (``librispeech-pc-en-train``): LibriSpeech audio with cased, punctuated text.

LibriSpeech-PC publishes text only: one JSON manifest per LibriSpeech split,
with the cased and punctuated transcription of each utterance. The audio is
LibriSpeech's own, from OpenSLR SLR12, and the ids match, so this source is the
training source of `librispeech-en-train` — the same archive, the same order,
the same evaluation speakers left out — with the uppercase, unpunctuated
transcripts swapped for these. A preparation reads that whole archive for the
audio and one 25 MiB archive of manifests for the text.

Some utterances were dropped when the text was restored (2 498 of the 28 539
``train-clean-100`` utterances, 8.8 %, have no line in the manifest). They are
left out rather than labelled with the original uppercase text.

The restored text keeps a chapter title in capitals at the start of the first
sentence of an utterance (``THE PEOPLE MARCH He became aware ...``). A run of
three or more words in capitals marks such an utterance — punctuation is stripped
from a word first, so ``MARCH,`` and ``MIXER'S`` count — and 144 of the split's
26 041 utterances match it. The preparation leaves those out, because a heading
in capitals is not the text a dictation model should write.

The audio is the babble of the noisy evaluation sets: they mix four speakers
taken from ``librispeech-en-train``, and this source relabels the very audio. A
training set must not train on what a frozen panel mixes under it, so a
preparation leaves out the babble utterances of the prepared noisy sets — the
committed corpus ids of ``babble_utterances.json`` (one list per noisy clip
set, beside ``noisy.py``) — and a training run containing a babble utterance is
refused (the training-time refusal is tracked as [C-01](../../docs/backlog.md)).
The base model very likely trained on this audio with this text (NeMo's
ASR Set 3.0 lists LibriSpeech and keeps punctuation and casing), so this source
guards English in a mix rather than bringing gains.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path

from audiolab.data import noisy
from audiolab.data.build import ManifestBuilder
from audiolab.data.datasets import base_dataset
from audiolab.data.download import DownloadError, download, open_archive
from audiolab.data.librispeech import PINNED_SHA256 as LIBRISPEECH_SHA256
from audiolab.data.librispeech import (
    TRAIN_ARCHIVE_NAME,
    TRAIN_ARCHIVE_URL,
    prepare_train_audio,
)

MANIFESTS_URL = "https://www.openslr.org/resources/145/manifests.tar.gz"
MANIFESTS_NAME = "manifests.tar.gz"
# The manifest of the split this source reads, inside the archive.
MANIFEST_MEMBER = "train-clean-100.json"
# OpenSLR publishes no checksum for the manifests; this is the sha256 of the
# archive the source was prepared from, checked on every fresh download.
PINNED_SHA256 = {
    "manifests.tar.gz": "96d4eae2222b29b66437a21959252419bcd4762e5042e71e023790171054d1c0",
}  # fmt: skip
LICENSE = "CC-BY-4.0"
# The audio archive of librispeech-en-train is the same file, so its pinned
# checksum is reused rather than recorded twice.
PINNED_AUDIO_SHA256 = LIBRISPEECH_SHA256[TRAIN_ARCHIVE_NAME]
# A run of this many words written in capitals is a heading, not speech.
TITLE_RUN = 3
_CAPITAL_WORD = re.compile(r"[A-ZÀ-Þ]+")
_NOT_A_LETTER = re.compile(r"[^A-Za-zÀ-Þà-ÿ]")
# Why a babble utterance is left out of a preparation.
BABBLE_REASON = "a babble utterance of the noisy evaluation sets"


def babble_exclusions(
    utterance_ids: Iterable[str], babble: Mapping[str, list[str]]
) -> dict[str, str]:
    """The corpus utterance ids a LibriSpeech-PC preparation leaves out, and why.

    The pure rule: an utterance is left out when its id is in the committed
    babble list (one list per noisy clip set) — the audio of that utterance is
    mixed into a frozen noisy panel as babble, and a training set must not
    train on it. The French pool's ids never match a LibriSpeech id, so the one
    rule covers both pools. ``utterance_ids`` are corpus ids (a manifest id
    with its dataset prefix off) and the function reads nothing but its
    arguments, so a fetcher and the measurement of what a prepared set would
    hold use the same rule.
    """
    blocked = set().union(*babble.values()) if babble else set()
    return {
        utterance: BABBLE_REASON
        for utterance in dict.fromkeys(utterance_ids)
        if utterance in blocked
    }


def fetch_librispeech_pc_en_train(
    name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool, clips: int | None = None
) -> int:
    """Prepare LibriSpeech-PC over LibriSpeech train-clean-100, up to the cap.

    ``clips`` is the cap of a larger preparation (``librispeech-pc-en-train-40k``):
    the clips are recorded under ``name``, the directory they land in, and the
    audio and the manifests are read from the source's own cache, so the larger
    preparation downloads nothing the usual one already read.

    Every preparation, 12 000 and larger alike, leaves out the babble
    utterances of the prepared noisy sets (``noisy.load_babble_utterances``):
    this source relabels the very audio they mix, and a broken or emptied list
    is refused, so the exclusion cannot silently stop.
    """
    manifests = download(
        MANIFESTS_URL,
        cache_dir / base_dataset(name) / MANIFESTS_NAME,
        sha256=PINNED_SHA256.get(MANIFESTS_NAME, ""),
    )
    # The audio comes from the cache entry of librispeech-en-train, so the two
    # sources share one 6 GiB download instead of fetching the archive twice.
    archive = download(
        TRAIN_ARCHIVE_URL,
        cache_dir / "librispeech-en-train" / TRAIN_ARCHIVE_NAME,
        sha256=PINNED_AUDIO_SHA256,
    )
    texts = read_texts(name, manifests)
    excluded = babble_exclusions(texts, noisy.load_babble_utterances())
    with ManifestBuilder(
        dataset_dir, verbose=verbose, total=clips if clips is not None else None
    ) as builder:
        dropped = {
            "without restored text": 0,
            "with a heading in capitals": 0,
            BABBLE_REASON: 0,
        }
        take = prepare_train_audio(
            name,
            archive,
            builder,
            text_for_clip=_restored(texts, dropped, excluded),
            text_has_casing_punct=True,
            cap=clips,
        )
        if verbose:
            left_out = ", ".join(f"{count} {what}" for what, count in dropped.items())
            print(f"  {take.report()}; left out by this source: {left_out}")
        builder.write()
    return take.kept


def read_texts(name: str, archive: Path) -> dict[str, str]:
    """Every utterance of the manifest, keyed by its LibriSpeech utterance id."""
    texts: dict[str, str] = {}
    with open_archive(archive) as handle:
        for member in handle:
            if member.name != MANIFEST_MEMBER:
                continue
            data = handle.extractfile(member)
            if data is None:
                raise DownloadError(f"{name}: {member.name} is not a readable file")
            texts.update(_parse_manifest(name, data.read()))
    if not texts:
        raise DownloadError(f"{name}: {MANIFESTS_NAME} holds no {MANIFEST_MEMBER}")
    return texts


def _restored(texts: dict[str, str], dropped: dict[str, int], excluded: dict[str, str]):
    """The ``text_for_clip`` callback of this source: label a clip or drop it."""

    def text_for_clip(utterance: str) -> str | None:
        reason = excluded.get(utterance)
        if reason is not None:
            dropped[reason] += 1
            return None
        text = texts.get(utterance)
        if text is None:
            dropped["without restored text"] += 1
            return None
        if title_run(text) >= TITLE_RUN:
            dropped["with a heading in capitals"] += 1
            return None
        return text

    return text_for_clip


def _parse_manifest(name: str, data: bytes) -> dict[str, str]:
    """Read the manifest: one JSON object per utterance, with its restored text."""
    texts = {}
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            path = row["audio_filepath"]
            text = row["text"]
        except (json.JSONDecodeError, KeyError, TypeError) as failure:
            raise DownloadError(
                f"{name}: unreadable line in {MANIFEST_MEMBER}: {failure}"
            ) from None
        # ``train-clean-100/103/1240/103-1240-0001.flac``: the file name without
        # its extension is the utterance id the archive uses too.
        texts[Path(path).stem] = text
    return texts


def title_run(text: str) -> int:
    """The longest run of words written in capitals, in a text.

    The rest of a word is dropped before it is looked at, so a heading that
    carries punctuation (``MARCH,``, ``MIXER'S``) counts like any other word in
    capitals; a word with a lowercase letter in it never does.
    """
    longest = current = 0
    for word in text.split():
        if _CAPITAL_WORD.fullmatch(_NOT_A_LETTER.sub("", word)):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
