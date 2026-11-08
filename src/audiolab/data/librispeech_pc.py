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
in capitals is not the text a dictation model should write. The base model very
likely trained on this audio with this text (NeMo's ASR Set 3.0 lists LibriSpeech
and keeps punctuation and casing), so this source guards English in a mix rather
than bringing gains.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from audiolab.data.build import ManifestBuilder
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


def fetch_librispeech_pc_en_train(
    name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool
) -> int:
    """Prepare LibriSpeech-PC over LibriSpeech train-clean-100, up to the cap."""
    manifests = download(
        MANIFESTS_URL,
        cache_dir / name / MANIFESTS_NAME,
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
    builder = ManifestBuilder(dataset_dir, verbose=verbose)
    dropped = {"without restored text": 0, "with a heading in capitals": 0}
    take = prepare_train_audio(
        name,
        archive,
        builder,
        text_for_clip=_restored(texts, dropped),
        text_has_casing_punct=True,
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


def _restored(texts: dict[str, str], dropped: dict[str, int]):
    """The ``text_for_clip`` callback of this source: label a clip or drop it."""

    def text_for_clip(utterance: str) -> str | None:
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
