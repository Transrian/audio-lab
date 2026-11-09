"""PxCorpus (``pxcorpus-fr-train``): spoken French drug prescriptions.

PxCorpus is a corpus of spoken drug prescriptions in French: 1 981 recordings
from 903 sessions and 55 participants (doctors, pharmacists and people with no
medical training), about four hours of speech. The transcripts were checked by
hand against the recordings. The corpus is published as one 376 MB zip holding
16 kHz mono WAV files, three parallel text files that describe them — the
recording path, the participant, and the transcript — and the semantic labels
of a spoken-language-understanding task, which a speech model does not use.

One recording is one utterance: the participant pressed a button, dictated a
prescription, and the corpus aligned that recording with its transcript in
``seq.in``. Nothing has to be cut, so a training clip is a whole recording,
and the recordings longer than 20 s are left out like any other row outside the
window. ``metadata.txt`` names 54 participant ids; the readme says the corpus
has 55 participants.

The text is lowercase and unpunctuated. Some lines mark a word the recording
button cut off, or a self-repair the annotator joined, with a slash inside the
word (``comp/primés``, ``/pone``); the slash is a mark of the transcription, not
something the speaker said, so the training text drops it and rejoins the word.
Its register is a spoken dialogue with an application, not read speech, and its
numbers are written as digits (``2,5 milligrammes``), which is what makes this
source interesting next to the others.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from audiolab.core.audio import load_audio
from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError, download
from audiolab.data.train import CLIP_CAP_PER_GROUP, TRAIN_CLIP_CAP, Take

ARCHIVE_URL = "https://zenodo.org/api/records/10080490/files/PxCorpus.zip/content"
ARCHIVE_NAME = "PxCorpus.zip"
# Zenodo publishes this file with an md5, not a sha256, so that is what the
# download is checked against.
PINNED_MD5 = {"PxCorpus.zip": "97247fc5278cf4e3ab147aab4fc72f14"}  # fmt: skip
LICENSE = "CC-BY-4.0"
LANGUAGE = "fr"
SPLIT = "recordings"
ROOT = "PxCorpus/PxSLU"
# The three files that describe the recordings, one line each, in one order.
PATHS = f"{ROOT}/paths.txt"
METADATA = f"{ROOT}/metadata.txt"
TRANSCRIPTS = f"{ROOT}/seq.in"
# What the corpus's paths.txt calls a recording. The files are 16 kHz mono
# already, but every source is re-encoded on preparation.
AUDIO_SUFFIX = ".wav"
# The mark the corpus puts inside a word it could not transcribe whole, as in
# "comp/primés" or "/pone": an annotation mark, not speech.
FRAGMENT_MARK = "/"
WHITESPACE = re.compile(r"\s+")


def fetch_pxcorpus_fr_train(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare the recordings of PxCorpus, returning the number of clips written."""
    archive = download(
        ARCHIVE_URL,
        cache_dir / name / ARCHIVE_NAME,
        md5=PINNED_MD5.get(ARCHIVE_NAME, ""),
    )
    rows = read_rows(name, archive)
    builder = ManifestBuilder(dataset_dir, verbose=verbose, total=TRAIN_CLIP_CAP)
    take = Take(
        name=name,
        builder=builder,
        language=LANGUAGE,
        source_split=SPLIT,
        license=LICENSE,
        text_has_casing_punct=False,
        cap=TRAIN_CLIP_CAP,
        cap_per_group=CLIP_CAP_PER_GROUP,
    )

    with zipfile.ZipFile(archive) as handle:
        for row in rows:
            if take.full():
                break
            take.offer(
                row.id,
                lambda member=row.member: load_audio(io.BytesIO(handle.read(member))),
                text=row.text,
                group=row.speaker,
            )
    if verbose:
        print(f"  {take.report()}")
    builder.write()
    return take.kept


@dataclass(frozen=True)
class Row:
    """One recording of the corpus: what to read, what was said, and by whom."""

    member: str
    text: str
    speaker: str

    @property
    def id(self) -> str:
        """The name of the recording inside the corpus, without its root."""
        return self.member.removeprefix(f"{ROOT}/recordings/").removesuffix(AUDIO_SUFFIX)


def read_rows(name: str, archive: Path) -> list[Row]:
    """Every recording of the corpus, in the order its own files list them.

    ``paths.txt``, ``metadata.txt`` and ``seq.in`` have one line per recording
    and are parallel: the recording, the participant who dictated it and its
    transcript are on the same line number of the three files. A preparation
    reads them from the archive, so an archive that does not hold them (or holds
    a different number of lines) fails here rather than preparing a corpus whose
    lines do not line up.
    """
    with zipfile.ZipFile(archive) as handle:
        paths = _lines(name, archive, handle, PATHS)
        metadata = _lines(name, archive, handle, METADATA)
        transcripts = _lines(name, archive, handle, TRANSCRIPTS)
    try:
        rows = [
            Row(f"{ROOT}/recordings/{path}", training_text(text), participant.partition(",")[0])
            for path, participant, text in zip(paths, metadata, transcripts, strict=True)
        ]
    except ValueError as error:
        raise DownloadError(
            f"{name}: the corpus's own files disagree: {len(paths)} paths, "
            f"{len(metadata)} participants, {len(transcripts)} transcripts ({error})"
        ) from None
    for row in rows:
        if not row.member.endswith(AUDIO_SUFFIX):
            raise DownloadError(f"{name}: {PATHS} names a file that is not audio: {row.member!r}")
    return rows


def training_text(text: str) -> str:
    """The text of one recording as a training label.

    The corpus writes a slash inside a word whose recording was cut off, or that
    the participant repaired mid-word; dropping it gives back the word the
    speaker said (``comp/primés`` is "comprimés").
    """
    return WHITESPACE.sub(" ", text.replace(FRAGMENT_MARK, "")).strip()


def _lines(name: str, archive: Path, handle: zipfile.ZipFile, member: str) -> list[str]:
    """The lines of one parallel text file of the corpus, decoded as UTF-8."""
    try:
        data = handle.read(member)
    except KeyError:
        raise DownloadError(f"{name}: {archive.name} holds no {member}") from None
    return data.decode("utf-8").splitlines()
