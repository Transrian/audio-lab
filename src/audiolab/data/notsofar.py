"""NOTSOFAR-1 (``notsofar-en-train``): real recorded meetings, one close mic per participant.

NOTSOFAR-1 is Microsoft's 2024 challenge on transcribing meetings that were
actually held, with several microphones in the room. Its release holds two
corpora: a simulated one of a thousand hours, and the **recorded meeting
dataset** — meetings of about six minutes, four to eight attendees around a
table in one of 30 conference rooms, transcribed by hand word by word, in which
35 speakers recur across the rooms (the challenge's own README counts 237
meetings, the dataset's page on Hugging Face 315). Only the recorded one is used
here: it is the speech humans really spoke, and the simulated one is a
simulation of it. The data and its annotations are CC-BY-4.0.

The release has three subsets and this source reads one of them, ``train_set``
of the revision ``240825.1_train``, on Hugging Face, without a login: 72
meetings. ``dev_set`` and ``eval_set`` are the development and evaluation
subsets of the challenge; they are not read here at all — not for training and
not for counting. ``gt_transcription.json``, the transcript of one meeting,
names the speaker, the close-talking recording, and the start and end time of
every utterance, with the corpus's annotation markup (``<ST/>``, ``<FILL/>``,
``<UNKNOWN/>``, ``<PName>``) inside the text.

Each participant wears a **close-talking** microphone and its recordings are
published (``close_talk/CT_<n>.wav``, 16 kHz mono), so a clip comes from the
microphone of the participant who said the words, like the AMI and ICSI
sources, and not from a room microphone that hears everyone. A participant is a
group by the alias the corpus gives them (``Peter``, ``Sophie``); the same
person carries the same alias through every room she sat in, which is what makes
the aliases the grouping.

The corpus's own segmentation is used as it is: one clip per transcribed
utterance. Nothing is cut differently, because the published start and end times
are what the annotators aligned.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from audiolab.core.audio import Audio, load_audio
from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError, download, huggingface_entries
from audiolab.data.meeting import Chunk, merged
from audiolab.data.train import (
    CLIP_CAP_PER_GROUP,
    TRAIN_CLIP_CAP,
    Take,
    plan_clips,
    write_clips,
)

REPO = "microsoft/NOTSOFAR"
# A pinned revision makes the repository immutable, which is what lets every
# file be checked against the sha256 the Hub itself publishes for it.
REVISION = "ba8fd0f034ce185fe4d24f47e53b4b8194795f07"
# The recorded meetings with human transcripts. The simulated 1 000-hour set of
# the same release (``css-datasets``) is not read, and neither are the dev and
# evaluation subsets.
SPLIT = "240825.1_train"
DIRECTORY = f"benchmark-datasets/train_set/{SPLIT}"
TRANSCRIPT_NAME = "gt_transcription.json"
CLOSE_TALK = "close_talk"
# The directory the split keeps its meetings under, which the corpus names MTG.
MEETINGS_DIRECTORY = "MTG"
LICENSE = "CC-BY-4.0"
LANGUAGE = "en"
# The corpus's markup: an empty tag where the annotators heard something that is
# not a word to transcribe — a pause, a filled pause, a laugh, a stretch nobody
# made out, a background noise — or a tag around a word they marked, such as a
# name. The tags are taken out of the text; the words inside them are not.
MARKUP = re.compile(r"<[^>]*>")
# The marks the annotators left where they could not write the words down: a
# word nobody caught, and a stretch they had a problem with. The audio is there
# but the reference is not, so a row holding one is left out rather than half
# transcribed.
UNWRITTEN = ("<UNKNOWN/>", "<ISSUE/>")
WHITESPACE = re.compile(r"\s+")
# The corpus pads a tag with spaces (``<PName> Peter </PName> .``), so taking the
# tag out can leave a space between a word and the punctuation the annotators
# attached to it. Punctuation is joined to the word before it, as the AMI source
# does, so the text reads the way the annotators wrote it.
SPACED_PUNCTUATION = re.compile(r"\s+([,.;:!?])")
# How many files are downloaded at the same time: the work waits for the
# network, not for the CPU.
WORKERS = 8


@dataclass(frozen=True)
class Meeting:
    """One recorded meeting: where it lives in the corpus, and what was said.

    The utterances come in the corpus's own order, with ``None`` where a row is
    markup and nothing else: the audio under such a row is speech the annotators
    did not write down, and the clip builder must not build a clip across it.
    """

    meeting: str
    # The meeting's own directory in the repository, which every file of the
    # meeting is found under.
    directory: str
    utterances: list[Chunk | None]


def fetch_notsofar_en_train(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare the recorded training meetings of NOTSOFAR-1 up to the cap."""
    release = Release(name, cache_dir)
    meetings = release.meetings()
    with ManifestBuilder(dataset_dir, verbose=verbose, total=TRAIN_CLIP_CAP) as builder:
        take = Take(
            name=name,
            builder=builder,
            language=LANGUAGE,
            source_split=SPLIT,
            license=LICENSE,
            text_has_casing_punct=True,
            cap=TRAIN_CLIP_CAP,
            cap_per_group=CLIP_CAP_PER_GROUP,
        )
        # Which rows become clips is decided on the transcripts, before any clip is
        # written: 12 speakers allow 2 400 clips at most, so a meeting has to be read
        # knowing what another meeting has already taken. A meeting is then read in
        # one piece — its recordings are downloaded at the same time — and only the
        # rows the plan kept are written, from the recording each of them is on.
        planned = plan_clips(
            take, [chunk for meeting in meetings for chunk in merged(meeting.utterances)]
        )
        for meeting in meetings:
            rows = [chunk for chunk in planned if chunk.meeting == meeting.meeting]
            if not rows:
                continue
            audio = release.recordings(meeting, rows)
            by_recording: dict[str, list[Chunk]] = {}
            for chunk in rows:
                by_recording.setdefault(chunk.recording, []).append(chunk)
            for recording, group in by_recording.items():
                write_clips(take, recording, audio[recording], group)
        if verbose:
            print(f"  {take.report()}")
        builder.write()
    return take.kept


def clean_text(text: str) -> str:
    """What one utterance says, with the corpus's annotation markup taken out.

    The markup is an empty tag where the annotators heard something that is not
    a word to transcribe — a pause (``<ST/>``), a filler (``<FILL/>``), a laugh,
    a stretch nobody made out (``<UNKNOWN/>``), a background noise — or a tag
    around a word they marked, such as a name (``<PName> Peter </PName>``). Only
    the tags go: a pause leaves the words around it, and a marked word stays
    what the speaker said. Nothing is put in their place, so the only change
    beyond the words themselves is the punctuation the corpus's padding left
    behind. An utterance that was nothing but empty tags cleans to no text at
    all, and the plan of the preparation then drops it like any other row
    without a reference.
    """
    without_tags = WHITESPACE.sub(" ", MARKUP.sub("", text)).strip()
    return SPACED_PUNCTUATION.sub(r"\1", without_tags)


def _digests(entry: dict) -> dict[str, str]:
    """The checksum the Hub publishes for one file, as ``download`` wants it.

    A file the Hub stores through its LFS carries a sha256 under ``lfs.oid``; a
    file it keeps as a plain git object carries the sha1 of that object under
    ``oid``. A file with neither is a file nobody can pin, and the preparation
    refuses it rather than downloading it unchecked.
    """
    if oid := (entry.get("lfs") or {}).get("oid"):
        return {"sha256": str(oid)}
    if oid := entry.get("oid"):
        return {"git_blob_sha1": str(oid)}
    return {}


class Release:
    """The pinned revision of the corpus, as the Hub serves it.

    Every file is asked for by URL and checked against the checksum the Hub
    publishes for that revision, so a file that changed under the pinned revision
    stops the preparation instead of being trained on. One listing names every
    file of the split and is read once; the meetings, their transcripts, and each
    recording's checksum and URL are all found through it.
    """

    def __init__(self, name: str, cache_dir: Path) -> None:
        self.name = name
        self.cache_dir = cache_dir
        self._files: dict[str, dict] | None = None

    def meetings(self) -> list[Meeting]:
        """Every meeting of the split, with its transcript read.

        The meetings come from the Hub's own listing, by the transcript it holds
        for each of them, so a meeting added to the split is read rather than
        silently skipped and a release that moves the meetings stops this source
        instead of preparing an empty set. The transcripts are read at the same
        time: each one is a wait on the network.
        """
        transcripts = sorted(path for path in self.files() if path.endswith(f"/{TRANSCRIPT_NAME}"))
        if not transcripts:
            raise DownloadError(f"{REPO}@{REVISION} holds no {TRANSCRIPT_NAME}")
        return sorted(self._in_parallel(self._meeting, transcripts), key=lambda held: held.meeting)

    def files(self) -> dict[str, dict]:
        """Every file of the split, by its path inside the repository.

        One answer holds a thousand files and the split has more, so the listing
        is followed page by page; a file left out would look like a file the
        corpus does not have. Only files are kept: a directory and the file named
        after it would share one path, and only the file has a checksum and a URL.
        """
        if self._files is None:
            self._files = {
                entry["path"]: entry
                for entry in huggingface_entries(REPO, REVISION, DIRECTORY)
                if entry["type"] == "file"
            }
        return self._files

    def recordings(self, meeting: Meeting, utterances: list[Chunk]) -> dict[str, Audio]:
        """Download and read the close-talking recordings one meeting's clips need.

        Only the recordings the planned clips are on are read, and they are read
        at the same time, so a meeting costs one wait on the network rather than
        one per participant.
        """
        wanted = sorted({utterance.recording for utterance in utterances})
        for recording in wanted:
            if Path(recording).parts[:1] != (CLOSE_TALK,) or len(Path(recording).parts) != 2:
                raise DownloadError(
                    f"{recording} is not a {CLOSE_TALK}/<name> recording of {meeting.meeting}: "
                    "a transcript must name the microphone of the meeting it belongs to"
                )
        paths = self._in_parallel(
            lambda recording: self._download(f"{meeting.directory}/{recording}"), wanted
        )
        return {recording: load_audio(path) for recording, path in zip(wanted, paths, strict=True)}

    def _meeting(self, path: str) -> Meeting:
        """One meeting, with its transcript read.

        ``path`` is the transcript's path in the repository, which the listing
        gave; the directory above it is the meeting.
        """
        directory = path.rsplit("/", 1)[0]
        transcript = self._download(path)
        try:
            rows = json.loads(transcript.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise DownloadError(f"{transcript.name}: is not readable JSON ({error})") from None
        if not isinstance(rows, list):
            raise DownloadError(f"{transcript.name}: does not hold a list of utterances")
        meeting = directory.rsplit("/", 1)[-1]
        return Meeting(
            meeting=meeting,
            directory=directory,
            utterances=[
                self._chunk(meeting, transcript.name, index, row) for index, row in enumerate(rows)
            ],
        )

    def _chunk(self, meeting: str, source: str, index: int, row) -> Chunk | None:
        """One row of a transcript, as a chunk of speech, or ``None`` when it says nothing.

        A row the annotators filled with markup alone is a stretch of speech
        nobody wrote down, and so is a row holding a word they could not make out;
        a clip built across either would carry audio its text does not describe.
        """
        try:
            text = clean_text(row["text"])
            if not text or any(mark in row["text"] for mark in UNWRITTEN):
                return None
            return Chunk(
                meeting=meeting,
                speaker=row["speaker_id"],
                recording=row["ct_wav_file_name"],
                index=index,
                start=float(row["start_time"]),
                end=float(row["end_time"]),
                text=text,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise DownloadError(
                f"{source}: row {index} is not an utterance of this corpus ({error})"
            ) from None

    def _download(self, path: str) -> Path:
        """One file of the corpus, checked against the checksum it is pinned by.

        ``path`` is the file's own path inside the repository, as the listing
        gives it, and it is looked up there: a file the listing does not name —
        or names without a checksum — is refused rather than downloaded
        unchecked. The file lands in the cache under that path inside the split,
        so the recording of one meeting is never taken for the recording of
        another.
        """
        digests = _digests(self.files().get(path, {}))
        if not digests:
            raise DownloadError(
                f"{REPO}@{REVISION} lists no checksum for {path}: this source downloads only "
                "the files the revision it is pinned to names and checksums"
            )
        relative = Path(path).relative_to(DIRECTORY)
        return download(self.url(path), self.cache_dir / self.name / relative, **digests)

    def url(self, path: str) -> str:
        """Where the Hub serves one file of the dataset at the pinned revision.

        ``path`` is the file's own path inside the repository, which the listing
        gives and which tells a file of this source from a file of the simulated
        set that shares the repository.
        """
        if not path.startswith(f"{DIRECTORY}/"):
            raise DownloadError(f"{path} is not a file of {DIRECTORY}")
        return f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{path}"

    def _in_parallel(self, work, items):
        """Run ``work`` over ``items`` at the same time, keeping the order of the result."""
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            return list(pool.map(work, items))
