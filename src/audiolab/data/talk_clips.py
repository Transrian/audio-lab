"""From one talk to the clips worth labelling, and the answers kept on the way.

Nothing here knows where the talk came from: a talk is a WAV and a language.
Whisper answers for the whole talk with timestamped segments: phrases of about
two seconds, some of them doubtful. This module turns them into candidate
clips — sentences of a few seconds, with a little silence on each side — and
keeps every answer of every model in a cache, so a stopped preparation resumes
where it was. :mod:`audiolab.data.talks` runs the models over a whole batch of
talks.
"""

from __future__ import annotations

import io
import json
import os
import threading
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import soundfile

from audiolab.core.audio import SAMPLE_RATE, Audio, to_int16
from audiolab.data.train import MAX_SECONDS, MIN_SECONDS
from audiolab.eval.normalize import normalize

# Whisper is sent five minutes at a time: the server takes fifteen-minute
# requests, and a short chunk is cheaper to retry.
CHUNK_SECONDS = 300.0
# The label model's server answers HTTP 500 to audio longer than about 19 s,
# so this source's clips stop there rather than at the 20 s of every source.
MAX_CLIP_SECONDS = min(MAX_SECONDS, 19.0)
# Whisper's own loop rule, the default of openai-whisper: a text that compresses
# better than 2.4 to 1 repeats itself.
MAX_COMPRESSION_RATIO = 2.4
# Segments are joined across pauses up to this long; a longer pause ends the
# clip. A full stop ends it too, once it lasts CLIP_TARGET_S: a dictated
# request is a sentence or two, and a talk is full of one-word replies ("Oui.")
# that would otherwise make most of the clips.
MAX_PAUSE_S = 1.0
CLIP_TARGET_S = 5.0
SENTENCE_ENDS = (".", "?", "!", "…")
# Whisper's timestamps fall at the very edge of the speech, and a model that
# gets only the clip misses a first or last word cut that tight.
PAD_S = 0.2

# A Python callable that sends one WAV to one model and returns its JSON answer.
Transcribe = Callable[[bytes], dict]


@dataclass(frozen=True)
class Segment:
    """One stretch of a talk and its text, in seconds from the video start."""

    start_s: float
    end_s: float
    text: str

    @property
    def key(self) -> str:
        """What a cached answer belongs to: the stretch of audio it was heard in."""
        return f"{self.start_s:.2f}-{self.end_s:.2f}"


class Answers:
    """One model's answers for one video: one JSON line per request, appended as it arrives.

    The raw answer is kept, not what was read from it, so a change to the
    reading never reuses an old reading. A crash in the middle of a write can
    leave a torn last line; it is skipped, and its request is sent again.
    Requests for different keys may be answered from several threads at once;
    their lines are written one at a time, each whole.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.answers: dict[str, dict] = {}
        self.writing = threading.Lock()
        if not path.is_file():
            return
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: {path}:{number} is a torn line; its request is sent again")
                continue
            self.answers[row["key"]] = row["answer"]

    def answer(self, key: str, ask: Callable[[], dict]) -> dict:
        """The cached answer for ``key``, or ``ask()``'s, cached on the way."""
        if key not in self.answers:
            self.answers[key] = ask()
            self._append({"key": key, "answer": self.answers[key]})
        return self.answers[key]

    def _append(self, row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.writing, self.path.open("a+b") as handle:
            # Never glue a new answer onto a torn line.
            if handle.seek(0, os.SEEK_END) and _last_byte(handle) != b"\n":
                line = "\n" + line
            handle.write(line.encode("utf-8"))


def _last_byte(handle) -> bytes:
    handle.seek(-1, os.SEEK_END)
    return handle.read(1)


def segment_audio(audio: Audio, transcribe: Transcribe, answers: Answers) -> list[Segment]:
    """Every segment Whisper hears in the audio, sent in chunks of ``CHUNK_SECONDS``.

    A chunk ends at a fixed time, which may fall inside a word, so its last
    segment is not trusted: the next chunk starts where the segment before it
    ended, and hears that sentence again whole. Each chunk's answer is cached
    under its stretch of audio.
    """
    segments: list[Segment] = []
    start = 0.0
    while audio.duration_s - start >= MIN_SECONDS:
        end = min(start + CHUNK_SECONDS, audio.duration_s)
        chunk = Segment(round(start, 2), round(end, 2), "")
        answer = answer_on(chunk, audio, transcribe, answers)
        heard = [_segment(row, chunk.start_s) for row in answer["segments"]]
        if end >= audio.duration_s:
            return segments + heard
        heard = heard[:-1]
        segments.extend(heard)
        start = heard[-1].end_s if heard and heard[-1].end_s > start else end
    return segments


def _segment(row: dict, offset_s: float) -> Segment:
    return Segment(
        start_s=round(offset_s + float(row["start"]), 2),
        end_s=round(offset_s + float(row["end"]), 2),
        text=" ".join(str(row["text"]).split()),
    )


def candidates(segments: list[Segment], language: str, tally: Counter[str]) -> list[Segment]:
    """Whisper's segments joined into sentences: the clips that may be labelled.

    A phrase cut out of a sentence is a poor clip, so consecutive segments are
    joined until a full stop past :data:`CLIP_TARGET_S`, a pause longer than
    :data:`MAX_PAUSE_S`, or :data:`MAX_CLIP_SECONDS`. A segment without words or
    without duration is left out and ends the clip it was in. A clip whose text
    repeats itself is a decoding loop and is dropped. Each clip is widened by
    :func:`padded`.
    """
    segments = without_echoes(segments, tally)
    runs: list[tuple[int, int]] = []  # first and last index of each clip's segments
    first = None

    def close(last: int) -> None:
        nonlocal first
        if first is not None:
            runs.append((first, last))
            first = None

    for index, segment in enumerate(segments):
        reason = doubt(segment, language)
        if reason:
            tally[f"segments {reason}"] += 1
            close(index - 1)
            continue
        if first is not None and (
            segment.start_s - segments[index - 1].end_s > MAX_PAUSE_S
            or segment.end_s - segments[first].start_s > MAX_CLIP_SECONDS
        ):
            close(index - 1)
        if first is None:
            first = index
        lasted = segment.end_s - segments[first].start_s
        if segment.text.endswith(SENTENCE_ENDS) and lasted >= CLIP_TARGET_S:
            close(index)
    close(len(segments) - 1)

    clips = []
    for first, last in runs:
        clip = _joined(segments[first : last + 1])
        if compression_ratio(clip.text) > MAX_COMPRESSION_RATIO:
            tally["clips in a repetition loop"] += 1
            continue
        previous_end = segments[first - 1].end_s if first > 0 else 0.0
        next_start = segments[last + 1].start_s if last + 1 < len(segments) else float("inf")
        clips.append(padded(clip, previous_end, next_start))
    return clips


def without_echoes(segments: list[Segment], tally: Counter[str]) -> list[Segment]:
    """The segments, each without the words it repeats from the start of the one before.

    Whisper often starts a segment by saying the previous one again ("Merci à
    vous." seven times over seven segments); the words were heard once.
    """
    cleaned = []
    for previous, segment in zip([None, *segments], segments, strict=False):
        before = _words(previous.text) if previous else []
        said = segment.text.split()
        if before and _words(" ".join(said[: len(before)])) == before:
            segment = replace(segment, text=" ".join(said[len(before) :]))
            tally["segments that repeat the one before"] += 1
        cleaned.append(segment)
    return cleaned


def _words(text: str) -> list[str]:
    return [word for word in (token.strip(".,;:!?…").lower() for token in text.split()) if word]


def doubt(segment: Segment, language: str) -> str:
    """Why a segment cannot be part of a clip, or an empty string."""
    if segment.end_s <= segment.start_s:
        return "without duration"
    if not normalize(segment.text, language):
        return "without words"
    return ""


def padded(clip: Segment, previous_end: float, next_start: float) -> Segment:
    """The clip widened by up to :data:`PAD_S` on each side.

    Never past half the gap to the neighbouring segment, whose words must stay
    out, nor past :data:`MAX_CLIP_SECONDS`.
    """
    room = max(0.0, MAX_CLIP_SECONDS - (clip.end_s - clip.start_s))
    before = min(PAD_S, max(0.0, (clip.start_s - previous_end) / 2), room)
    after = min(PAD_S, max(0.0, (next_start - clip.end_s) / 2), room - before)
    return replace(
        clip, start_s=round(clip.start_s - before, 2), end_s=round(clip.end_s + after, 2)
    )


def _joined(segments: list[Segment]) -> Segment:
    text = " ".join(segment.text for segment in segments)
    return Segment(start_s=segments[0].start_s, end_s=segments[-1].end_s, text=text)


def compression_ratio(text: str) -> float:
    """How much better than plain text a text compresses: high for a loop."""
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data))


def answer_on(stretch: Segment, audio: Audio, transcribe: Transcribe, answers: Answers) -> dict:
    """A model's answer on one stretch of a talk, asked once."""
    return answers.answer(stretch.key, lambda: transcribe(wav_bytes(cut(audio, stretch).samples)))


def clip_text(answers: Answers, clip: Segment, audio: Audio, transcribe: Transcribe) -> str:
    """What a clip model heard in one clip, asked once."""
    return " ".join(str(answer_on(clip, audio, transcribe, answers)["text"]).split())


def cut(audio: Audio, segment: Segment) -> Audio:
    """The samples of one segment, copied so the whole talk is not kept alive."""
    start, end = round(segment.start_s * SAMPLE_RATE), round(segment.end_s * SAMPLE_RATE)
    return Audio(samples=audio.samples[start:end].copy(), sample_rate=SAMPLE_RATE)


def wav_bytes(samples: np.ndarray) -> bytes:
    """16 kHz mono samples as a 16-bit WAV in memory: what a request sends."""
    buffer = io.BytesIO()
    soundfile.write(buffer, to_int16(samples), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()
