"""From Whisper's segments to the candidate clips of a talk, and the answer cache."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from audiolab.core.audio import SAMPLE_RATE, Audio
from audiolab.data import peertube_clips
from audiolab.data.peertube_clips import Answers, Segment


def silence(seconds: float) -> Audio:
    return Audio(
        samples=np.zeros(round(seconds * SAMPLE_RATE), np.float32), sample_rate=SAMPLE_RATE
    )


def whisper_row(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": f" {text}"}


def seconds_sent(wav: bytes) -> float:
    return (len(wav) - 44) / 2 / SAMPLE_RATE


def segment(text: str = "bonjour à tous", start: float = 0.0, end: float = 2.0) -> Segment:
    return Segment(start_s=start, end_s=end, text=text)


def test_a_talk_is_sent_in_chunks_that_restart_after_the_last_whole_segment(tmp_path):
    sent = []

    def whisper(wav: bytes) -> dict:
        sent.append(seconds_sent(wav))
        if len(sent) == 1:
            rows = [whisper_row(0.0, 120.0, "un"), whisper_row(120.0, 290.0, "deux")]
            return {"segments": rows + [whisper_row(290.0, 300.0, "coupé au mi-")]}
        return {"segments": [whisper_row(0.0, 5.0, "trois"), whisper_row(5.0, 100.0, "quatre")]}

    answers = Answers(tmp_path / "whisper.jsonl")
    segments = peertube_clips.segment_audio(silence(390.0), whisper, answers)

    assert sent == [300.0, 100.0], "the second chunk starts at 290 s, where 'deux' ended"
    assert [(s.start_s, s.end_s, s.text) for s in segments] == [
        (0.0, 120.0, "un"),
        (120.0, 290.0, "deux"),
        (290.0, 295.0, "trois"),
        (295.0, 390.0, "quatre"),
    ]

    def unreachable(wav: bytes) -> dict:
        raise AssertionError("every chunk is in the cache")

    again = peertube_clips.segment_audio(silence(390.0), unreachable, Answers(answers.path))
    assert again == segments, "the raw answers are cached and read again"


def test_a_chunk_without_a_whole_segment_moves_on_by_a_full_chunk(tmp_path):
    answers = iter([{"segments": [whisper_row(0.0, 300.0, "musique")]}, {"segments": []}])
    sent = []

    def whisper(wav: bytes) -> dict:
        sent.append(wav)
        return next(answers)

    cache = Answers(tmp_path / "whisper.jsonl")
    assert peertube_clips.segment_audio(silence(400.0), whisper, cache) == []
    assert len(sent) == 2


def test_a_torn_last_line_is_asked_again_and_never_glued_to(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_text('{"key": "0.00-1.00", "answer": {"text": "un"}}\n{"key": "1.00-2', "utf-8")

    answers = Answers(path)
    assert answers.answers == {"0.00-1.00": {"text": "un"}}
    answers.answer("1.00-2.00", lambda: {"text": "deux"})

    again = Answers(path)
    assert again.answers == {"0.00-1.00": {"text": "un"}, "1.00-2.00": {"text": "deux"}}


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (segment(), ""),
        (segment(""), "without words"),
        (segment("Euh..."), "without words"),
        (segment(start=3.0, end=3.0), "without duration"),
    ],
)
def test_a_segment_without_words_or_duration_is_doubtful(candidate, reason):
    assert peertube_clips.doubt(candidate) == reason


def texts(clips: list[Segment]) -> list[str]:
    return [clip.text for clip in clips]


def test_phrases_are_joined_up_to_a_full_stop_once_the_clip_lasts_five_seconds():
    phrases = [
        segment("Alors, F-Droid,", 0.0, 1.5),
        segment("c'est un magasin d'applications.", 1.5, 6.0),
        segment("Oui.", 6.2, 6.5),
        segment("Mais pas comme le Play Store ?", 6.5, 12.0),
    ]
    assert texts(peertube_clips.candidates(phrases, Counter())) == [
        "Alors, F-Droid, c'est un magasin d'applications.",
        "Oui. Mais pas comme le Play Store ?",
    ]


def test_a_long_pause_a_doubtful_segment_or_the_length_limit_end_a_clip():
    tally = Counter()
    phrases = [
        segment("avant la pause", 0.0, 2.0),
        segment("après la pause", 3.5, 5.0),
        segment("...", 5.0, 6.0),
        segment("et on reprend", 6.0, 7.0),
        segment("une phrase sans fin", 7.0, 26.0),
    ]
    assert texts(peertube_clips.candidates(phrases, tally)) == [
        "avant la pause",
        "après la pause",
        "et on reprend",
        "une phrase sans fin",
    ]
    assert tally == Counter({"segments without words": 1})


def test_a_clip_whose_text_loops_is_dropped_whole():
    tally = Counter()
    phrases = [segment("c'est un test", 0.0, 2.0), segment("merci " * 20, 2.0, 6.0)]
    assert peertube_clips.candidates(phrases, tally) == []
    assert tally == Counter({"clips in a repetition loop": 1})


def test_a_segment_loses_the_words_it_repeats_from_the_one_before():
    tally = Counter()
    phrases = [
        segment("Merci à vous.", 0.0, 1.0),
        segment("Merci à vous.", 1.0, 2.0),
        segment("Chaque membre...", 2.0, 2.5),
        segment("Chaque membre coopérateur vote.", 2.5, 5.0),
        segment("Non.", 5.0, 5.5),
        segment("Oui.", 5.5, 6.0),
    ]
    assert texts(peertube_clips.without_echoes(phrases, tally)) == [
        "Merci à vous.",
        "",
        "Chaque membre...",
        "coopérateur vote.",
        "Non.",
        "Oui.",
    ]
    assert tally == Counter({"segments that repeat the one before": 2})


def test_a_clip_is_padded_by_at_most_half_the_gap_to_its_neighbours():
    clip = segment("bonjour", 1.0, 3.0)
    assert peertube_clips.padded(clip, previous_end=0.5, next_start=3.1) == segment(
        "bonjour", 0.8, 3.05
    )
    assert peertube_clips.padded(clip, previous_end=1.2, next_start=2.0) == clip, (
        "a neighbour that overlaps leaves the edge where it is"
    )


def test_padding_never_takes_a_clip_past_the_server_limit():
    clip = segment("une longue phrase", 10.0, 28.9)
    widened = peertube_clips.padded(clip, previous_end=0.0, next_start=float("inf"))
    assert widened == segment("une longue phrase", 9.9, 28.9)
    assert widened.end_s - widened.start_s == pytest.approx(peertube_clips.MAX_CLIP_SECONDS)
