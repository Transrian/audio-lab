"""What the meeting sources share: the chunk, its merging, and the letter spellings."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from audiolab.data.meeting import Chunk, merged, spelled_out  # noqa: E402


def chunk(
    index: int = 0,
    start: float = 0.0,
    end: float = 1.0,
    text: str = "bonjour",
    speaker: str = "spkA",
    recording: str = "chan0",
) -> Chunk:
    return Chunk(
        meeting="meeting",
        speaker=speaker,
        recording=recording,
        index=index,
        start=start,
        end=end,
        text=text,
    )


def test_a_spelled_out_word_is_the_word_it_spells():
    assert spelled_out("T_V_") == "TV"
    assert spelled_out("L_C_D_s") == "LCDs"
    assert spelled_out("anti-R_S_I_") == "anti-RSI"
    # A word without that mark is left alone, and so is an underscore that is
    # not between letters.
    assert spelled_out("Mm-hmm") == "Mm-hmm"
    assert spelled_out("tv_set") == "tv_set"


def test_two_chunks_of_one_speaker_join_when_the_silence_between_them_is_short():
    clips = merged([chunk(0, 0.0, 2.0, "We have"), chunk(1, 2.5, 4.0, "littler heads so,")])

    assert len(clips) == 1
    # The clip keeps the name and the start of the chunk it begins at.
    assert clips[0].id == "meeting.spkA.0000"
    assert (clips[0].start, clips[0].end) == (0.0, 4.0)
    assert clips[0].text == "We have littler heads so,"


def test_a_sentence_ends_the_clip_even_when_the_speaker_goes_on():
    clips = merged([chunk(0, 0.0, 2.0, "That is fine."), chunk(1, 2.5, 4.0, "And so is this.")])

    assert [clip.text for clip in clips] == ["That is fine.", "And so is this."]


def test_a_clip_does_not_span_a_silence_a_length_or_another_speaker():
    long_gap = merged([chunk(0, 0.0, 1.0, "one"), chunk(1, 3.0, 4.0, "two")])
    assert len(long_gap) == 2

    # Two chunks that would make a clip longer than the cap's window stay apart,
    # so the length the training recipe batches is never exceeded silently.
    too_long = merged([chunk(0, 0.0, 15.0, "one"), chunk(1, 15.5, 22.0, "two")])
    assert len(too_long) == 2
    assert too_long[1].text == "two"

    other = merged([chunk(0, 0.0, 1.0, "one"), chunk(1, 1.5, 2.0, "two", speaker="spkB")])
    assert [clip.speaker for clip in other] == ["spkA", "spkB"]

    channel = merged([chunk(0, 0.0, 1.0, "one"), chunk(1, 1.5, 2.0, "two", recording="chan1")])
    assert len(channel) == 2


def test_a_stretch_that_was_left_out_ends_the_clip_being_built():
    """A clip never carries audio the source's own text does not describe."""
    clips = merged([chunk(0, 0.0, 1.0, "one"), None, chunk(2, 1.5, 2.0, "two")])

    assert [clip.text for clip in clips] == ["one", "two"]


def test_a_chunk_longer_than_the_window_is_kept_for_the_cap_to_refuse():
    clips = merged([chunk(0, 0.0, 25.0, "a segment nobody cut")])

    assert len(clips) == 1
    assert clips[0].seconds == 25.0
