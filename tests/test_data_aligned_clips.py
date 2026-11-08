"""Labelling Whisper's clips from a human transcript: the alignment and the rules.

Every example is written by hand: a transcript as a page would give it, and the
clips Whisper would cut, with the insertions, deletions and changes of speaker
the rules are there for.
"""

from __future__ import annotations

from collections import Counter

from audiolab.data.aligned_clips import Line, align, labelled_clips, transcript
from audiolab.data.talk_clips import Segment

HOST = "Gary Jordan"
GUEST = "Kathryn Martinez"


def label(clips: list[Segment], lines: list[Line]) -> tuple[list[str], Counter[str]]:
    tally: Counter[str] = Counter()
    labelled = labelled_clips(clips, lines, "en", tally)
    return [text for _, text in labelled], tally


def test_the_transcript_stream_leaves_out_labels_and_notes_and_numbers_its_passages():
    lines = [
        Line(HOST, "Welcome back, Kathryn."),
        Line("", "[ Music ] We have 2,000 questions."),
        Line(GUEST, "(laughs) Go ahead."),
    ]

    stream = transcript(lines, "en")

    assert " ".join(stream.written) == "Welcome back, Kathryn. We have 2,000 questions. Go ahead."
    assert " ".join(stream.words) == "welcome back kathryn we have two thousand questions go ahead"
    assert stream.written_of[5:7] == [5, 5], "both words of 2,000 come from one written word"
    # The host speaks (1), a note ends that passage (2), the guest speaks (3)
    # and starts with a note (4).
    assert stream.passage_of == [1, 1, 1, 2, 2, 2, 2, 2, 4, 4]


def test_the_alignment_pairs_the_shared_words_around_insertions_and_deletions():
    human = "so we flew to the moon and back".split()
    whisper = "so uh we flew to moon and back again".split()

    alignment = align(human, whisper)

    # "uh" and "again" are Whisper's own; "the" is the transcript's own.
    assert alignment.whisper == [0, -1, 1, 2, 3, 5, 6, 7, -1]
    assert alignment.human == [0, 2, 3, 4, -1, 5, 6, 7]


def test_a_clip_heard_as_written_is_labelled_with_the_transcripts_own_text():
    lines = [Line(HOST, "Welcome to Houston, Episode 239! Today: the Moon.")]
    clips = [Segment(0.0, 6.0, "welcome to houston episode two hundred thirty nine today the moon")]

    labels, _ = label(clips, lines)

    assert labels == ["Welcome to Houston, Episode 239! Today: the Moon."]


LONG = (
    "The suit is heavy and the gloves make every small task slow so we practise each "
    "step many times in the pool before we ever try it outside the station in orbit."
)


def test_one_word_whisper_adds_passes_and_three_in_a_row_do_not():
    lines = [
        Line(HOST, "We trained for the spacewalk in the big pool for many months."),
        Line(GUEST, LONG),
    ]
    clips = [
        Segment(0.0, 5.0, "We trained for the the spacewalk in the big pool for many months."),
        Segment(5.0, 15.0, LONG.replace("slow so", "slow you know what so")),
    ]

    labels, tally = label(clips, lines)

    assert labels == ["We trained for the spacewalk in the big pool for many months."]
    assert tally["clips left out: more than 2 unpaired words in a row"] == 1


def test_a_word_whisper_missed_is_in_the_label_and_three_missed_words_reject_the_clip():
    lines = [
        Line(HOST, "The station orbits the Earth every ninety minutes at a very high speed."),
        Line(GUEST, LONG.replace("slow so", "slow really truly honestly so")),
    ]
    clips = [
        Segment(0.0, 5.0, "The station orbits Earth every ninety minutes at a very high speed."),
        Segment(5.0, 15.0, LONG),
    ]

    labels, tally = label(clips, lines)

    assert labels == ["The station orbits the Earth every ninety minutes at a very high speed."]
    assert tally["clips left out: more than 2 unpaired words in a row"] == 1


def test_a_clip_with_too_many_words_of_its_own_is_left_out_even_one_at_a_time():
    lines = [Line(HOST, "We launch at dawn and land on the Moon three days later.")]
    clips = [Segment(0.0, 5.0, "we launch at dawn and then land on the Moon some three days later")]

    labels, tally = label(clips, lines)

    assert labels == []
    assert tally["clips left out: too few words paired"] == 1


def test_a_clip_that_spans_a_change_of_speaker_or_a_note_is_left_out():
    lines = [
        Line(HOST, "Thanks for coming on the show today."),
        Line(GUEST, "My pleasure, thank you."),
        Line("", "I love it here. [laughter] It is a lot of fun."),
    ]
    clips = [
        Segment(0.0, 6.0, "Thanks for coming on the show today. My pleasure, thank you."),
        Segment(6.0, 10.0, "I love it here. It is a lot of fun."),
    ]

    labels, tally = label(clips, lines)

    assert labels == []
    assert tally["clips left out: a change of speaker or a note inside"] == 2


def test_an_unpaired_transcript_word_at_an_edge_rejects_the_clip_it_may_belong_to():
    lines = [Line(HOST, "Well, today we are talking about rockets and engines.")]
    # Whisper did not hear "Well,": was it said inside the first clip or before?
    clips = [Segment(0.0, 4.0, "today we are talking about rockets and engines")]

    labels, tally = label(clips, lines)

    assert labels == []
    assert tally["clips left out: the transcript has an unpaired word at an edge"] == 1


def test_a_clip_edge_inside_a_written_number_rejects_both_sides():
    lines = [Line(HOST, "We packed 2,000 meals for the long trip to the station.")]
    clips = [
        Segment(0.0, 3.0, "we packed two"),
        Segment(3.0, 7.0, "thousand meals for the long trip to the station"),
    ]

    labels, tally = label(clips, lines)

    assert labels == []
    assert tally["clips left out: an edge cuts a written word"] == 2


def test_a_clip_shorter_than_a_second_or_heard_without_the_transcript_is_left_out():
    lines = [Line(HOST, "Yes. Now let us move on to the next question about the launch.")]
    clips = [
        Segment(0.0, 0.6, "Yes."),
        Segment(1.0, 5.0, "now let us move on to the next question about the launch"),
        Segment(6.0, 9.0, "music plays in the background"),
    ]

    labels, tally = label(clips, lines)

    assert labels == ["Now let us move on to the next question about the launch."]
    assert tally["clips left out: outside 1-19 s"] == 1
    assert tally["clips left out: an edge word is not in the transcript"] == 1
