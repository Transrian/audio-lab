"""Clips labelled by a human transcript that has no timestamps.

Some talks come with a transcript somebody wrote: cased, punctuated, close to
what was said, but with no timing at all. Whisper's segments say *when* each
stretch was said; the transcript says *what* was said, better than Whisper does.
This module joins the two:

1. Both texts become word streams in the scorer's ``normalized`` view
   (:mod:`audiolab.eval.normalize`), one written word at a time, so that casing,
   punctuation and ``2024`` against ``twenty twenty-four`` do not count as
   differences. Each normalized word remembers the written word it comes from.
2. :func:`align` pairs the two streams with :class:`difflib.SequenceMatcher`:
   the pairs are the words both texts have, in the same order.
3. :func:`labelled_clips` takes Whisper's candidate clips
   (:func:`audiolab.data.talk_clips.candidates`, which cut on segment
   boundaries) and keeps those the transcript covers closely
   (:func:`judge`). A kept clip's label is the transcript's own written words for
   that stretch, with their casing and punctuation.

Nothing here knows where the talk came from: a transcript is a list of
:class:`Line`, and the module of each source reads its pages into one.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass

from audiolab.data.talk_clips import Segment
from audiolab.eval.normalize import normalize

# What a kept clip needs (a choice for the training data, not an owner
# directive): nine words in ten paired on each side, and never more than two
# words in a row that one side has and the other does not.
MIN_MATCHED_SHARE = 0.9
MAX_UNMATCHED_RUN = 2
MIN_CLIP_SECONDS = 1.0
# A note of the transcriber, which nobody says: "[ Music ]", "[[Sound: Beep]]",
# "(laughs)", a timestamp such as "[00:01:26]". A note ends a passage, like a
# change of speaker (see Transcript).
NOTE = re.compile(r"\[+[^\]]*\]+|\([^)]*\)")


@dataclass(frozen=True)
class Line:
    """One line of a transcript: who speaks (empty when it does not say), and what."""

    speaker: str
    text: str


@dataclass(frozen=True)
class Transcript:
    """A human transcript as a stream of normalized words.

    ``written`` holds the transcript's own words, as written, without speaker
    labels and notes. ``words`` holds the normalized words the alignment
    compares; for each of them, ``written_of`` is the index of the written word
    it comes from (``1,000`` gives two words, ``—`` none), and ``passage_of`` the
    passage it is said in. A passage is what one speaker says between two
    changes of speaker or notes: a clip never spans two of them, since a change
    of voice or a laugh inside a clip is where a label is least reliable.
    """

    written: list[str]
    words: list[str]
    written_of: list[int]
    passage_of: list[int]


@dataclass(frozen=True)
class Alignment:
    """For each word of each stream, the index of its pair in the other, or -1."""

    human: list[int]
    whisper: list[int]


def words_of(text: str, language: str) -> list[str]:
    """One written word in the ``normalized`` view: zero, one or several words."""
    return normalize(text, language).split()


def transcript(lines: list[Line], language: str) -> Transcript:
    """The word stream of a transcript, its passages numbered as they come."""
    written: list[str] = []
    words: list[str] = []
    written_of: list[int] = []
    passage_of: list[int] = []
    speaker, passage = "", 0
    for line in lines:
        if line.speaker and line.speaker != speaker:
            speaker, passage = line.speaker, passage + 1
        for piece_number, piece in enumerate(NOTE.split(line.text)):
            if piece_number:  # a note came before this piece
                passage += 1
            for token in piece.split():
                written.append(token)
                for word in words_of(token, language):
                    words.append(word)
                    written_of.append(len(written) - 1)
                    passage_of.append(passage)
    return Transcript(written, words, written_of, passage_of)


def align(human: list[str], whisper: list[str]) -> Alignment:
    """The pairs of words the two streams share, in order.

    :class:`difflib.SequenceMatcher` repeatedly takes the longest run of words
    both streams have and recurses on each side of it, so a word is paired only
    where its neighbours agree as well. ``autojunk`` is off: it would ignore the
    words a long text uses most ("the", "and"), which are words like any other.
    """
    human_pairs, whisper_pairs = [-1] * len(human), [-1] * len(whisper)
    matcher = difflib.SequenceMatcher(None, human, whisper, autojunk=False)
    for human_start, whisper_start, size in matcher.get_matching_blocks():
        for offset in range(size):
            human_pairs[human_start + offset] = whisper_start + offset
            whisper_pairs[whisper_start + offset] = human_start + offset
    return Alignment(human_pairs, whisper_pairs)


def labelled_clips(
    clips: list[Segment], lines: list[Line], language: str, tally: Counter[str]
) -> list[tuple[Segment, str]]:
    """The candidate clips the transcript covers, each with the transcript's text for it.

    Whisper's words are numbered across all the clips, so that one alignment
    covers the whole talk, and each clip is then judged on its own stretch of it.
    The reason a clip is left out is counted in ``tally``.
    """
    human = transcript(lines, language)
    whisper: list[str] = []
    spans: list[range] = []
    for clip in clips:
        heard = words_of(clip.text, language)
        spans.append(range(len(whisper), len(whisper) + len(heard)))
        whisper += heard
    alignment = align(human.words, whisper)
    labelled = []
    for clip, span in zip(clips, spans, strict=True):
        label, reason = judge(clip, span, human, alignment)
        if reason:
            tally[f"clips left out: {reason}"] += 1
        else:
            labelled.append((clip, label))
    return labelled


def judge(clip: Segment, heard: range, human: Transcript, alignment: Alignment) -> tuple[str, str]:
    """The label of one clip and an empty reason, or no label and why there is none.

    ``heard`` is the clip's stretch of Whisper's words. The transcript's stretch
    is the one between the pairs of the clip's first and last words, and the two
    must agree closely:

    - the clip lasts :data:`MIN_CLIP_SECONDS` at least (the longest clip is
      what the candidates allow);
    - the first and last words Whisper heard are in the transcript, and the
      transcript has no word of its own just outside its stretch: such a word
      may have been said inside the clip, or not, and nothing says which;
    - the stretch starts and ends on whole written words, so the label never
      holds half of ``1,000``;
    - the stretch is one passage: no change of speaker and no note inside;
    - :data:`MIN_MATCHED_SHARE` of the words of each side are paired, and no
      side has more than :data:`MAX_UNMATCHED_RUN` unpaired words in a row.
    """
    if clip.end_s - clip.start_s < MIN_CLIP_SECONDS:
        return "", f"shorter than {MIN_CLIP_SECONDS:g} s"
    if not heard or alignment.whisper[heard[0]] < 0 or alignment.whisper[heard[-1]] < 0:
        return "", "an edge word is not in the transcript"
    first, last = alignment.whisper[heard[0]], alignment.whisper[heard[-1]]
    said = range(first, last + 1)
    before, after = first - 1, last + 1
    if (before >= 0 and alignment.human[before] < 0) or (
        after < len(human.words) and alignment.human[after] < 0
    ):
        return "", "the transcript has an unpaired word at an edge"
    if (before >= 0 and human.written_of[before] == human.written_of[first]) or (
        after < len(human.words) and human.written_of[after] == human.written_of[last]
    ):
        return "", "an edge cuts a written word"
    if human.passage_of[first] != human.passage_of[last]:
        return "", "a change of speaker or a note inside"
    paired = sum(alignment.whisper[index] >= 0 for index in heard)
    if paired < MIN_MATCHED_SHARE * max(len(heard), len(said)):
        return "", "too few words paired"
    if (
        max(longest_unpaired(alignment.whisper, heard), longest_unpaired(alignment.human, said))
        > MAX_UNMATCHED_RUN
    ):
        return "", f"more than {MAX_UNMATCHED_RUN} unpaired words in a row"
    written = human.written[human.written_of[first] : human.written_of[last] + 1]
    return " ".join(written), ""


def longest_unpaired(pairs: list[int], stretch: range) -> int:
    """The longest run of words of ``stretch`` that have no pair."""
    longest = run = 0
    for index in stretch:
        run = run + 1 if pairs[index] < 0 else 0
        longest = max(longest, run)
    return longest
