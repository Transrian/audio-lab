"""What the meeting sources share: how an annotation becomes the clips of a speaker.

Three sources of this project are meetings recorded with one close microphone per
participant, and each corpus publishes its own annotation: AMI annotates one word
at a time with its times in a separate layer, ICSI writes one segment per
utterance by hand, NOTSOFAR one row per utterance as JSON. They are not the same
data, but they raise the same questions, and this module answers them the same
way for all three:

- a word the annotators wrote letter by letter (``T_V_``) is the word it spells;
- two stretches of one speaker's speech belong in one clip when the silence
  between them is short, when the first does not end a sentence, and while the
  clip stays inside the length the training recipe batches;
- what a corpus leaves out of its text — a laugh, a cough, a number read out
  loud, a phrase nobody made out — ends the clip being built, so a clip never
  holds audio that its reference does not describe.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, replace

from audiolab.data.train import MAX_SECONDS

# A silence longer than this separates two utterances, so a clip never spans it.
GAP_SECONDS = 1.0
# The marks that end a sentence. A clip ends there even when the speaker goes on
# after a short silence: the two sentences are separate references.
SENTENCE_END = frozenset(".!?")
# A word spelled letter by letter, each letter followed by an underscore and
# closed by the last letter (``T_V_`` is "TV", ``L_C_D`` is "LCD").
SPELLED_LETTERS = re.compile(r"(?<![A-Za-z])((?:[A-Za-z]_)+[A-Za-z]?)(?![A-Za-z])")


def spelled_out(text: str) -> str:
    """A word the annotators wrote letter by letter, as the word it spells.

    The corpora mark a spelled-out word by an underscore after each letter, which
    is a mark of how it was written, not of how it was said.
    """
    return SPELLED_LETTERS.sub(lambda match: match.group(1).replace("_", ""), text)


@dataclass(frozen=True)
class Chunk:
    """One stretch of one speaker's speech, as a corpus's annotation gives it."""

    meeting: str
    speaker: str
    # What has to be read to hear this stretch: a channel of the meeting for
    # ICSI, the close-talking recording of that participant for NOTSOFAR.
    recording: str
    # The row's number in the corpus's own file, so that the name of a clip says
    # which row it starts at and does not move when a filter above it changes.
    index: int
    start: float
    end: float
    text: str

    @property
    def id(self) -> str:
        """A stable name for the clip, unique inside the corpus."""
        return f"{self.meeting}.{self.speaker}.{self.index:04d}"

    @property
    def seconds(self) -> float:
        """How long the chunk lasts."""
        return self.end - self.start

    @property
    def ends_sentence(self) -> bool:
        """Whether the annotators closed this stretch with sentence punctuation."""
        return self.text.rstrip().endswith(tuple(SENTENCE_END))

    def joined_with(self, following: Chunk) -> Chunk:
        """This chunk and the next one of the same speaker, as one clip.

        The clip keeps the name, the speaker and the start of its first chunk,
        because that is where it begins; only its end and its text grow.
        """
        text = f"{self.text} {following.text}".strip()
        return replace(self, end=following.end, text=text)


def merged(
    chunks: Iterable[Chunk | None],
    *,
    max_gap: float = GAP_SECONDS,
    max_seconds: float = MAX_SECONDS,
) -> list[Chunk]:
    """Join the consecutive chunks of one speaker into the clips of a source.

    ``None`` is a stretch that is not a clip — a laugh, a phrase nobody made out,
    a number read out loud, a stretch on another speaker's microphone, a segment
    the corpus does not let this source use — and it ends the clip being built, so
    a clip never spans speech that was left out of its own text.

    Two chunks of one speaker are joined when at most ``max_gap`` of silence
    separates them, when the first does not end a sentence, and while the joined
    clip would stay within ``max_seconds``. The joined clip is what the training
    recipe sees, and a clip cut at an arbitrary boundary is a reference nobody
    would write: it starts and ends mid-phrase.

    A chunk longer than ``max_seconds`` is returned as it is: the length filter of
    the cap is what refuses it, so the report says the source holds it.
    """
    clips: list[Chunk] = []
    current: Chunk | None = None

    def close() -> None:
        nonlocal current
        if current is not None:
            clips.append(current)
            current = None

    for chunk in chunks:
        if chunk is None:
            close()
            continue
        if current is not None and (
            chunk.speaker != current.speaker
            or chunk.recording != current.recording
            or chunk.start - current.end > max_gap
            or chunk.end - current.start > max_seconds
            or current.ends_sentence
        ):
            close()
        current = chunk if current is None else current.joined_with(chunk)
    close()
    return clips
