"""AMI (``ami-fr-speakers-en-train``): English meetings spoken by French-L1 participants.

The AMI meeting corpus is 171 meetings of four people around a table, recorded
with one close headset microphone per participant and transcribed by hand word
by word. The corpus and its annotations are CC-BY-4.0. It is part of the
training data of several English models, Parakeet among them, so it is not a
source of new data — it is the closest thing open corpora have to the target
use: spontaneous English spoken by people whose first language is French.

``corpusResources/participants.xml`` gives the native language of every
participant, ``corpusResources/meetings.xml`` says which speaker letter of which
meeting a participant is, and ``words/<meeting>.<letter>.words.xml`` holds that
speaker's words with their timestamps. 41 participants have French as their
native language; 20 of them are named by a meeting, in 43 meetings and 69
speaker recordings — mostly the IDIAP meetings (IS, IB, IN) and one scenario
meeting (ES2007). No meeting of these two resources names the other 21, so no
word file describes them and they cannot contribute a clip. The audio is one WAV
per participant, published
on the corpus's own mirror under ``amicorpus/<meeting>/audio/``, over HTTP and
without a login: a preparation downloads only the headset of the meetings it
takes.

A manual word file holds words, punctuation marks (``punc="true"``) and the
elements ``vocalsound``, ``disfmarker`` and ``gap``, which mark a laugh, a
disfluency and a stretch nobody transcribed. Only the words are read here,
punctuation is attached to the word before it, and a clip ends at sentence-final
punctuation, after a silence of more than one second, or when it would last
longer than 20 s. The text is cased and punctuated as the annotators wrote it.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

from audiolab.core.audio import SAMPLE_RATE, Audio, load_audio
from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError, download
from audiolab.data.train import (
    CLIP_CAP_PER_GROUP,
    MAX_SECONDS,
    TRAIN_CLIP_CAP,
    Take,
)

ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ANNOTATIONS_NAME = "ami_public_manual_1.6.2.zip"
# The corpus's own mirror, which serves the individual headset WAV files over
# plain HTTP without a login (and redirects to HTTPS).
AUDIO_URL = "http://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
PARTICIPANTS = "corpusResources/participants.xml"
MEETINGS = "corpusResources/meetings.xml"
WORDS_DIRECTORY = "words"
# The speaker letters of a meeting are the headset channels, in this order.
HEAD_NUMBERS = {"A": 0, "B": 1, "C": 2, "D": 3}
NATIVE_LANGUAGE = "French"
LICENSE = "CC-BY-4.0"
LANGUAGE = "en"
SPLIT = "manual"
# A silence longer than this separates two utterances, so a clip never spans it.
GAP_SECONDS = 1.0
SENTENCE_END = frozenset(".!?")
# A punctuation mark is written as its own word and belongs to the word before.
PUNCTUATION = re.compile(r"^[.,;:!?'\u2019-]+$")
NITE = "{http://nite.sourceforge.net/}"


@dataclass(frozen=True)
class Words:
    """One annotated word: when it is said, what it says, and whether it ends a sentence."""

    start: float
    end: float
    text: str
    ends_sentence: bool


@dataclass(frozen=True)
class Channel:
    """One speaker of one meeting: which headset to read and which file describes it."""

    meeting: str
    agent: str
    speaker: str

    @property
    def head_number(self) -> int:
        return head_number(self.agent)


@dataclass(frozen=True)
class Utterance:
    """One clip to write: when it is, what was said, and where it belongs."""

    meeting: str
    agent: str
    speaker: str
    index: int
    start: float
    end: float
    text: str

    @property
    def id(self) -> str:
        """A stable name for the clip, unique inside the corpus."""
        return f"{self.meeting}.{self.agent}.{self.index:04d}"


def fetch_ami_fr_speakers_en_train(
    name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool
) -> int:
    """Prepare the French-L1 speakers of AMI up to the cap, returning the clip count."""
    annotations = download(ANNOTATIONS_URL, cache_dir / name / ANNOTATIONS_NAME)
    channels = read_channels(annotations)
    builder = ManifestBuilder(dataset_dir, verbose=verbose)
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

    with zipfile.ZipFile(annotations) as archive:
        for channel in channels:
            if take.full():
                break
            # A headset file holds a whole meeting: a speaker that already holds
            # its share of the cap is not downloaded, and the words of one
            # speaker are read once.
            if take.group_full(channel.speaker):
                continue
            utterances = utterances_of(archive, channel)
            audio = load_audio(_headset_file(name, cache_dir, channel))
            for utterance in utterances:
                if take.full() or take.group_full(channel.speaker):
                    break
                take.offer(
                    utterance.id,
                    _slice(audio, utterance.start, utterance.end),
                    text=utterance.text,
                    group=utterance.speaker,
                )
    if verbose:
        print(f"  {take.report()}")
    builder.write()
    return take.kept


def read_channels(archive: Path) -> list[Channel]:
    """Every French-L1 speaker of the corpus, in the corpus's own meeting order.

    The pair (meeting, speaker letter) is what names a headset recording and a
    word file, and the participant id is what a clip is grouped by. The two
    resources are read together because neither alone says who spoke English as a
    second language.
    """
    french = _french_speakers(archive)
    channels = [
        Channel(meeting=meeting, agent=agent, speaker=speaker)
        for meeting, agent, speaker in _speakers(archive)
        if speaker in french
    ]
    if not channels:
        raise DownloadError(f"{archive.name} holds no speaker of native language {NATIVE_LANGUAGE}")
    return sorted(channels, key=lambda channel: (channel.meeting, channel.agent))


def utterances_of(
    archive: zipfile.ZipFile, channel: Channel, *, max_gap: float = GAP_SECONDS
) -> list[Utterance]:
    """Cut one speaker's words into utterances of the length a clip may have.

    A word file holds the words of one speaker of one meeting, in order, with a
    ``punc`` flag on the punctuation marks. A clip ends at sentence-final
    punctuation, at a silence longer than ``max_gap``, or where the next word
    would push it past :data:`audiolab.data.train.MAX_SECONDS`; the length filter
    of :class:`Take` drops what is still too short or too long afterwards.
    """
    words = read_words(archive, channel)
    utterances: list[Utterance] = []
    current: list[Words] = []

    def close() -> None:
        if not current:
            return
        text = _text_of(current)
        if text:
            utterances.append(
                Utterance(
                    meeting=channel.meeting,
                    agent=channel.agent,
                    speaker=channel.speaker,
                    index=len(utterances),
                    start=current[0].start,
                    end=current[-1].end,
                    text=text,
                )
            )
        current.clear()

    for word in words:
        if current and (
            word.start - current[-1].end > max_gap or word.end - current[0].start > MAX_SECONDS
        ):
            close()
        current.append(word)
        if word.ends_sentence:
            close()
    close()
    return utterances


def read_words(archive: zipfile.ZipFile, channel: Channel) -> list[Words]:
    """The spoken words of one speaker of one meeting, without the non-speech elements.

    A word file also holds ``vocalsound`` (a laugh or a cough), ``disfmarker``
    (the annotators' mark of a disfluency) and ``gap`` (a stretch nobody
    transcribed). They carry no text to train on and are left out here, so they
    neither join two words nor end an utterance. ``uh`` and ``um`` are words the
    annotators did write, and they stay: they are what the speaker said.
    """
    member = f"{WORDS_DIRECTORY}/{channel.meeting}.{channel.agent}.words.xml"
    try:
        data = archive.read(member)
    except KeyError:
        raise DownloadError(f"{archive.filename}: holds no {member}") from None
    root = ET.fromstring(data)
    words = []
    for element in root:
        if _local(element.tag) != "w":
            continue
        text = (element.text or "").strip()
        if not text:
            continue
        ends = element.attrib.get("punc") == "true" and text in SENTENCE_END
        words.append(
            Words(
                start=float(element.attrib["starttime"]),
                end=float(element.attrib["endtime"]),
                text=text,
                ends_sentence=ends,
            )
        )
    return words


def head_number(agent: str) -> int:
    """The headset channel of a speaker letter (``A`` is ``Headset-0``)."""
    if agent not in HEAD_NUMBERS:
        raise DownloadError(f"AMI has no headset for speaker letter {agent!r}")
    return HEAD_NUMBERS[agent]


def _headset_file(name: str, cache_dir: Path, channel: Channel) -> Path:
    """Download the headset recording of one speaker, or take it from the cache."""
    recording = f"{channel.meeting}.Headset-{channel.head_number}.wav"
    return download(
        f"{AUDIO_URL}/{channel.meeting}/audio/{recording}",
        cache_dir / name / "audio" / recording,
    )


def _slice(audio: Audio, start: float, end: float) -> Audio:
    """The samples of one utterance, taken from the meeting's headset recording."""
    first = max(int(round(start * SAMPLE_RATE)), 0)
    last = min(int(round(end * SAMPLE_RATE)), len(audio.samples))
    return Audio(samples=audio.samples[first:last], sample_rate=SAMPLE_RATE)


def _text_of(words: list[Words]) -> str:
    """Join words and punctuation into one sentence, as an annotator wrote it."""
    text = ""
    for word in words:
        if PUNCTUATION.match(word.text) and text:
            text += word.text
        else:
            text = f"{text} {word.text}".strip()
    return text


def _french_speakers(archive: Path) -> set[str]:
    """The participant ids whose native language is French."""
    root = _resource(archive, PARTICIPANTS)
    return {
        _identifier(element)
        for element in root
        if _local(element.tag) == "participant"
        and element.attrib.get("native_language") == NATIVE_LANGUAGE
    }


def _speakers(archive: Path) -> list[tuple[str, str, str]]:
    """Every ``(meeting, speaker letter, participant id)`` of the corpus."""
    root = _resource(archive, MEETINGS)
    speakers = []
    for meeting in root:
        for element in meeting:
            if _local(element.tag) != "speaker":
                continue
            speakers.append(
                (
                    meeting.attrib["observation"],
                    element.attrib["nxt_agent"],
                    element.attrib["global_name"],
                )
            )
    return speakers


def _local(tag: str) -> str:
    """The name of an element without its namespace.

    The AMI resources declare the NITE namespace on their root element, so the
    root is ``{http://nite.sourceforge.net/}root`` while the elements under it
    (``participant``, ``meeting``, ``w``) carry no prefix. The identifiers and
    cross-references are namespaced attributes (``nite:id``) instead.
    """
    return tag.rpartition("}")[2]


def _identifier(element) -> str:
    """The ``nite:id`` of an element."""
    return element.attrib.get(f"{NITE}id") or element.attrib["id"]


def _resource(archive: Path, member: str):
    """Read one XML resource of the annotations archive."""
    with zipfile.ZipFile(archive) as handle:
        try:
            data = handle.read(member)
        except KeyError:
            raise DownloadError(f"{archive.name}: holds no {member}") from None
    return ET.fromstring(data)
