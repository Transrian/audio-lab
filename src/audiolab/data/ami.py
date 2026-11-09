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
per participant, published on the corpus's own mirror under
``amicorpus/<meeting>/audio/``, without a login: a preparation downloads only the
headset of the meetings it takes, and ``meetings.xml`` says which headset channel
a speaker letter is.

A manual word file holds words, punctuation marks (``punc="true"``) and the
elements ``vocalsound``, ``disfmarker`` and ``gap``, which mark a laugh, a
disfluency and a stretch nobody transcribed. Only the words are read here,
punctuation is attached to the word before it, and a clip ends at sentence-final
punctuation, after a silence of more than one second, at a ``gap`` — the audio
under a gap has no text to match, so a clip never spans one — or when it would
last longer than 20 s. A word written letter by letter with underscores between
the letters (``T_V_``, ``L_C_D_``) is written as the word it spells, because that
is what the speaker said: the text is otherwise cased and punctuated as the
annotators wrote it.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

from audiolab.core.audio import load_audio
from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError, download
from audiolab.data.meeting import GAP_SECONDS, SENTENCE_END, spelled_out
from audiolab.data.train import (
    CLIP_CAP_PER_GROUP,
    MAX_SECONDS,
    TRAIN_CLIP_CAP,
    Take,
)

ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
ANNOTATIONS_NAME = "ami_public_manual_1.6.2.zip"
# The corpus's own mirror, which serves the individual headset WAV files without
# a login.
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
# The corpus publishes no checksum for its files, so the two it is prepared from
# are pinned here: the annotations release, and each headset recording a
# preparation reads. It asks for one recording per speaker and skips a speaker
# that already holds its share of the cap, which is what makes the list finite:
# 42 of the corpus's 69 French-L1 speaker recordings. A recording that is not in
# the map stops the preparation instead of being taken unverified.
PINNED_SHA256 = {
    ANNOTATIONS_NAME: "b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d",
}
PINNED_RECORDINGS = {
    "ES2007a.Headset-1.wav": "45dc35f3e94221d997c9776bdbce2501e213d9ace3d34d4e51e51d063efbf29c",
    "ES2007b.Headset-1.wav": "80eaa67c9882be6209119fa1b29d99447a97656ecdf737a60f02904d5eac16a7",
    "ES2007c.Headset-1.wav": "01a51de77d7a4938c1bc03ed10235db57a997d5f4b79e849d3d9fbf3d11d460f",
    "ES2007d.Headset-1.wav": "922a8fb286d2ec7ad4c0f14c16c9f2a33d739acc068a9446cd1687cc6b0a669d",
    "IB4001.Headset-0.wav": "e50a6137e227fedeedb1cea6846d42b99d812c5e83229c1ed9c71ecf026f9a26",
    "IB4002.Headset-0.wav": "effee4a099ea594eabb400c662b971a5f9b7a24e831622bd4187b7adbf4beebd",
    "IB4003.Headset-3.wav": "fdccd187bf9436c05a398d52c3eb495f9492fc3e492a81312847de0c519035bd",
    "IB4004.Headset-3.wav": "d25740092b4f9ccb390dba49126f7ae1d715b772c2fec08f0bf17898052c3278",
    "IB4010.Headset-3.wav": "f3f81f08b84cb324a17fe4c194c7b0e402c9917f5d7c2fb2f285e12d779c44aa",
    "IN1001.Headset-0.wav": "a3a08ff0dfd08542947ff3a44e58f9c3ff916e2c6597e9ac46d89c6b7a3b5185",
    "IN1001.Headset-1.wav": "ac2abef5871af2303743f8d00d2852b78e983620d1e09a2148e997681fa0f2fd",
    "IN1001.Headset-2.wav": "42b17557136d84b3dfc48c31167141f3c49f1e84e80a3ba59e7fdf9cf5b51eee",
    "IN1005.Headset-0.wav": "1e63cf56c5757e8ead7847ce8f1aad3a9a7d99028d96947316e000bf8b80d874",
    "IN1005.Headset-2.wav": "bfa148592eeabcfe2557a8779efafe8c6d6f200ef1d7e4c1c716f1d368ec3cc4",
    "IN1005.Headset-3.wav": "7ee6280100513b6edc99e9c0b02ec4718f2868356446c44b8c53816d3bdc408e",
    "IN1008.Headset-0.wav": "5f613556a90c4c8bb60d6a0d0437eaaf6dbb1a3afd29bb8b6357709aff6c1b66",
    "IN1008.Headset-2.wav": "b68a8579672c24d438381a31133a75e593b914b98375a538908c892fbe75317e",
    "IN1008.Headset-3.wav": "f0d16196bd3f435ea97231d2d29b81fa9cfd36628dcc818628611bd689ace8d5",
    "IN1009.Headset-0.wav": "a4b802db2336bccf62eb939cb4ffa3be055d9a5c50aee136526cc84161785b4c",
    "IN1014.Headset-1.wav": "2129c80927875210cbbfced326c7cae0a7da2300afa8893af00bbb48c4ba0fac",
    "IS1001a.Headset-0.wav": "f460bafa57daa45fe195caea937053a1b6a251067ace08dd149a397351c9105c",
    "IS1001a.Headset-3.wav": "854e166be176e8d3ecd16ccc6b9272baada7d64786becc805e9c8685d720a775",
    "IS1001b.Headset-3.wav": "4b1a082a83d547ca5ecfa5d320e45a158ead491a7d45b33e28662f7be9b6cdf6",
    "IS1001c.Headset-3.wav": "771dddf8d0ba86096d61a06d2e24f6a91d6a2a05c44fd72c99ee996e943d5544",
    "IS1001d.Headset-3.wav": "547c260a7f7d74bee9c39796b6a2dbcb38f2e21a010ef696f279c60661260e71",
    "IS1003a.Headset-0.wav": "5a633e76470335e5e7d830980f00269f1dee28ceecf38a07257b1b7c1e5bd8d6",
    "IS1003a.Headset-3.wav": "f9e62fd70eaec13016096d89c1e383ca5a25047930b1916b1cfdfc6124c5bb41",
    "IS1003b.Headset-0.wav": "e61400523d49f8416287e225233a23f425a92f6c2d36e92fe9fddc8580f12cc1",
    "IS1003b.Headset-2.wav": "3576772733b55c1fdbfb7b81f47779f1c60b37847021671de183732ffb7117a7",
    "IS1003c.Headset-0.wav": "3f1bbd1dfb01583ac8dd6f1b954932bedc53c3c1e96ae7d2ab53defd807ca7cc",
    "IS1003c.Headset-3.wav": "b60e177eabd9c5a40e14d3928493297fba7017435db50dc6ec5ef44fcf6c0feb",
    "IS1004a.Headset-0.wav": "71e76cde3cbb83c6f4a36c3c99e7310a00e1ec6c2d23f05f04b4a27a74b4b886",
    "IS1004b.Headset-0.wav": "9c773b1024b522c11b4d2e529fa2fb284753cbf9d72e2cc9b019e314be5d5787",
    "IS1005a.Headset-0.wav": "ffb9ec65e75871a5293e893226558a49cf6e562645f777964fed0d1f0dc078cf",
    "IS1005b.Headset-0.wav": "e709ef9ef123171d960cf0e1d9a0200abfd0203676f91150bd43f3a6af042706",
    "IS1006a.Headset-0.wav": "fe9a82406b9d2b4557fd0b97203bae326ae5048798161b90d407a40f2233c9a7",
    "IS1006a.Headset-3.wav": "41e53d029f2a3dfcccfa474a71c3a112b021eecf4d28cb55d9f7a66b3fd14181",
    "IS1006b.Headset-0.wav": "7676907309033d28e91374d4b9a3fcd2475e5f9257942df9e54ed3daf0998544",
    "IS1006b.Headset-3.wav": "37d1ec078b8fd3b1d231d01351199fc4ec85f506da53fe69372bbe822962d408",
    "IS1007a.Headset-2.wav": "e25f06439bb685ba70aa007d57fe04d6d6d9a44574226e2ab125a73517b60765",
    "IS1007b.Headset-2.wav": "81b4dd4ac6a3a57716c219fc36b46617b20a858205db0f4a9395a0a92d3ce9f8",
    "IS1007c.Headset-2.wav": "237560f42466b4083fa3d2669bd60471efc41eaba3d50e94d450d152628b7f8d",
}
PARTICIPANTS = "corpusResources/participants.xml"
MEETINGS = "corpusResources/meetings.xml"
WORDS_DIRECTORY = "words"
NATIVE_LANGUAGE = "French"
LICENSE = "CC-BY-4.0"
LANGUAGE = "en"
SPLIT = "manual"
# A punctuation mark is written as its own word and belongs to the word before.
PUNCTUATION = re.compile(r"^[.,;:!?'\u2019-]+$")
NITE = "{http://nite.sourceforge.net/}"


@dataclass(frozen=True)
class Word:
    """One annotated word: when it is said, what it says, and whether it ends a sentence."""

    start: float
    end: float
    text: str
    ends_sentence: bool


@dataclass(frozen=True)
class Untranscribed:
    """A ``gap``: a stretch the annotators left without text.

    It is a marker, not a word: the words around it are read in the order the
    file lists them, which is what it does to a clip, so its own times are not
    read — 41 of the corpus's gaps have no end time and 7 have no time at all.
    """


@dataclass(frozen=True)
class Channel:
    """One speaker of one meeting: which headset to read and which file describes it."""

    meeting: str
    agent: str
    speaker: str
    # The headset channel of this speaker, read from ``meetings.xml``.
    channel: int


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
    annotations = download(
        ANNOTATIONS_URL,
        cache_dir / name / ANNOTATIONS_NAME,
        sha256=PINNED_SHA256.get(ANNOTATIONS_NAME, ""),
    )
    channels = read_channels(annotations)
    builder = ManifestBuilder(dataset_dir, verbose=verbose, total=TRAIN_CLIP_CAP)
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
                    audio.cut(utterance.start, utterance.end),
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
    word file, the participant id is what a clip is grouped by, and ``channel``
    is which headset of the meeting carries the recording. The two resources are
    read together because neither alone says who spoke English as a second
    language.
    """
    french = _french_speakers(archive)
    channels = [
        Channel(meeting=meeting, agent=agent, speaker=speaker, channel=channel)
        for meeting, agent, speaker, channel in _speakers(archive)
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
    punctuation, at a silence longer than ``max_gap``, at a ``gap`` — the audio
    under it has no text, so a clip that spanned it would be labelled with words
    that are not all it says — or where the next word would push it past
    :data:`audiolab.data.train.MAX_SECONDS`; the length filter of :class:`Take`
    drops what is still too short or too long afterwards.
    """
    words = read_words(archive, channel)
    utterances: list[Utterance] = []
    current: list[Word] = []

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
        if isinstance(word, Untranscribed):
            close()
            continue
        if current and (
            word.start - current[-1].end > max_gap or word.end - current[0].start > MAX_SECONDS
        ):
            close()
        current.append(word)
        if word.ends_sentence:
            close()
    close()
    return utterances


def read_words(archive: zipfile.ZipFile, channel: Channel) -> list[Word | Untranscribed]:
    """The spoken words of one speaker of one meeting, and the gaps between them.

    A word file also holds ``vocalsound`` (a laugh or a cough) and ``disfmarker``
    (the annotators' mark of a disfluency). Neither carries text to train on, so
    both are left out here and they neither join two words nor end an utterance.
    ``uh`` and ``um`` are words the annotators did write, and they stay: they are
    what the speaker said. A ``gap`` is returned, because the audio under it is
    speech nobody transcribed and a clip must not span it.
    """
    member = f"{WORDS_DIRECTORY}/{channel.meeting}.{channel.agent}.words.xml"
    try:
        data = archive.read(member)
    except KeyError:
        raise DownloadError(f"{archive.filename}: holds no {member}") from None
    root = ET.fromstring(data)
    words: list[Word | Untranscribed] = []
    for element in root:
        name = _local(element.tag)
        if name == "gap":
            words.append(Untranscribed())
            continue
        if name != "w":
            continue
        text = spelled_out((element.text or "").strip())
        if not text:
            continue
        try:
            start, end = float(element.attrib["starttime"]), float(element.attrib["endtime"])
        except KeyError:
            raise DownloadError(
                f"{archive.filename}: {member} holds a word without a time: {text!r}"
            ) from None
        ends = element.attrib.get("punc") == "true" and text in SENTENCE_END
        words.append(
            Word(
                start=start,
                end=end,
                text=text,
                ends_sentence=ends,
            )
        )
    return words


def _headset_file(name: str, cache_dir: Path, channel: Channel) -> Path:
    """Download the headset recording of one speaker, or take it from the cache."""
    recording = f"{channel.meeting}.Headset-{channel.channel}.wav"
    pinned = PINNED_RECORDINGS.get(recording)
    if pinned is None:
        raise DownloadError(
            f"{recording} is not one of the recordings this source was prepared from and "
            "has no pinned sha256: the corpus changed, so add the recording to "
            "PINNED_RECORDINGS in src/audiolab/data/ami.py and prepare the source again"
        )
    return download(
        f"{AUDIO_URL}/{channel.meeting}/audio/{recording}",
        cache_dir / name / "audio" / recording,
        sha256=pinned,
    )


def _text_of(words: list[Word]) -> str:
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
        element.attrib[f"{NITE}id"]
        for element in root
        if _local(element.tag) == "participant"
        and element.attrib.get("native_language") == NATIVE_LANGUAGE
    }


def _speakers(archive: Path) -> list[tuple[str, str, str, int]]:
    """Every ``(meeting, letter, participant id, headset channel)`` of the corpus."""
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
                    int(element.attrib["channel"]),
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


def _resource(archive: Path, member: str):
    """Read one XML resource of the annotations archive."""
    with zipfile.ZipFile(archive) as handle:
        try:
            data = handle.read(member)
        except KeyError:
            raise DownloadError(f"{archive.name}: holds no {member}") from None
    return ET.fromstring(data)
