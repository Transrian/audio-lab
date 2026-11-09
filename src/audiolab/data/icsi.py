"""ICSI (``icsi-en-train``): English research meetings, one close mic per participant.

The ICSI Meeting Corpus is 75 meetings recorded at the International Computer
Science Institute in Berkeley between 2000 and 2002: research meetings that
would have happened anyway, with three to nine close-talking microphones each
(469 of them over the whole corpus, for 52 participants) and a word-by-word
transcription made by hand. The corpus and its annotations are CC-BY-4.0.

The transcription used here is the corpus's own **MRT** release, not the NXT
annotations built from it, because a segment of an MRT file already carries the
start and end time of one utterance by one participant, while the NXT word
layer rebuilds those times from the words themselves. A word layer leaves a
word unplaced when the annotators placed it only at its other end (``We`` with a
start and no end, ``'re`` with an end and no start), so its boundaries would
have to be guessed; MRT gives the boundaries as published numbers. An MRT file
also names each participant's own microphone in its preamble
(``<Participant Name="me013" Channel="chan0"/>``), which is the channel a clip
is sliced from, so nothing has to be inferred from a file name.

The audio is one SPHERE file per channel, served by the corpus's own download
page over plain HTTP without a login
(``ICSIsignals/SPH/<meeting>/<channel>.sph``). Its samples are
shorten-compressed, which the readers this project uses do not decode
(libsndfile refuses the file), so the preparation decodes one with ``ffmpeg`` —
the one place in the repository that runs an external program. The corpus
publishes no checksum of its own, but it writes the count and the sum of every
channel's samples into the file's own header, and the preparation checks the
samples it decoded against both: a decoder that quietly dropped, reordered or
rescaled samples stops the preparation instead of writing audio the transcript
times do not describe.

A segment is one clip: its own start and end time, sliced from the speaker's own
close microphone. A segment whose words were heard somewhere other than that
microphone (``CloseMic="false"``, 1 413 of the corpus's 129 813 segments) is
left out, because the annotation says the close microphone was not what carried
the speech. Non-speech markup (``VocalSound``, ``NonVocalSound``, ``Comment``,
``Pause``) is dropped — it is not something to transcribe — while ``Emphasis``,
``Pronounce``, ``Foreign`` and ``Uncertain`` hold words the speaker did say and
are kept. The
text is cased and punctuated as the annotators wrote it, with their spellings
(``O_K``, ``Mm-hmm``) and their marks of a false start (``how much w- how
easy``).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import soundfile as sf

from audiolab.core.audio import Audio, load_audio
from audiolab.core.stop import raise_if_stopped
from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError, download
from audiolab.data.meeting import Chunk, merged, spelled_out
from audiolab.data.train import (
    CLIP_CAP_PER_GROUP,
    TRAIN_CLIP_CAP,
    Take,
    plan_clips,
    write_clips,
)

TRANSCRIPTS_URL = (
    "https://groups.inf.ed.ac.uk/ami/ICSICorpusAnnotations/ICSI_original_transcripts.zip"
)
TRANSCRIPTS_NAME = "ICSI_original_transcripts.zip"
# The corpus publishes no checksum for its files, so the annotation archive this
# source is prepared from is pinned here, like the AMI annotations.
PINNED_SHA256 = {
    TRANSCRIPTS_NAME: "9db98ebdd61a085b4f45f24f46e9c9f6c6f26c3c9cd41643a3b9e74362c85de7",
}
# Where the corpus serves its signals: one directory per meeting, one SPH file
# per microphone channel. No login, no checksum and no listing; the URL is the
# meeting and the channel, both of which the annotations name.
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/ICSIsignals/SPH"
TRANSCRIPTS_DIRECTORY = "transcripts"
# The archive also holds ``transcripts/preambles.mrt``: the 75 preambles of the
# corpus in one file, with no segment to prepare.
MEETINGS_FILE = "transcripts/preambles.mrt"
LICENSE = "CC-BY-4.0"
LANGUAGE = "en"
SPLIT = "mrt"
# The segment elements that carry no words to transcribe: a laugh, a noise, an
# annotator's remark, a silence. Emphasis, Pronounce, Foreign and Uncertain do
# carry words and are kept.
NON_SPEECH = frozenset({"VocalSound", "NonVocalSound", "Comment", "Pause"})
# Where the annotators could not make the words out they wrote this mark
# (the corpus's own guide, ``doc/trans_guide.txt``): the audio is there but the
# reference is not, and a clip whose text is missing a word is not a reference.
UNINTELLIGIBLE = "@@"
WHITESPACE = re.compile(r"\s+")
# A SPHERE header is plain text in the first kilobyte of the file.
SPHERE_HEADER_BYTES = 1024
SAMPLE_COUNT = re.compile(r"^sample_count -i (\d+)$", re.MULTILINE)
SAMPLE_CHECKSUM = re.compile(r"^sample_checksum -i (\d+)$", re.MULTILINE)
# The sum the SPHERE format asks a checksum to be: its own modulo of 2^16.
CHECKSUM_MODULUS = 1 << 16


def fetch_icsi_en_train(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare ICSI up to the cap, returning the number of clips written."""
    transcripts = download(
        TRANSCRIPTS_URL,
        cache_dir / name / TRANSCRIPTS_NAME,
        sha256=PINNED_SHA256.get(TRANSCRIPTS_NAME, ""),
    )
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

    # The annotation is read first and the audio is downloaded afterwards,
    # because the cap and the per-speaker share decide which stretches become
    # clips and that decision can be taken from the transcript alone. Planning
    # it first downloads the 105 channels a full preparation slices instead of
    # the 469 the corpus has.
    segments = merged(read_chunks(transcripts))
    planned = plan_clips(take, segments)
    by_channel: dict[tuple[str, str], list[Chunk]] = {}
    for segment in planned:
        by_channel.setdefault((segment.meeting, segment.recording), []).append(segment)
    if verbose:
        print(f"  {len(by_channel)} channels of audio to decode for {len(planned)} clips")
    for (meeting, channel), rows in sorted(by_channel.items()):
        write_clips(take, channel, partial(load_recording, name, cache_dir, meeting, channel), rows)
    if verbose:
        print(f"  {take.report()}")
    builder.write()
    return take.kept


def read_chunks(transcripts: Path) -> list[Chunk | None]:
    """The corpus's annotation, in the corpus's own order, with what it drops marked.

    The archive lists its meetings in the order the corpus's download page lists
    them, which is the order this reads: an MRT file per meeting, and inside it
    the segments in the order the annotators wrote them, which is time order.
    A segment that is not speech this source can use — a laugh the annotators
    marked, words of a participant the preamble gives no microphone to, speech
    their own microphone did not carry — is ``None``, so that the clip builder
    knows a clip may not be built across it.
    """
    with zipfile.ZipFile(transcripts) as archive:
        members = [
            member
            for member in archive.namelist()
            if member.startswith(f"{TRANSCRIPTS_DIRECTORY}/")
            and member.endswith(".mrt")
            and member != MEETINGS_FILE
        ]
        if not members:
            raise DownloadError(f"{transcripts.name}: holds no {TRANSCRIPTS_DIRECTORY}/*.mrt")
        segments: list[Chunk | None] = []
        for member in members:
            segments.extend(_meeting_chunks(archive, member))
    if not any(segments):
        raise DownloadError(f"{transcripts.name}: holds no transcribed segment")
    return segments


def recording_url(source: str) -> str:
    """Where the corpus serves one SPH file."""
    return f"{AUDIO_URL}/{source}"


def load_recording(name: str, cache_dir: Path, meeting: str, channel: str) -> Audio:
    """Decode one channel of one meeting, or take it from the cache.

    ``channel`` is the channel as the transcript names it (``chan0``): the file
    the corpus serves is that channel with ``.sph`` on it, and the decoded WAV is
    kept next to it, so a preparation that reads the same channel twice decodes
    it once and the two files together are what the channel was prepared from.
    The decoded file is written under a temporary name and moved, like a
    download, so an interrupted decode leaves nothing that looks finished.
    """
    directory = cache_dir / name / "audio" / meeting
    decoded = directory / f"{channel}.wav"
    if decoded.is_file():
        return load_audio(decoded)
    source = download(recording_url(f"{meeting}/{channel}.sph"), directory / f"{channel}.sph")
    _decode(source, decoded)
    return load_audio(decoded)


def _decode(source: Path, destination: Path) -> None:
    """Turn one shorten-compressed SPH file into the 16 kHz mono WAV of a clip.

    ``ffmpeg`` is the only decoder on hand for shortened SPHERE audio; it is
    asked for the format every prepared dataset uses, so a source that changed
    its sample rate cannot pass silently. The samples it produced are checked
    against the count and the checksum the corpus itself wrote into the header.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise DownloadError(
            "ffmpeg is needed to decode the shorten-compressed SPHERE audio of ICSI: "
            "install it, or prepare a source that needs no external decoder"
        )
    staged = destination.with_name(f"{destination.name}.part")
    result = subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(staged),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        staged.unlink(missing_ok=True)
        # A Ctrl-C in the terminal reaches ffmpeg too: an ffmpeg a stop
        # interrupted is not a file it cannot decode.
        raise_if_stopped(f"decoding of {source.name} stopped")
        raise DownloadError(
            f"ffmpeg cannot decode {source.name}: {result.stderr.decode(errors='replace').strip()}"
        )
    samples, _ = sf.read(str(staged), dtype="int16", always_2d=False)
    try:
        verify_samples(source, samples)
    except DownloadError:
        staged.unlink(missing_ok=True)
        raise
    staged.replace(destination)


def verify_samples(source: Path, samples: np.ndarray) -> None:
    """Refuse samples that are not what the recording's own header declares.

    Every SPH file of the corpus carries the count of its samples and the sum of
    them, and the sum tells apart a file whose samples were changed or decoded
    from a file whose samples are the recording: a decoder that quietly dropped
    or repeated samples would misalign every clip on that channel, and a file
    that changed since the times were made is not the file the annotators read.
    The check reads the header alone, so it is exercised without a decoder.
    """
    expected_count = int(_sphere_field(source, SAMPLE_COUNT))
    if len(samples) != expected_count:
        raise DownloadError(
            f"{source.name} decoded to {len(samples)} samples but its own header says "
            f"{expected_count}: the file or the decoder is wrong"
        )
    checksum = int(samples.astype("int64").sum() % CHECKSUM_MODULUS)
    if checksum != int(_sphere_field(source, SAMPLE_CHECKSUM)):
        raise DownloadError(
            f"{source.name} decoded to samples that do not match its own checksum: "
            "the decode is not the recording the transcript times refer to"
        )


def _sphere_field(path: Path, pattern: re.Pattern[str]) -> str:
    """One field of a SPHERE header, which is plain text in its first kilobyte."""
    with path.open("rb") as handle:
        header = handle.read(SPHERE_HEADER_BYTES).decode("latin-1")
    found = pattern.search(header)
    if found is None:
        raise DownloadError(f"{path.name} is not a SPHERE file: it has no {pattern.pattern!r}")
    return found.group(1)


def _meeting_chunks(archive: zipfile.ZipFile, member: str) -> list[Chunk | None]:
    """One meeting's segments, in the order the file lists them."""
    try:
        root = ET.fromstring(archive.read(member))
    except ET.ParseError as error:
        raise DownloadError(f"{archive.filename}: {member} is not readable XML ({error})") from None
    meeting = root.attrib.get("Session")
    if not meeting:
        raise DownloadError(f"{archive.filename}: {member} names no Session")
    # A participant without a channel is one the corpus says was unmiked: they
    # are heard on other people's microphones, and every segment of theirs says
    # so (``CloseMic="false"``, 219 of the corpus's segments), so none of them
    # reaches the lookup below.
    channels = {
        participant.attrib["Name"]: participant.attrib["Channel"]
        for participant in root.iter("Participant")
        if "Channel" in participant.attrib
    }
    segments: list[Chunk | None] = []
    for position, element in enumerate(root.iter("Segment")):
        speaker = element.attrib.get("Participant")
        # A segment without a participant is a stretch of room noise the
        # annotators marked, and one with CloseMic="false" is speech the
        # speaker's own close microphone did not carry; neither is a clip.
        if not speaker or element.attrib.get("CloseMic") == "false":
            segments.append(None)
            continue
        # A digit task is the annotators reading out digits one by one; the
        # corpus marks it because transcribing it is a task of its own, and what
        # the speaker says is not the sentence the rest of the corpus holds.
        if element.attrib.get("DigitTask") == "true":
            segments.append(None)
            continue
        # A segment whose own Channel differs from the one the preamble gives
        # its speaker is the same speech the corpus re-sliced from another
        # microphone: taking it would prepare that speech twice.
        own_channel = element.attrib.get("Channel")
        if own_channel and speaker in channels and own_channel != channels[speaker]:
            segments.append(None)
            continue
        text = segment_text(element)
        if not text:
            segments.append(None)
            continue
        if UNINTELLIGIBLE in text:
            segments.append(None)
            continue
        if speaker not in channels:
            raise DownloadError(
                f"{archive.filename}: {member} has {speaker} speaking on their own "
                "microphone, which the preamble does not give them"
            )
        try:
            start, end = float(element.attrib["StartTime"]), float(element.attrib["EndTime"])
        except KeyError:
            raise DownloadError(
                f"{archive.filename}: {member} holds a segment without a time: {text!r}"
            ) from None
        segments.append(
            Chunk(
                meeting=meeting,
                speaker=speaker,
                recording=channels[speaker],
                # The number of the segment in the file, so that the name of a
                # clip says which segment of the corpus it starts at and does
                # not move when a filter above the loop changes.
                index=position,
                start=start,
                end=end,
                text=text,
            )
        )
    return segments


def segment_text(element: ET.Element) -> str:
    """What one segment says, with the annotation markup taken out.

    A segment holds its words as text with annotation elements between them (a
    pause, a laugh, an emphasised word, a comment by the annotator). The
    elements that carry no words are dropped whole; the ones that do — an
    emphasis, a pronunciation, a foreign word, an uncertain guess — contribute
    their text, because that is what the speaker said. The words around the
    markup are joined with the whitespace the file left out, and a token the
    annotators wrote letter by letter is read as the word it spells (`O_K` is
    "OK", `S_R_I` is "SRI", a lone `L_` is "L"), the way the AMI source reads the
    words of its own annotation.
    """
    pieces = [piece.strip() for piece in _pieces(element) if piece.strip()]
    return spelled_out(WHITESPACE.sub(" ", " ".join(pieces)).strip())


def _pieces(element: ET.Element) -> list[str]:
    """The words of one element: its text, its children's, and the tails between them.

    The markup nests — an emphasis holds a pronunciation, which holds the word —
    so the walk goes all the way down. An element that carries no words is
    dropped with everything inside it, since a laugh holds no words, but its tail
    is the file's own punctuation after it and belongs to the segment.
    """
    pieces = [element.text or ""]
    for child in element:
        if child.tag not in NON_SPEECH:
            pieces.extend(_pieces(child))
        pieces.append(child.tail or "")
    return pieces
