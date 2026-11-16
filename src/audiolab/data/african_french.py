"""African Accented French (``african-accented-fr-train``): read and conversational speech.

OpenSLR SLR57 is Apache-2.0 speech recorded in Yaoundé and Libreville from
French speakers of African accents: prompted sentences, answers and short
conversations. The corpus is one 1.7 GiB tar archive holding the four parts it
publishes (``train``, ``dev``, ``devtest``, ``test``) with their transcripts;
the train part and the test part are prepared here, and ``dev`` and ``devtest``
are not read. The whole archive has to be downloaded either way, because the
corpus publishes no smaller unit.

Three transcript files describe the train part, one per recording style, and
they name their files in three ways: a bare name, a name with ``.tdf`` and an
absolute path of the machine the corpus was built on. All three give the same
key once the directory and the extension are removed, which is what
:func:`_key` does. The corpus has no transcript for 2 934 of its 14 394 train
recordings; those are left out.

The text is lowercase and unpunctuated. Its apostrophes are written as separate
words (``d' accord``, ``aujourd' hui``, ``grand ' mère``), which is a defect of
the transcripts rather than of the language: the preparation removes the spaces
around an apostrophe that follows a letter and collapses the runs of whitespace
the Yaoundé part leaves where punctuation was deleted. Eight lines of
``transcripts/train/yaounde/fn_text.txt`` are the prompting text (``etes-vous
sûr que c' est lui / elle``): they offer the reader two answers separated by a
slash, so the transcript cannot say which one was read and the recordings are
dropped rather than labelled with text that does not match the audio. Casing and punctuation
are not restored: a source that has neither teaches the base model to drop what
it writes, and putting them back would take another model. The card in
`docs/stt/datasets.md` records both the cleanup and that confound.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import partial
from pathlib import Path

from audiolab.data.build import ManifestBuilder, require_clips
from audiolab.data.download import DownloadError, audio_from_archive, download, open_archive
from audiolab.data.store import MANIFEST_NAME, read_manifest
from audiolab.data.train import CLIP_CAP_PER_GROUP, TRAIN_CLIP_CAP, Take, evaluation_groups

ARCHIVE_URL = "https://www.openslr.org/resources/57/African_Accented_French.tar.gz"
ARCHIVE_NAME = "African_Accented_French.tar.gz"
# OpenSLR publishes this file without a checksum, so the sha256 seen when the
# source was prepared is recorded here and checked on every fresh download.
PINNED_SHA256 = {
    "African_Accented_French.tar.gz":
        "3d9edcd82c51a56686ac40f5106dee2ed5e5cb5a6d6b7fb0a9d6e6e19967187b",
}  # fmt: skip
LICENSE = "Apache-2.0"
LANGUAGE = "fr"
SPLIT = "train"
ROOT = "African_Accented_French"
SPEECH = f"{ROOT}/speech/{SPLIT}"
TRANSCRIPTS = (
    f"{ROOT}/transcripts/{SPLIT}/ca16_read/conditioned.txt",
    f"{ROOT}/transcripts/{SPLIT}/ca16_conv/transcripts.txt",
    f"{ROOT}/transcripts/{SPLIT}/yaounde/fn_text.txt",
)
# The evaluation part. The corpus publishes one transcript file for it, the
# prompted sentences of its CA16 speakers, and one recording directory per
# speaker. The two parts name the CA16 speakers in two formats over one shared
# numbering: the train part's directory is the bare number (``012``) and the
# test part's is ``<country>_<m|f>_<number>`` (``cameroon_m_012``). The test
# preparation checks its speakers, compared as the ids are written, against
# the prepared train manifest before writing a clip; the train preparation is
# the one that acts on the mapped match.
TEST_SPLIT = "test"
TEST_SPEECH = f"{ROOT}/speech/{TEST_SPLIT}"
TEST_TRANSCRIPTS = (f"{ROOT}/transcripts/{TEST_SPLIT}/ca16/prompts.txt",)
# The published size of the test part; tests patch this.
TEST_EXPECTED_CLIPS = 515
# The name a transcript line and an audio member share, without the extension.
SUFFIXES = (".wav", ".tdf")
# A space after an apostrophe is the corpus's own typing (``d' accord``), not a
# word break: the model learns it and the scorer counts two words.
# The corpus writes an apostrophe as a separate word ("d' accord", "grand '
# mère"). The spaces around it are removed only when the apostrophe stands
# between two letters, which is what an elision looks like; an apostrophe that
# quotes something is left alone. All 5 697 apostrophes of the train transcripts
# join under this rule, so nothing of the corpus's own text is left untouched.
# The CA16 speaker formats of the two parts, one shared numbering: the train
# part's directory is the bare number, the test part's directory is
# ``<country>_<m|f>_<number>``. Every other id — the Yaoundé part's
# ``ctell1-01`` and its recording numbers, a two-digit ``12`` — has no CA16
# number: a trailing number of such an id is not one.
CA16_TRAIN_SPEAKER = re.compile(r"\d{3}\Z")
CA16_TEST_SPEAKER = re.compile(r"[a-z]+_[mf]_(\d{3})\Z")
APOSTROPHE_SPACING = re.compile(r"(?<=[^\W\d_]) *' *(?=[^\W\d_])")
# A line holding this offers the speaker two answers ("c' est lui / elle", "un
# / une"): the recording is one of them, the transcript does not say which, so
# the line is not a label that matches its audio.
PROMPT_MARKER = " / "
WHITESPACE_RUN = re.compile(r"\s+")


def ca16_speaker_number(speaker_id: str) -> str | None:
    """The shared CA16 number of an SLR57 speaker id in either part's format.

    The two parts name the same CA16 speaker in two formats, ``ca16_speaker_number("012")``
    and ``ca16_speaker_number("cameroon_m_012")`` are both ``"012"``. An id
    outside the two formats returns ``None`` and is compared as it is: the
    Yaoundé part's ``ctell1-01`` carries no CA16 number (a trailing number of
    such an id is not one), nor do a two-digit ``12``, a four-digit ``0123``,
    an uppercase country or a gender letter other than ``m`` and ``f``.
    """
    test = CA16_TEST_SPEAKER.fullmatch(speaker_id)
    if test is not None:
        return test.group(1)
    if CA16_TRAIN_SPEAKER.fullmatch(speaker_id):
        return speaker_id
    return None


def same_ca16_speaker(train_id: str, test_id: str) -> bool:
    """Whether a train speaker and a test speaker are the same CA16 speaker.

    The ids agree when they are equal as strings, or when both carry a CA16
    number and the numbers are equal: ``same_ca16_speaker("012",
    "cameroon_m_012")`` is True. An id without a number agrees only with
    itself, so ``same_ca16_speaker("ctell1-01", "chad_m_001")`` is False,
    although the first id ends in a number.
    """
    if train_id == test_id:
        return True
    number = ca16_speaker_number(train_id)
    return number is not None and number == ca16_speaker_number(test_id)


def excluded_train_speakers(test_speakers: Iterable[str]) -> dict[str, str]:
    """The train-side speaker ids a train preparation leaves out, and why.

    ``test_speakers`` are the groups of the prepared test part, in that part's
    own format. Every test id is left out as it is — the comparison the id
    check has always made — and a test id that carries a CA16 number also
    leaves out the train speaker of that bare number, because the two parts
    name the same CA16 speaker in the two formats (``cameroon_m_012`` and
    ``012``). The function reads nothing but its arguments, so the fetcher
    that applies the rule and the measurement of what a prepared train set
    would hold use the same one.
    """
    left_out: dict[str, str] = {}
    for speaker in test_speakers:
        left_out[speaker] = "a test speaker"
        number = ca16_speaker_number(speaker)
        if number is not None and number != speaker:
            left_out[number] = f"the CA16 speaker of test speaker {speaker}"
    return left_out


def fetch_african_accented_fr_test(
    name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool
) -> int:
    """Prepare the test part in full, returning the number of clips written.

    The test part is an evaluation set (decision 0002), so nothing is capped:
    every recording with a transcript becomes a clip. Its speakers are checked
    against the prepared train manifest, because the two parts of one corpus
    must not share a voice; a speaker the train set already holds is dropped
    from the test set, and more than a tenth of the test speakers overlapping is
    a stop, not a cleanup. The check compares the ids as they are written: the
    two parts name a shared CA16 speaker in two formats, and it is the train
    preparation — the other way round — that acts on that match. The reference
    text is the corpus's own: casing and
    punctuation are kept, and only the apostrophe spacing and the whitespace
    runs are cleaned, exactly as the train text is.
    """
    archive = download(
        ARCHIVE_URL,
        cache_dir / "african-accented-fr-train" / ARCHIVE_NAME,
        sha256=PINNED_SHA256.get(ARCHIVE_NAME, ""),
    )
    transcripts = _read_transcripts(name, archive, TEST_TRANSCRIPTS)
    excluded = _train_speakers(dataset_dir)
    with open_archive(archive) as handle:
        members = [
            (member.name, Path(member.name).parent.name)
            for member in handle
            if _is_test_speech(member)
        ]
    speakers = {speaker for member_name, speaker in members if _key(member_name) in transcripts}
    overlap = sorted(speakers & excluded)
    if overlap and len(overlap) > len(speakers) / 10:
        raise DownloadError(
            f"{name}: {len(overlap)} of the {len(speakers)} test speakers overlap the "
            f"prepared train set ({', '.join(overlap[:3])} …): the test part is not "
            "disjoint from the training data"
        )
    if overlap:
        print(
            f"  {name}: {len(overlap)} test speakers overlap the train set and are dropped: "
            f"{', '.join(overlap)}"
        )
    expected = TEST_EXPECTED_CLIPS - sum(
        1 for member_name, speaker in members if speaker in overlap
    )
    with ManifestBuilder(dataset_dir, verbose=verbose, total=expected) as builder:
        with open_archive(archive) as handle:
            for member in handle:
                if not _is_test_speech(member):
                    continue
                key = _key(member.name)
                text = transcripts.get(key)
                if text is None:
                    continue
                speaker = Path(member.name).parent.name
                if speaker in overlap:
                    continue
                # The part's one transcript is its prompted sentences: every
                # recording is a read sentence, one per speaker's directory.
                builder.add(
                    f"{name}:{key}",
                    partial(audio_from_archive, handle, member),
                    language=LANGUAGE,
                    text=training_text(text),
                    text_has_casing_punct=True,
                    group=f"{name}:{speaker}",
                    group_kind="speaker",
                    source_split=TEST_SPLIT,
                    license=LICENSE,
                    category="read",
                )
        if verbose:
            print(f"  {name}: {len(builder.clips)} clips; {len(speakers)} speakers")
        require_clips(name, len(builder.clips), expected)
        builder.write()
    return len(builder.clips)


def _is_test_speech(member) -> bool:
    return (
        member.isfile()
        and member.name.startswith(f"{TEST_SPEECH}/")
        and member.name.endswith(".wav")
    )


def _train_speakers(dataset_dir: Path) -> set[str]:
    """The speakers the prepared train set holds, or none when it is not prepared.

    The check reads the manifest of ``african-accented-fr-train``, the train part
    of the same corpus, from the data directory next to this one. A machine that
    has not prepared it yet excludes nothing: the train fetcher keeps its own
    check, the other way round.
    """
    train_manifest = dataset_dir.parent / "african-accented-fr-train" / MANIFEST_NAME
    if not train_manifest.is_file():
        return set()
    return {clip.group.partition(":")[2] for clip in read_manifest(train_manifest.parent)}


def fetch_african_accented_fr_train(
    name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool
) -> int:
    """Prepare the train part up to the cap, returning the number of clips written."""
    archive = download(
        ARCHIVE_URL,
        cache_dir / name / ARCHIVE_NAME,
        sha256=PINNED_SHA256.get(ARCHIVE_NAME, ""),
    )
    transcripts = _read_transcripts(name, archive, TRANSCRIPTS)
    with ManifestBuilder(dataset_dir, verbose=verbose, total=TRAIN_CLIP_CAP) as builder:
        take = Take(
            name=name,
            builder=builder,
            language=LANGUAGE,
            source_split=SPLIT,
            license=LICENSE,
            text_has_casing_punct=False,
            # The test part of the same corpus is an evaluation set (decision
            # 0002): its speakers are never trained on. The two parts name the
            # CA16 speakers in two formats (``012`` and ``cameroon_m_012``), so
            # the exclusion holds every train-side id the test speakers map to:
            # a train speaker whose id — or whose CA16 number — is a test
            # speaker's is left out.
            excluded=frozenset(
                excluded_train_speakers(evaluation_groups("african-accented-fr-test"))
            ),
            cap=TRAIN_CLIP_CAP,
            cap_per_group=CLIP_CAP_PER_GROUP,
        )

        untranscribed = 0
        prompts = 0
        with open_archive(archive) as handle:
            for member in handle:
                if take.full():
                    break
                if not _is_speech(member):
                    continue
                text = transcripts.get(_key(member.name))
                if text is None:
                    untranscribed += 1
                    continue
                if PROMPT_MARKER in text:
                    prompts += 1
                    continue
                # The directory above the recording is the speaker of the train
                # part (``speech/train/ca16/113/...``, ``speech/train/yaounde/read/
                # ctell5-78/...``), and one speaker appears in both the read and the
                # answer recordings of the Yaoundé part.
                take.offer(
                    _key(member.name),
                    partial(audio_from_archive, handle, member),
                    text=training_text(text),
                    group=Path(member.name).parent.name,
                )
        if verbose:
            print(
                f"  {take.report()}; audio members with no transcript: {untranscribed}; "
                f"prompting lines left out: {prompts}"
            )
        builder.write()
    return take.kept


def training_text(text: str) -> str:
    """Clean one transcript into the text a clip of this corpus is labelled with.

    The cleanup is deterministic and only ever removes spaces: the ones the
    corpus writes around an apostrophe that follows a letter (``d' accord``,
    ``grand ' mère``), and the runs of whitespace it leaves where it dropped
    punctuation (``il y en a plus  il y en a encore``). Nothing is added back —
    the text stays lowercase and unpunctuated, as the corpus published it.

    Both prepared sets of the corpus store the cleaned text, the training
    labels and the evaluation references alike. The cleanup is not neutral for
    the scorer: it keeps the apostrophe and tokenizes on whitespace, so
    ``d' accord`` scores as two words and ``d'accord`` as one. It keeps the
    corpus's own typing out of what the train labels teach. For the test
    references it changes nothing the scorer could see: none of the 515
    published prompts writes a space after an apostrophe (verified against
    the pinned archive), and the runs of whitespace it removes are collapsed
    by the tokenization.
    """
    return WHITESPACE_RUN.sub(" ", APOSTROPHE_SPACING.sub("'", text)).strip()


def _is_speech(member) -> bool:
    return member.isfile() and member.name.startswith(f"{SPEECH}/") and member.name.endswith(".wav")


def _key(path_in_archive: str) -> str:
    """The name a transcript line and an audio member share.

    The recordings are named after their date (``afc-gabon_16.06.15_048_read``),
    so no generic "stem" can be taken: only a known extension can be removed.
    """
    base = path_in_archive.rsplit("/", 1)[-1]
    for suffix in SUFFIXES:
        base = base.removesuffix(suffix)
    return base


def _read_transcripts(
    name: str, archive: Path, transcript_names: tuple[str, ...]
) -> dict[str, str]:
    """Every transcript of the archive's part, keyed by recording name."""
    transcripts: dict[str, str] = {}
    read = 0
    with open_archive(archive) as handle:
        for member in handle:
            if member.name not in transcript_names:
                continue
            # The published archive holds its transcript files before the
            # recordings, so the walk can stop once all of them are read; if
            # that ever changes it simply reads further and still finds them.
            read += 1
            data = handle.extractfile(member)
            if data is None:
                raise DownloadError(f"{name}: {member.name} is not a readable file")
            transcripts.update(_parse_transcripts(name, member.name, data.read()))
            if read == len(transcript_names):
                break
    if not transcripts:
        raise DownloadError(f"{name}: the archive holds no transcript of its part")
    return transcripts


def _parse_transcripts(name: str, source: str, data: bytes) -> dict[str, str]:
    """Read a transcript file: ``<recording> <text>`` per line."""
    transcripts = {}
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        if len(fields) < 2:
            raise DownloadError(f"{name}: {source} has an unexpected line: {line[:60]!r}")
        transcripts[_key(fields[0])] = fields[1].strip()
    return transcripts
