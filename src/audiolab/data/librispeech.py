"""LibriSpeech (``librispeech-en`` test-clean, ``librispeech-en-train``): audiobook excerpts.

Only the test-clean archive is downloaded for evaluation: test-other is harder
speech and is left for a later phase. A training source takes the first clips of
``train-clean-100`` instead, in the archive's own order, so it stops reading as
soon as it has enough. The archive holds FLAC files named
``speaker/chapter/file.flac`` next to a ``.trans.txt`` per chapter, and a tar
stream cannot be read backwards, so the transcripts are read in a first pass
and the audio in a second one.

LibriSpeech publishes uppercase, unpunctuated transcripts, and its speakers are
books read aloud. The base model writes casing and punctuation, so this source
teaches it to drop both: a known confound of the screening, recorded in its card.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from audiolab.data.build import ManifestBuilder, require_clips
from audiolab.data.download import DownloadError, audio_from_archive, download, open_archive
from audiolab.data.train import CLIP_CAP_PER_GROUP, TRAIN_CLIP_CAP, Take, evaluation_groups

ARCHIVE_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
ARCHIVE_NAME = "test-clean.tar.gz"
TRAIN_ARCHIVE_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
TRAIN_ARCHIVE_NAME = "train-clean-100.tar.gz"
LICENSE = "CC-BY-4.0"
LANGUAGE = "en"
SPLIT = "test-clean"
TRAIN_SPLIT = "train-clean-100"
# The evaluation set prepared from test-clean: LibriSpeech keeps the speakers of
# its splits apart, so nothing should be found in common here, but a speaker
# that is scored is never trained on (decision 0002).
EVALUATION_DATASET = "librispeech-en"
# OpenSLR publishes this file without a checksum, so the sha256 saw when the
# panels were built is recorded here and checked on every fresh download.
PINNED_SHA256 = {
    "test-clean.tar.gz": "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
    # Recorded when this source was prepared; OpenSLR publishes md5 sums only,
    # and the archive's md5 matches the one published for it (2a93770f...).
    "train-clean-100.tar.gz": "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",
}
# The published size of the archive; tests patch this to a small number.
EXPECTED_CLIPS = 2620


def fetch_librispeech_en(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare LibriSpeech test-clean, returning the number of clips written."""
    archive_path = download(
        ARCHIVE_URL,
        cache_dir / name / ARCHIVE_NAME,
        sha256=PINNED_SHA256.get(ARCHIVE_NAME, ""),
    )
    transcripts = _read_transcripts(name, archive_path)
    builder = ManifestBuilder(dataset_dir, verbose=verbose)

    with open_archive(archive_path) as handle:
        for member in handle:
            if member.name.endswith(".flac"):
                text = transcripts.get(Path(member.name).stem)
                if text is None:
                    raise DownloadError(f"{name}: {member.name} has no transcript in the archive")
                add_librispeech_clip(builder, name, handle, member, text)

    require_clips(name, len(builder.clips), EXPECTED_CLIPS)
    builder.write()
    return len(builder.clips)


def fetch_librispeech_en_train(
    name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool
) -> int:
    """Prepare the first clips of LibriSpeech train-clean-100, up to the cap."""
    archive_path = download(
        TRAIN_ARCHIVE_URL,
        cache_dir / name / TRAIN_ARCHIVE_NAME,
        sha256=PINNED_SHA256.get(TRAIN_ARCHIVE_NAME, ""),
    )
    builder = ManifestBuilder(dataset_dir, verbose=verbose)
    take = prepare_train_audio(name, archive_path, builder)
    if verbose:
        print(f"  {take.report()}")
    builder.write()
    return take.kept


def prepare_train_audio(
    name: str,
    archive_path: Path,
    builder: ManifestBuilder,
    *,
    text_for_clip: Callable[[str], str | None] | None = None,
    text_has_casing_punct: bool = False,
) -> Take:
    """Prepare the clips of ``train-clean-100``, in the archive's published order.

    A clip's text and speaker are read from the archive itself, and a source
    that has better text for the same audio (``librispeech-pc-en-train``) passes
    ``text_for_clip``: it receives the utterance id, returns the label to use in
    place of the published text, or ``None`` for a clip it cannot label. Such a
    source counts what it drops itself, in its own closure, and only it knows
    what its text looks like, hence ``text_has_casing_punct``.

    The whole archive is read once for its transcripts, then again for the audio
    of the clips taken from its beginning.
    """
    transcripts = _read_transcripts(name, archive_path)
    take = Take(
        name=name,
        builder=builder,
        language=LANGUAGE,
        source_split=TRAIN_SPLIT,
        license=LICENSE,
        text_has_casing_punct=text_has_casing_punct,
        excluded=evaluation_groups(EVALUATION_DATASET),
        cap=TRAIN_CLIP_CAP,
        cap_per_group=CLIP_CAP_PER_GROUP,
    )

    with open_archive(archive_path) as handle:
        for member in handle:
            if take.full():
                return take
            if not member.name.endswith(".flac"):
                continue
            utterance = Path(member.name).stem
            text = transcripts.get(utterance)
            if text is None:
                raise DownloadError(f"{name}: {member.name} has no transcript in the archive")
            if text_for_clip is not None:
                restored = text_for_clip(utterance)
                if restored is None:
                    continue
                text = restored
            take.offer(
                utterance,
                audio_from_archive(handle, member),
                text=text,
                group=_speaker_of(name, member),
            )
    return take


def _read_transcripts(name: str, archive_path: Path) -> dict[str, str]:
    """First pass: every ``.trans.txt`` of the archive, keyed by utterance id."""
    transcripts: dict[str, str] = {}
    with open_archive(archive_path) as handle:
        for member in handle:
            if member.name.endswith(".trans.txt"):
                transcripts.update(_parse_transcripts(_read_member(name, handle, member)))
    if not transcripts:
        raise DownloadError(f"{name}: the archive holds no .trans.txt file")
    return transcripts


def _read_member(name: str, handle, member) -> bytes:
    data = handle.extractfile(member)
    if data is None:
        raise DownloadError(f"{name}: {member.name} is not a readable file")
    return data.read()


def _speaker_of(name: str, member) -> str:
    """The speaker of one archive member, read from ``speaker/chapter/file``."""
    parts = Path(member.name).parts
    if len(parts) < 3:
        raise DownloadError(f"{name}: unexpected path in the archive: {member.name}")
    return parts[-3]


def add_librispeech_clip(builder: ManifestBuilder, name: str, handle, member, text: str) -> None:
    """Add one archive member; its path gives the speaker (``speaker/chapter/file``)."""
    builder.add(
        f"{name}:{Path(member.name).stem}",
        audio_from_archive(handle, member),
        language=LANGUAGE,
        text=text,
        text_has_casing_punct=False,
        group=f"{name}:{_speaker_of(name, member)}",
        group_kind="speaker",
        source_split=SPLIT,
        license=LICENSE,
    )


def _parse_transcripts(data: bytes) -> dict[str, str]:
    """Read a ``.trans.txt`` file: one ``<utterance id> <text>`` per line."""
    lines = data.decode("utf-8").splitlines()
    return {line.split(" ", 1)[0]: line.split(" ", 1)[1] for line in lines if " " in line}
