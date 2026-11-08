"""African Accented French (``african-accented-fr-train``): read and conversational speech.

OpenSLR SLR57 is Apache-2.0 speech recorded in Yaoundé and Libreville from
French speakers of African accents: prompted sentences, answers and short
conversations. The corpus is one 1.7 GiB tar archive holding the four parts it
publishes (``train``, ``dev``, ``devtest``, ``test``) with their transcripts;
only the train part is prepared here, so the other parts stay untouched for
evaluation. The whole archive has to be downloaded either way, because the
corpus publishes no smaller unit.

Three transcript files describe the train part, one per recording style, and
they name their files in three ways: a bare name, a name with ``.tdf`` and an
absolute path of the machine the corpus was built on. All three give the same
key once the directory and the extension are removed, which is what
:func:`_key` does. The corpus has no transcript for 2 934 of its 14 394 train
recordings; those are left out.

The text is lowercase and unpunctuated, and a few lines keep a word split in
two (``il y en a plus  il y en a encore``). The base model writes casing and
punctuation, so this source teaches it to drop both: a known confound of the
screening, recorded in its card rather than corrected here, because rewriting
the reference would mean scoring a corpus nobody published.
"""

from __future__ import annotations

from pathlib import Path

from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError, audio_from_archive, download, open_archive
from audiolab.data.train import TRAIN_CLIP_CAP, Take

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
# The name a transcript line and an audio member share, without the extension.
SUFFIXES = (".wav", ".tdf")


def fetch_african_accented_fr_train(
    name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool
) -> int:
    """Prepare the train part up to the cap, returning the number of clips written."""
    archive = download(
        ARCHIVE_URL,
        cache_dir / name / ARCHIVE_NAME,
        sha256=PINNED_SHA256.get(ARCHIVE_NAME, ""),
    )
    transcripts = _read_transcripts(name, archive)
    builder = ManifestBuilder(dataset_dir, verbose=verbose)
    take = Take(
        name=name,
        builder=builder,
        language=LANGUAGE,
        source_split=SPLIT,
        license=LICENSE,
        text_has_casing_punct=False,
        cap=TRAIN_CLIP_CAP,
    )

    untranscribed = 0
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
            # The directory above the recording is the speaker of the train
            # part (``speech/train/ca16/113/...``, ``speech/train/yaounde/read/
            # ctell5-78/...``), and one speaker appears in both the read and the
            # answer recordings of the Yaoundé part.
            take.offer(
                _key(member.name),
                audio_from_archive(handle, member),
                text=text,
                group=Path(member.name).parent.name,
            )
    if verbose:
        print(f"  {take.report()}; audio members with no transcript: {untranscribed}")
    builder.write()
    return take.kept


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


def _read_transcripts(name: str, archive: Path) -> dict[str, str]:
    """Every train transcript of the archive, keyed by recording name."""
    transcripts: dict[str, str] = {}
    read = 0
    with open_archive(archive) as handle:
        for member in handle:
            if member.name not in TRANSCRIPTS:
                continue
            # The published archive holds its transcript files before the
            # recordings, so the walk can stop once all three are read; if that
            # ever changes it simply reads further and still finds them.
            read += 1
            data = handle.extractfile(member)
            if data is None:
                raise DownloadError(f"{name}: {member.name} is not a readable file")
            transcripts.update(_parse_transcripts(name, member.name, data.read()))
            if read == len(TRANSCRIPTS):
                break
    if not transcripts:
        raise DownloadError(f"{name}: the archive holds no train transcript")
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
