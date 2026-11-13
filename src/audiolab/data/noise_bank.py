"""Noise banks: MUSAN noise and RIRS_NOISES room impulse responses.

Two non-speech datasets. They are not evaluation sets and have no panels: they
hold the files that ``noisy.py`` mixes into the noisy evaluation sets and that
plan 02 mixes into training data. One rule decides forever which half of the
MUSAN noise is for evaluation and which for training, so an evaluation noise
can never be the noise a model was augmented with (decision 0002, applied to
non-speech audio).

MUSAN (OpenSLR 17, CC-BY-4.0 for the corpus) keeps its noise in two
sub-collections: ``musan/noise/free-sound/`` (marked public domain) and
``musan/noise/sound-bible/`` (per block "CC BY 3.0" or "Public Domain", in the
sub-collection's ``LICENSE``). A sound-bible block whose licence is anything
else is dropped and counted. The ``musan/speech/`` and ``musan/music/`` parts
are not read here; the music part is the material of ``nonspeech-music``.

RIRS_NOISES (OpenSLR 28, Apache-2.0) keeps room impulse responses and noises;
its ``pointsource_noises/`` are copies of MUSAN files and are never read. The
real-room RIRs are the evaluation half and the simulated ones the training
half: a model augmented in simulated rooms is then scored in real rooms, which
is the honest direction for a noise-robustness claim.
"""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
import zipfile
from pathlib import Path

import soundfile as sf
import soxr

from audiolab.core.audio import SAMPLE_RATE, Audio, load_audio
from audiolab.data.build import ManifestBuilder, require_clips
from audiolab.data.download import DownloadError, download

MUSAN_URL = "https://openslr.trmal.net/resources/17/musan.tar.gz"
# The digest of musan.tar.gz as served by the OpenSLR mirrors at first download;
# the corpus publishes no checksum of its own.
MUSAN_SHA256 = "86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1"
RIRS_URL = "https://www.openslr.org/resources/28/rirs_noises.zip"
# The digest of rirs_noises.zip at first download; the corpus publishes no
# checksum of its own.
RIRS_SHA256 = "3b50cfde915b3984738169b4beb341e9f6b8062ae4c2076146c5db71c2c05dc7"

RIRS_LICENSE = "Apache-2.0"


def musan_split(name: str) -> str:
    """The half a MUSAN noise file belongs to: ``eval`` or ``train``.

    The rule is a constant, not a choice: an earlier private project trained on
    the odd half of these files, so the even half is the one no model has seen.
    The exact string ``musan-split-<name>`` is part of the rule and must not be
    changed, or the halves move.
    """
    digest = hashlib.sha256(f"musan-split-{name}".encode()).hexdigest()
    return "eval" if int(digest[:16], 16) % 2 == 0 else "train"


def parse_licence_blocks(text: str, file_prefix: str) -> dict[str, str | None]:
    """The per-file licence of one MUSAN sub-collection, from its LICENSE file.

    The file is a sequence of blocks separated by ``====`` lines. A licence
    line inside a block that also names files applies to those files, no matter
    where in the block it sits. A licence line in a block that names no files
    (the rfm part writes its ``CC BY 3.0`` header above the file list) is a
    sticky header: it applies to the files of the following blocks until a
    block names its own licence. A file with neither gets no licence, and the
    caller drops and counts it.
    """
    licenses: dict[str, str | None] = {}
    sticky: str | None = None
    for block in re.split(r"={10,}", text):
        files: list[str] = []
        in_block: str | None = None
        for line in block.splitlines():
            line = line.strip()
            if line.lower().startswith("license:"):
                in_block = line.split(":", 1)[1].strip() or None
                continue
            if line.startswith(file_prefix):
                files.append(_file_name(line))
                continue
            if line.startswith("http") or line.startswith('"') or line.startswith("'"):
                continue
            if _looks_like_licence(line):
                in_block = line
        if in_block is not None:
            if not files:
                sticky = in_block
            for name in files:
                licenses.setdefault(name, in_block)
        else:
            for name in files:
                licenses.setdefault(name, sticky)
    return licenses


def _looks_like_licence(line: str) -> bool:
    """A line that states the licence of a block, in either part's wording."""
    lowered = line.lower()
    return (
        "public domain" in lowered
        or "cc by" in lowered
        or "cc-by" in lowered
        or "attribution" in lowered
        or "royalty" in lowered
    )


def _sound_bible_licenses(archive: Path) -> dict[str, str | None]:
    """The per-file licence of the sound-bible noise, from its LICENSE file."""
    text = _tar_text(archive, "musan/noise/sound-bible/LICENSE")
    return parse_licence_blocks(text, "noise-")


def _tar_text(archive: Path, member_name: str) -> str:
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if member.name == member_name:
                source = tar.extractfile(member)
                break
        else:
            raise DownloadError(f"the MUSAN archive holds no {member_name}")
    if source is None:
        raise DownloadError(f"the MUSAN {member_name} is unreadable")
    return source.read().decode("utf-8", errors="replace")


def _file_name(line: str) -> str:
    stem = line.split(".wav")[0]
    return stem + ".wav"


def fetch_musan_noise(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare the MUSAN noise files, split forever by :func:`musan_split`."""
    cache = cache_dir / "musan"
    archive = download(MUSAN_URL, cache / "musan.tar.gz", sha256=MUSAN_SHA256)
    licenses = _sound_bible_licenses(archive)
    per_split: dict[str, int] = {}
    per_category: dict[str, int] = {}
    dropped = 0
    with ManifestBuilder(dataset_dir, verbose=verbose) as builder:
        with tarfile.open(archive, mode="r|gz") as tar:
            for member in tar:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                base = Path(member.name).name
                if member.name.startswith("musan/noise/free-sound/"):
                    category, licence = "free-sound", "public domain"
                elif member.name.startswith("musan/noise/sound-bible/"):
                    category = "sound-bible"
                    licence = licenses.get(base)
                    if licence is None or not _usable(licence):
                        dropped += 1
                        continue
                else:
                    continue
                stem = base.removesuffix(".wav")
                clip_id = f"{name}:{stem}"
                fields = _fields(name, stem, category, licence, base)
                if builder.prepared(clip_id, **fields):
                    continue
                # A streaming tar gives a non-seekable member, and the reader needs one.
                audio = load_audio(io.BytesIO(tar.extractfile(member).read()))
                builder.add(clip_id, audio, **fields)
                per_split[fields["source_split"]] = per_split.get(fields["source_split"], 0) + 1
                per_category[category] = per_category.get(category, 0) + 1
        if verbose:
            counts = ", ".join(f"{half} {n}" for half, n in sorted(per_split.items()))
            cats = ", ".join(f"{c} {n}" for c, n in sorted(per_category.items()))
            print(f"  {name}: {len(builder.clips)} files ({counts}; {cats}); {dropped} dropped by licence")
        require_clips(name, len(builder.clips), MUSAN_EXPECTED_CLIPS)
        builder.write()
    return len(builder.clips)


def _usable(licence: str) -> bool:
    """A sound-bible block the corpus lets us redistribute: PD or Attribution 3.0."""
    lowered = licence.lower()
    if "public domain" in lowered:
        return True
    return "cc by 3.0" in lowered or "attribution 3.0" in lowered


def _fields(name: str, stem: str, category: str, licence: str, base: str) -> dict:
    return {
        "text": "",
        "text_has_casing_punct": False,
        "language": "none",
        "group": f"{name}:{stem}",
        "group_kind": "recording",
        "source_split": musan_split(base),
        "license": licence,
        "category": category,
    }


def fetch_rirs_noises(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare the RIRS room impulse responses; real rooms evaluate, simulated train."""
    cache = cache_dir / "rirs-noises"
    archive = download(RIRS_URL, cache / "rirs_noises.zip", sha256=RIRS_SHA256)
    rirs = _rirs(archive)
    with zipfile.ZipFile(archive) as zip:  # the archive stays open: one open per clip would take hours
        with ManifestBuilder(dataset_dir, verbose=verbose, total=len(rirs)) as builder:
            for member, category, split, stem in rirs:
                clip_id = f"{name}:{stem}"
                fields = {
                    "text": "",
                    "text_has_casing_punct": False,
                    "language": "none",
                    "group": f"{name}:{stem}",
                    "group_kind": "recording",
                    "source_split": split,
                    "license": RIRS_LICENSE,
                    "category": category,
                }
                if builder.prepared(clip_id, **fields):
                    continue
                builder.add(clip_id, lambda m=member, z=zip: _first_channel(z, m), **fields)
            per: dict[str, int] = {}
            cats: dict[str, int] = {}
            for clip in builder.clips:
                per[clip.source_split] = per.get(clip.source_split, 0) + 1
                cats[clip.category] = cats.get(clip.category, 0) + 1
            if verbose:
                counts = ", ".join(f"{split} {n}" for split, n in sorted(per.items()))
                kinds = ", ".join(f"{c} {n}" for c, n in sorted(cats.items()))
                print(f"  {name}: {len(builder.clips)} RIRs ({counts}; {kinds})")
            require_clips(name, len(builder.clips), RIRS_EXPECTED_CLIPS)
            builder.write()
    return len(builder.clips)


def _first_channel(zip: zipfile.ZipFile, member: str) -> Audio:
    """One RIR as mono 16 kHz: channel 0 of a multi-channel measurement.

    The channels of a real recording are different microphones, so channel 0 is
    kept instead of averaging the array; a single-channel RIR is unchanged.
    """
    # A zip member is not seekable, and the reader needs one; the RIRs are small.
    samples, rate = sf.read(io.BytesIO(zip.open(member).read()), dtype="float32", always_2d=True)
    if samples.shape[1] > 1:
        samples = samples[:, 0]
    if rate != SAMPLE_RATE:
        samples = soxr.resample(samples, rate, SAMPLE_RATE, quality="HQ")
    return Audio(samples=samples.astype("float32", copy=False), sample_rate=SAMPLE_RATE)


def _rirs(archive: Path) -> list[tuple[str, str, str, str]]:
    """(member, category, split, stem) for every RIR of the archive.

    The lists, not the file names, say which file is a RIR: the real directory
    also holds the isotropic noises of the same recordings, named by a
    noise_list that is never read, and the point-source noises of the archive
    are MUSAN copies that are never read.
    """
    out: list[tuple[str, str, str, str]] = []
    with zipfile.ZipFile(archive) as zip:
        for line in zip.read("RIRS_NOISES/real_rirs_isotropic_noises/rir_list").decode().splitlines():
            member = line.split()[-1]
            out.append((member, "real-rir", "eval", Path(member).name.removesuffix(".wav")))
        for room in ("smallroom", "mediumroom", "largeroom"):
            lines = zip.read(f"RIRS_NOISES/simulated_rirs/{room}/rir_list").decode().splitlines()
            prefix = f"RIRS_NOISES/simulated_rirs/{room}/"
            for line in lines:
                member = line.split()[-1]
                # The rooms hold their files in per-room subdirectories and name
                # them alike, so the stem is the path under the room directory.
                relative = Path(member.removeprefix(prefix))
                out.append((member, "simulated-rir", "train", f"simulated/{room}/{relative.with_suffix('').as_posix()}"))
    return out


# The sizes the archives hold; pinned from the first preparation, tests patch these.
MUSAN_EXPECTED_CLIPS = 929
RIRS_EXPECTED_CLIPS = 325 + 60000
