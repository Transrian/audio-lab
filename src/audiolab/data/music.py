"""nonspeech-music: a music battery for the hallucination test.

Music must give an empty transcript, the way silence and noise do (the
``nonspeech`` battery, backlog N-03). The material is the ``musan/music/`` part
of the same archive the noise bank uses: one 10 s excerpt per file, from files
that the sub-collection's own licence file marks public domain or CC-BY (any
version) and whose annotation says no vocals — a sung file would give the model
words to hallucinate, and a non-redistributable one would not belong in the
prepared data.

Only files that pass both tests are kept, one excerpt each, spread over the
sub-collections, capped at 300 files. The dataset has its own panels, drawn
family by family like the rest of the non-speech battery, with ``music`` as a
family of its own.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

from audiolab.core.audio import SAMPLE_RATE, Audio, load_audio
from audiolab.data.build import ManifestBuilder, require_clips
from audiolab.data.download import DownloadError, download
from audiolab.data.noise_bank import MUSAN_SHA256, MUSAN_URL, parse_licence_blocks

EXCERPT_SECONDS = 10
MAX_FILES = 300
MUSIC_DIR = "musan/music"


def music_licence_ok(licence: str | None) -> bool:
    """A sub-collection block the corpus lets us redistribute: PD or CC-BY, no variants.

    The corpus writes the licence in many wordings — ``CC BY 4.0``, ``Attribution
    3.0 International License.``, ``Public Domain Mark 1.0``, a sticky ``CC BY 3.0``
    header — so the test matches the licence's substance: a share-alike,
    non-commercial or no-derivatives mark makes the variant a different licence
    than CC-BY, and would have to be honoured on a prepared dataset.
    """
    if not licence:
        return False
    text = " ".join(licence.lower().split())
    if "share alike" in text or "sharealike" in text or "by-sa" in text:
        return False
    if "noncommercial" in text or "non-commercial" in text or "by-nc" in text:
        return False
    if "noderivatives" in text or "no-derivatives" in text or "noderivs" in text or "by-nd" in text:
        return False
    if "public domain" in text:
        return True
    return "cc by" in text or "attribution" in text


def _scan(archive: Path) -> tuple[set[str], dict[str, str]]:
    """One streaming pass: the music wav members, and the licence and annotation texts.

    The archive is a streaming tar, read once: a second read of a single member
    would mean re-downloading nothing but re-decompressing the whole archive, so
    everything a decision needs is gathered here.
    """
    wavs: set[str] = set()
    texts: dict[str, str] = {}
    with tarfile.open(archive, mode="r|gz") as tar:
        for member in tar:
            if not member.name.startswith(MUSIC_DIR + "/") or not member.isfile():
                continue
            parts = member.name.split("/")
            if len(parts) != 4:
                continue
            if member.name.endswith(".wav"):
                wavs.add(member.name)
            elif member.name.split("/")[-1] in ("LICENSE", "ANNOTATIONS"):
                source = tar.extractfile(member)
                texts[member.name] = source.read().decode("utf-8", errors="replace")
    return wavs, texts


def _annotations(text: str) -> dict[str, tuple[bool, str]]:
    """(has vocals, artist) per file, from a sub-collection's ANNOTATIONS.

    A line is ``<id> <genre> <vocal> <artist>``, with the vocal flag Y or N;
    the id carries no extension and the artist may hold spaces.
    """
    out: dict[str, tuple[bool, str]] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 3 or not fields[0].startswith("music-"):
            continue
        stem = fields[0].removesuffix(".wav")
        out[stem] = (fields[2] == "Y", " ".join(fields[3:]).strip())
    return out


def _decide(
    wavs: set[str], texts: dict[str, str]
) -> tuple[dict[str, dict[str, int]], list[tuple[str, str, str, str]]]:
    """The per-subcollection report, and (member, subcollection, stem, licence) per eligible file.

    The report and the selection share this one decision, so what the log says
    is what the manifest holds. A sub-collection without its LICENSE or
    ANNOTATIONS makes the fetch fail: silent missing metadata is how a corpus
    changes under an evaluation.
    """
    subcollections = sorted({member.split("/")[2] for member in wavs})
    report: dict[str, dict[str, int]] = {}
    eligible: list[tuple[str, str, str, str]] = []
    for subcollection in subcollections:
        licence_name = f"{MUSIC_DIR}/{subcollection}/LICENSE"
        annotation_name = f"{MUSIC_DIR}/{subcollection}/ANNOTATIONS"
        if licence_name not in texts or annotation_name not in texts:
            raise DownloadError(
                f"the MUSAN archive's {subcollection} part holds no LICENSE and ANNOTATIONS"
            )
        licences = parse_licence_blocks(texts[licence_name], "music-")
        annotations = _annotations(texts[annotation_name])
        per_licence: dict[str, int] = {}
        for stem, (has_vocals, _artist) in sorted(annotations.items()):
            member = f"{MUSIC_DIR}/{subcollection}/{stem}.wav"
            if member not in wavs:
                continue
            licence = licences.get(f"{stem}.wav")
            if has_vocals or not music_licence_ok(licence):
                continue
            per_licence[licence or "unknown"] = per_licence.get(licence or "unknown", 0) + 1
            eligible.append((member, subcollection, stem, licence or "unknown"))
        report[subcollection] = per_licence
    return report, eligible


def fetch_nonspeech_music(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare the music battery: 10 s excerpts of the licensed, instrumental files."""
    cache = cache_dir / "musan"
    archive = download(MUSAN_URL, cache / "musan.tar.gz", sha256=MUSAN_SHA256)
    wavs, texts = _scan(archive)
    report, eligible = _decide(wavs, texts)
    if verbose:
        for subcollection in sorted(report):
            counts = ", ".join(f"{kind} {n}" for kind, n in sorted(report[subcollection].items()))
            print(f"  {subcollection}: {counts}")
        print(f"  {name}: {len(eligible)} eligible files before the {MAX_FILES} cap")
    selected = _spread(eligible, MAX_FILES)
    by_member = {entry[0]: entry for entry in selected}
    with ManifestBuilder(dataset_dir, verbose=verbose, total=len(selected)) as builder:
        with tarfile.open(archive, mode="r|gz") as tar:
            for member in tar:
                entry = by_member.get(member.name)
                if entry is None or not member.isfile():
                    continue
                _member, subcollection, stem, licence = entry
                clip_id = f"{name}:{stem}"
                artist = _annotations(texts[f"{MUSIC_DIR}/{subcollection}/ANNOTATIONS"]).get(
                    stem, (False, "")
                )[1]
                fields = {
                    "text": "",
                    "text_has_casing_punct": False,
                    "language": "none",
                    "group": f"{name}:{artist or stem}",
                    "group_kind": "artist" if artist else "recording",
                    "source_split": subcollection,
                    "license": licence,
                    "category": "music",
                }
                if builder.prepared(clip_id, **fields):
                    continue
                # A streaming tar gives a non-seekable member, and the reader needs one.
                audio = load_audio(io.BytesIO(tar.extractfile(member).read()))
                length = int(EXCERPT_SECONDS * SAMPLE_RATE)
                if len(audio.samples) > length:
                    offset = _offset(stem) % (len(audio.samples) - length)
                    audio = Audio(
                        samples=audio.samples[offset : offset + length],
                        sample_rate=SAMPLE_RATE,
                    )
                builder.add(clip_id, audio, **fields)
        if verbose:
            per: dict[str, int] = {}
            for clip in builder.clips:
                per[clip.source_split] = per.get(clip.source_split, 0) + 1
            counts = ", ".join(f"{s} {n}" for s, n in sorted(per.items()))
            print(f"  {name}: {len(builder.clips)} excerpts prepared ({counts})")
        require_clips(name, len(builder.clips), MUSIC_EXPECTED_CLIPS)
        builder.write()
    return len(builder.clips)


def _spread(kept: list[tuple], count: int) -> list[tuple]:
    """Up to ``count`` files, one sub-collection after another in turn."""
    by_collection: dict[str, list[tuple]] = {}
    for entry in kept:
        by_collection.setdefault(entry[1], []).append(entry)
    collections = sorted(by_collection)
    selected: list[tuple] = []
    index = 0
    while len(selected) < count:
        added = False
        for collection in collections:
            if len(selected) >= count:
                break
            pool = by_collection[collection]
            if index < len(pool):
                selected.append(pool[index])
                added = True
        index += 1
        if not added:
            break
    return selected


def _offset(stem: str) -> int:
    return int(hashlib.sha256(f"nonspeech-music-offset-{stem}".encode()).hexdigest()[:16], 16)


# The number of files the archive's licences and annotations admit (up to the
# cap), pinned from the first preparation, tests patch this.
MUSIC_EXPECTED_CLIPS = 261
