"""CS-FLEURS French-English (``cs-fleurs-fr-en``): one sentence, two languages.

CS-FLEURS builds code-switched sentences out of FLEURS sentences: the frame is
one language and part of it is replaced by the other, at the word or clause
level. Its French-English pairs are the ones prepared here, in the two test
subsets that hold them:

- ``read/test``: one person reading the code-switched sentence (real audio);
- ``xtts/test1``: a generative voice, up to three synthetic voices per sentence.

Both are kept, told apart by ``category``, because the synthetic half shows what
a voice that has never heard this accent does with the same texts. The
``mms/test`` subset is left out: its reference texts are the corpus's own
word-level translations and read like machine output (``il aurait une 20s.
d'années``), so a score on them would measure the corpus as much as the model.

The sentences come from the FLEURS test split — the same sentences
``fleurs-fr`` scores, read here in a code-switched version — so the two datasets
overlap in text and never in audio. Each prepared clip gets the group of its
FLEURS sentence, exactly as ``fleurs-fr`` does.

The corpus has one speaker for the read subset and three synthetic voices for
the XTTS one, so a speaker-level bootstrap would rest on four groups; the
sentence is the group instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx

from audiolab.core.audio import load_audio
from audiolab.data.build import ManifestBuilder, require_clips
from audiolab.data.download import DownloadError, download, huggingface_sha256_map

REPO = "byan/cs-fleurs"
# Pinned so the files a panel was built from cannot change under it.
REVISION = "0cdbf166c5517ae4b6eb1c54248522eedec53017"
REPO_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}"
LICENSE = "CC-BY-NC-4.0 (testing only, decision 0009)"
# The corpus's own language field for these rows is "fra-eng": French frame,
# English embedded, in both subsets.
LANGUAGE_PAIR = "fra-eng"
LANGUAGE = "fr"
# The FLEURS sentence a clip was built from: "fra_1660_MW" and
# "1960_fra-spk0_sample0" are both sentence 1960's relatives.
SENTENCE = re.compile(r"^(?:fra_)?(\d+)_")
# Directory of one test subset, the category its clips are prepared under, and
# the sha256 of the metadata file that lists them at the pinned revision. The
# audio checksums are read from the Hub listing, one per file.
SUBSETS = (
    ("read/test", "read", 307, "1ed381025ed51c368e591e263eace85a6e4e59a215d3f8888ca07c1e188fd709"),
    ("xtts/test1", "xtts", 676, "677893d3db633645aedbb0a1d838e046afbad53c9817f66b3e5977c24fd0065d"),
)  # fmt: skip


@dataclass(frozen=True)
class Subset:
    """One test subset of the corpus, as prepared."""

    directory: str
    category: str
    clips: int
    metadata_sha256: str


def fetch_cs_fleurs_fr_en(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare every French-English test subset, returning the number of clips written."""
    builder = ManifestBuilder(dataset_dir, verbose=verbose)
    for directory, category, clips, metadata_sha256 in SUBSETS:
        subset = Subset(directory, category, clips, metadata_sha256)
        prepared = _prepare_subset(name, subset, cache_dir, builder, verbose=verbose)
        require_clips(f"{name} ({directory})", prepared, subset.clips)
    builder.write()
    return len(builder.clips)


def _prepare_subset(
    name: str, subset: Subset, cache_dir: Path, builder: ManifestBuilder, *, verbose: bool
) -> int:
    """Prepare one subset, returning the number of clips added to the builder."""
    metadata = download(
        f"{REPO_URL}/{subset.directory}/metadata.jsonl",
        cache_dir / name / f"{subset.category}-metadata.jsonl",
        sha256=subset.metadata_sha256,
    )
    records = _read_metadata(name, metadata)
    wanted = [record for record in records if record["language"] == LANGUAGE_PAIR]
    checksums = _published_checksums(name, subset, wanted, cache_dir / name)
    if verbose:
        print(f"  ... {subset.directory}: {len(wanted)} clips")

    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        for record in wanted:
            file_name = PurePosixPath(record["file_name"]).name
            source = download(
                f"{REPO_URL}/{subset.directory}/{record['file_name']}",
                cache_dir / name / subset.category / file_name,
                client=client,
                sha256=checksums.get(file_name, ""),
            )
            builder.add(
                f"{name}:{record['id']}",
                load_audio(source),
                language=LANGUAGE,
                text=record["text"],
                text_has_casing_punct=True,
                group=f"{name}:{_sentence(name, record['id'])}",
                group_kind="sentence",
                source_split=subset.directory,
                license=LICENSE,
                category=subset.category,
            )
    return len(wanted)


def _read_metadata(name: str, path: Path) -> list[dict]:
    """One JSON object per clip; a line without the fields used here stops preparation."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        missing = sorted({"id", "file_name", "text", "language"} - set(record))
        if missing:
            raise DownloadError(f"{name}: {path.name} has a line without {', '.join(missing)}")
        records.append(record)
    return records


def _sentence(name: str, clip_id: str) -> str:
    """The FLEURS sentence of a clip id, or a loud failure for an id of another shape."""
    match = SENTENCE.match(clip_id)
    if match is None:
        raise DownloadError(f"{name}: {clip_id} does not name its FLEURS sentence")
    return match.group(1)


def _published_checksums(
    name: str, subset: Subset, wanted: list[dict], cache: Path
) -> dict[str, str]:
    """sha256 of every audio file the subset needs, from the Hub listing.

    The listing is asked for once per directory of the subset rather than once
    per file, and only when a file still has to be downloaded: a file already in
    the cache was checked when it arrived. A file the Hub publishes no checksum
    for stops preparation instead of being downloaded unverified.
    """
    to_download = [
        record
        for record in wanted
        if not (cache / subset.category / PurePosixPath(record["file_name"]).name).is_file()
    ]
    if not to_download:
        return {}
    directories = sorted({str(PurePosixPath(record["file_name"]).parent) for record in wanted})
    checksums: dict[str, str] = {}
    for directory in directories:
        checksums.update(huggingface_sha256_map(REPO, REVISION, f"{subset.directory}/{directory}"))
    missing = [
        record["file_name"]
        for record in wanted
        if not checksums.get(PurePosixPath(record["file_name"]).name)
    ]
    if missing:
        raise DownloadError(
            f"{name}: the Hub lists no checksum for {len(missing)} audio files (first {missing[0]})"
        )
    return checksums
