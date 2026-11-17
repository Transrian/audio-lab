"""Which clips a run trains on, and the manifest NeMo reads them from.

The clips are drawn once, with a fixed seed, from a prepared dataset - or from
each dataset of a mix, with one seed per dataset - and the list is written into
the run directory. The same config therefore trains on the same clips however
often it is run, and ``train.json`` records their hash, so a later reader can
check that a reported result came from the clips it claims.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from audiolab.data.duplicates import duplicate_classes
from audiolab.data.store import Clip, manifest_id_sha256, read_manifest
from audiolab.train.config import DatasetShare

# Shorter clips are usually fragments that hurt more than they teach; a clip
# whose text is empty (silence, noise) has nothing to learn from at all.
MIN_CLIP_SECONDS = 0.3


class TrainDataError(ValueError):
    """The dataset cannot provide the clips this run asks for."""


@dataclass(frozen=True)
class TrainingClips:
    """The clips of one run, and why the others were left out."""

    clips: list[Clip]
    # Clips that were candidates: inside the duration window, with text.
    available: int
    # Clips shorter than MIN_CLIP_SECONDS or longer than the run's ceiling.
    outside_duration: int
    without_text: int

    @property
    def ids_sha256(self) -> str:
        """A hash of the clip ids, which is what the run record keeps."""
        return manifest_id_sha256(self.clips)


def _clips_within(clips: list[Clip], max_seconds: float) -> tuple[list[Clip], list[Clip]]:
    """The run's pool out of a list of clips: (the clips with text, the pool).

    The pool is the usable clips inside the run's duration window; the order of
    the list must not matter, so both are sorted by id. One home for the two
    passes, so a draw and the count of what it draws from can never drift.
    """
    usable = [clip for clip in sorted(clips, key=lambda clip: clip.id) if clip.text.strip()]
    within = [clip for clip in usable if MIN_CLIP_SECONDS <= clip.duration_s <= max_seconds]
    return usable, within


def select_clips(clips: list[Clip], *, count: int, seed: int, max_seconds: float) -> TrainingClips:
    """Draw ``count`` clips, the same ones every time for the same seed.

    The order the dataset happens to be in must not change the answer, so the
    candidates are sorted by id before anything is drawn.
    """
    usable, within = _clips_within(clips, max_seconds)
    without_text = len(clips) - len(usable)
    outside_duration = len(usable) - len(within)
    if len(within) < count:
        raise TrainDataError(
            f"the dataset holds {len(within)} clips between {MIN_CLIP_SECONDS}s and "
            f"{max_seconds}s, but this run asks for {count}; fetch more data or lower "
            "clips or max_clip_seconds"
        )
    drawn = random.Random(seed).sample(within, count)
    return TrainingClips(
        clips=sorted(drawn, key=lambda clip: clip.id),
        available=len(within),
        outside_duration=outside_duration,
        without_text=without_text,
    )


@dataclass(frozen=True)
class MixExclusion:
    """What one row of an exclusive mix lost to the earlier rows, for the record.

    ``pool_before`` and ``pool_after`` are the usable pools the draw sees -
    the same number the run's start log and ``clips_available`` count - before
    and after the subtraction; ``clips_removed`` is every clip of the dataset
    the subtraction dropped, usable or not. ``pool_after`` is what the row's
    draw then drew from.
    """

    pool_before: int
    pool_after: int
    clips_removed: int


@dataclass(frozen=True)
class MixDatasetClips:
    """What one dataset of a mix contributed to the run.

    It holds the draw itself - what a single-dataset run holds, with the counts
    of what it left out - plus the name of the dataset and the directory its
    audio lives in, which is what the manifest of a mix needs per clip. For a
    row of an exclusive mix, ``exclusion`` keeps what the subtraction took off
    it before the draw; for the other rows, and for every non-exclusive mix,
    it is ``None`` and the result is the one the mix always drew.
    """

    name: str
    drawn: TrainingClips
    directory: Path
    exclusion: MixExclusion | None = None


@dataclass(frozen=True)
class MixClips:
    """The clips of a mix run: the per-dataset results and their concatenation."""

    clips: list[Clip]
    datasets: tuple[MixDatasetClips, ...]

    @property
    def ids_sha256(self) -> str:
        """A hash of the clip ids of the whole mix, which is what the run record keeps."""
        return manifest_id_sha256(self.clips)


def mix_seed(seed: int, name: str) -> int:
    """The per-dataset seed of a mix: the same seed for that dataset in every mix.

    Deriving it from the run's seed and the dataset's name - and from nothing
    else - is what makes adding a dataset to a mix leave the clips drawn from
    the other datasets untouched.
    """
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "big")


class _ExclusiveState:
    """What the earlier rows of an exclusive mix hold, accumulated row by row.

    Each row contributes the whole dataset it names - its manifest, not the
    clips it drew - and two populations are kept from it: every clip's audio
    ``sha256``, and, for a dataset whose source holds a corpus of talks, the
    duplicate classes of its clips' talks, per class domain. A domain is the
    corpus the classes are computed over - the PeerTube sources share one,
    NASA holds its own - and a class of one domain is never compared to a
    class of another: group ids collide by accident across corpora.
    """

    def __init__(self) -> None:
        self.sha256: set[str] = set()
        self.classes: dict[str, set[str]] = {}
        self._tables: dict[str, dict[str, str]] = {}

    def _class_table(self, source: str) -> tuple[str, dict[str, str]]:
        """The domain and class table of one talk source's corpus, cached per domain.

        The classes are computed over the pinned listings the preparations
        read, with the very rule of the leakage guard - whose resolvers this
        reuses: a dataset name resolves to the source its clips come from, and
        that source's corpus gives the domain and the talk population.
        """
        from audiolab.train.guard import _class_corpus

        population, domain = _class_corpus(source)
        if domain not in self._tables:
            self._tables[domain] = duplicate_classes(population, domain=domain)
        return domain, self._tables[domain]

    def _source(self, name: str) -> str | None:
        """The talk or NASA source the dataset's clips come from, or ``None``.

        Imported at the call site: the leakage guard imports this module, so
        the import must not run when it loads.
        """
        from audiolab.train.guard import _clip_source

        return _clip_source(name)

    def subtract(
        self, name: str, clips: list[Clip], max_seconds: float
    ) -> tuple[list[Clip], MixExclusion]:
        """The row's pool after dropping what the earlier rows already hold.

        A clip is dropped by the key its own source gives it: the duplicate
        class of its talk, for a dataset with a corpus of talks, and the class
        appears when an earlier row's dataset holds a clip of it; its audio
        ``sha256``, for the other datasets, and it appears when an earlier
        row's dataset holds a clip with it.
        """
        source = self._source(name)
        if source is not None:
            domain, table = self._class_table(source)
            held = self.classes.get(domain)

            def dropped(clip: Clip) -> bool:
                class_id = table.get(clip.group.partition(":")[2])
                return held is not None and class_id is not None and class_id in held
        else:
            shas = self.sha256

            def dropped(clip: Clip) -> bool:
                return clip.sha256 in shas

        before = len(_clips_within(clips, max_seconds)[1])
        remaining = [clip for clip in clips if not dropped(clip)]
        after = len(_clips_within(remaining, max_seconds)[1])
        return remaining, MixExclusion(before, after, len(clips) - len(remaining))

    def absorb(self, name: str, clips: list[Clip]) -> None:
        """Add one row's whole dataset to what the later rows subtract from."""
        self.sha256.update(clip.sha256 for clip in clips)
        source = self._source(name)
        if source is not None:
            domain, table = self._class_table(source)
            held = self.classes.setdefault(domain, set())
            for clip in clips:
                class_id = table.get(clip.group.partition(":")[2])
                if class_id is not None:
                    held.add(class_id)


def select_mix(
    shares: tuple[DatasetShare, ...],
    *,
    seed: int,
    max_seconds: float,
    data_dir: Path | None,
    exclusive: bool = False,
) -> MixClips:
    """Draw a mix run's clips: one draw per dataset, in the config's order.

    Each dataset is drawn with :func:`mix_seed`, so the mix is the union of
    what each dataset would have drawn alone. The result is sorted by id: ids
    are prefixed by their dataset name, so they are unique, and that is
    asserted. The same audio in two datasets is refused: training on it twice
    would count the same clips against two shares.

    With ``exclusive``, the rows subtract each other, in the recipe's order:
    a row's pool drops every clip that a whole dataset of an earlier row
    already holds - the duplicate class of its talk for a dataset whose source
    holds a corpus of talks, its audio ``sha256`` for the other datasets - and
    the row draws its share from what remains. The subtraction reads only the
    earlier rows' manifests and the recipe, so the same recipe over the same
    manifests subtracts and draws the same clips again; what each row lost is
    what the run's record keeps, in each row's :class:`MixExclusion`.
    """
    results: list[MixDatasetClips] = []
    seen = _ExclusiveState() if exclusive else None
    for share in shares:
        directory = dataset_dir(share.name, data_dir)
        manifest = read_manifest(directory)
        pool, exclusion = manifest, None
        if seen is not None:
            pool, exclusion = seen.subtract(share.name, manifest, max_seconds)
            # The subtraction of the later rows runs on the dataset as
            # prepared, not on what this row kept.
            seen.absorb(share.name, manifest)
        try:
            drawn = select_clips(
                pool,
                count=share.clips,
                seed=mix_seed(seed, share.name),
                max_seconds=max_seconds,
            )
        except TrainDataError as failure:
            raise TrainDataError(f"{share.name}: {failure}") from failure
        results.append(
            MixDatasetClips(name=share.name, drawn=drawn, directory=directory, exclusion=exclusion)
        )
    all_clips = [clip for per in results for clip in per.drawn.clips]
    all_clips.sort(key=lambda clip: clip.id)
    ids = [clip.id for clip in all_clips]
    if len(set(ids)) != len(ids):
        counts = Counter(ids)
        repeated = sorted(clip_id for clip_id, number in counts.items() if number > 1)
        raise TrainDataError(f"clip ids are not unique across the mix (first: {repeated[0]})")
    # The same audio prepared twice: one clip in each of two datasets. The ids
    # are unique (they are prefixed by the dataset), so the audio hash is what
    # catches it. A repetition inside one dataset is that dataset's business,
    # not the mix's.
    by_sha: dict[str, dict[str, list[str]]] = {}
    for per in results:
        for clip in per.drawn.clips:
            by_sha.setdefault(clip.sha256, {}).setdefault(per.name, []).append(clip.id)
    duplicates = sorted(
        (sha, per_dataset) for sha, per_dataset in by_sha.items() if len(per_dataset) > 1
    )
    if duplicates:
        listing = "; ".join(
            f"{sha[:12]}…: "
            + " vs ".join(
                f"{name} ({', '.join(clip_ids)})" for name, clip_ids in sorted(per_dataset.items())
            )
            for sha, per_dataset in duplicates[:5]
        )
        raise TrainDataError(
            f"the same audio is in several datasets of the mix ({len(duplicates)} recordings "
            f"appear in more than one dataset, first 5: {listing}); prepare the datasets so "
            "each recording appears in only one of them"
        )
    return MixClips(clips=all_clips, datasets=tuple(results))


def load_training_clips(
    dataset_dir: Path, *, count: int, seed: int, max_seconds: float
) -> TrainingClips:
    """Read a prepared dataset's manifest and draw this run's clips from it."""
    return select_clips(read_manifest(dataset_dir), count=count, seed=seed, max_seconds=max_seconds)


def write_nemo_manifest(path: Path, entries: list[tuple[Clip, Path]]) -> int:
    """Write the manifest NeMo reads: one line per clip, with absolute audio paths.

    ``entries`` is a pair per clip: the clip and the dataset directory its
    audio lives in. A single-dataset run pairs every clip with the one
    directory, a mix run with each clip's own dataset, so one format serves
    both. The order of the entries is the order of the manifest, so a run
    gives it its clips sorted by id.

    NeMo's own manifest format is three fields - where the audio is, how long it
    is, and what was said - and nothing else. The audio path is absolute because
    NeMo opens it from wherever the training process happens to run.
    """
    lines = [
        json.dumps(
            {
                "audio_filepath": str((dataset_dir / clip.audio).resolve()),
                "duration": round(clip.duration_s, 3),
                "text": clip.text,
            },
            ensure_ascii=False,
        )
        for clip, dataset_dir in entries
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


# How long one row of a bank manifest may cover: a perturbation reads its one
# noise file from disk per augmented clip, and this is what keeps that read
# bounded instead of loading a whole multi-minute file for one clip.
BANK_WINDOW_SECONDS = 20.0
# The shortest tail window that reaches the manifest. A tail shorter than this
# on a file that has other windows is dropped: NeMo adds a short noise to a
# random part of the clip instead of looping it, so such a fragment adds a
# moment of noise, not a dose. A file shorter than one window keeps its single
# row, whatever its length: it is the whole noise that row holds.
MIN_TAIL_WINDOW_SECONDS = 5.0


def build_bank_manifest(bank_dir: Path) -> tuple[list[str], str]:
    """The NeMo noise manifest of a prepared bank, for its train half.

    NeMo's perturbations pick a random manifest line per augmented clip and
    read it from disk, one JSON line per row: ``audio_filepath``, ``duration``,
    ``offset``, ``text``. Only rows with ``source_split == "train"`` are
    kept, sorted by id, with the bank's audio as absolute paths: the eval
    half of a bank is what the noisy panels of the suite are scored against,
    and a run that augmented with it would have trained on its test set. A
    long file becomes one row per ``BANK_WINDOW_SECONDS`` window (its
    ``offset`` and ``duration``), so no line ever covers more than a window;
    a tail window shorter than ``MIN_TAIL_WINDOW_SECONDS`` is dropped when the
    file has other windows, and a file shorter than one window keeps its
    single row.

    Returns the lines and the sha256 of the content they make: a run records
    the hash in ``config.json`` before it writes the file, and a resume is
    checked against it, so a bank re-prepared between a stop and a resume is
    refused rather than mixed in silently.
    """
    train = [clip for clip in read_manifest(bank_dir) if clip.source_split == "train"]
    if not train:
        raise TrainDataError(f"{bank_dir} holds no row with source_split == 'train'")
    lines = []
    for clip in sorted(train, key=lambda clip: clip.id):
        audio = str((bank_dir / clip.audio).resolve())
        windows = []
        offset = 0.0
        while offset < clip.duration_s:
            end = min(offset + BANK_WINDOW_SECONDS, clip.duration_s)
            windows.append((offset, round(end - offset, 3)))
            offset = end
        if len(windows) > 1 and windows[-1][1] < MIN_TAIL_WINDOW_SECONDS:
            windows.pop()
        for offset, duration in windows:
            lines.append(
                json.dumps(
                    {
                        "audio_filepath": audio,
                        "duration": duration,
                        "offset": round(offset, 3),
                        "text": "",
                    },
                    ensure_ascii=False,
                )
            )
    content = "\n".join(lines) + "\n"
    return lines, hashlib.sha256(content.encode("utf-8")).hexdigest()


def write_bank_manifest(path: Path, bank_dir: Path) -> tuple[int, str]:
    """Write the bank's manifest to ``path``: the lines of
    :func:`build_bank_manifest`, one per row.

    Returns the number of lines written and the sha256 of the file's content,
    both of which the run record keeps.
    """
    lines, digest = build_bank_manifest(bank_dir)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines), digest


def dataset_dir(name: str, data_dir: Path | None = None) -> Path:
    """Where a prepared dataset lives; refused if nothing has prepared it."""
    from audiolab.core.paths import data_dir as default_data_dir

    root = Path(data_dir) if data_dir else default_data_dir()
    directory = root / name
    if not (directory / "manifest.jsonl").is_file():
        raise TrainDataError(
            f"{directory} holds no manifest.jsonl: prepare the dataset first "
            f"(audiolab data fetch {name})"
        )
    return directory
