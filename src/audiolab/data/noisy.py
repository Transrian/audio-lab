"""Noisy derived evaluation sets: the same panel clips, in harder acoustics.

A noisy set (``noisy-<source>``) takes every clip of the source dataset's three
frozen panels and mixes it with one deterministic acoustic condition. The text,
the group and the sealing come from the source: a clip sealed there is sealed
here, and a model trained on the source's train data is evaluated on the source's
evaluation clips, degraded but recognisably the same ones. Nothing is drawn from
the manifest outside the panels, and nothing here is random: every choice is a
hash of the clip id, so a second preparation writes byte-identical audio.

The material comes from the noise banks: the MUSAN noise in the eval half and
the real-room RIRs, so a model augmented with the training halves is scored
against the eval halves. Babble is four other speakers at once, taken from the
training-side sets the screening found harmful to mix (``mls-fr-train`` and
``librispeech-en-train``): the babble pools. They are training sets, so a new
training run may not hold their babble utterances: the ``librispeech-pc-en-train``
preparations exclude them, and a training run containing a babble utterance is
refused. The babble of the prepared noisy sets is committed as
``babble_utterances.json`` beside this module, and a preparation recomputes its
choice from the pools and is refused when it no longer equals the committed
list, so a changed pool manifest can never change a noisy set.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from audiolab.core.audio import SAMPLE_RATE, Audio, load_audio
from audiolab.core.paths import data_dir
from audiolab.data.build import ManifestBuilder, require_clips
from audiolab.data.download import DownloadError
from audiolab.data.panels import DEFAULT_PANELS_DIR, PANEL_NAMES, panel_path, read_panel
from audiolab.data.store import Clip, read_manifest

CONDITIONS = (
    "noise-snr10",
    "noise-snr5",
    "noise-snr0",
    "babble-snr5",
    "reverb",
    "reverb-noise-snr10",
)
# Which training-side sets supply babble for which source; a set that switches
# languages takes one speaker from each, in turn.
BABBLES = {
    "fleurs-fr": ("mls-fr-train",),
    "fleurs-en": ("librispeech-en-train",),
    "servicenow-codeswitched-en-fr": ("mls-fr-train", "librispeech-en-train"),
}
RIRS_LICENSE = "Apache-2.0"
# The committed babble of the prepared noisy sets, beside this module: one
# sorted list of corpus utterance ids per noisy clip set. It is read at
# preparation time, not at import, so a fetcher and a training-run check can
# point it at another file (a test does) while the default is the committed one.
BABBLE_FILE = Path(__file__).with_name("babble_utterances.json")
# The key of the file's note; the only value that is not a list of ids.
BABBLE_NOTE_KEY = "note"


def load_babble_utterances(path: Path | str | None = None) -> dict[str, list[str]]:
    """The committed babble lists: one sorted list of corpus utterance ids per noisy clip set.

    The file is a JSON object with a ``note`` and one key per noisy clip set
    (a name of :data:`NOISY_EXPECTED`), each the list of corpus utterance ids —
    public LibriSpeech/MLS ids, no audio, no text — that the set's babble
    condition mixes. A list must hold every id once and be sorted, and an entry
    that is not a string, a key that names no noisy set, a missing or empty
    note, or any other broken shape is refused: a broken pin would silently
    change a noisy preparation or a training exclusion, which is what the pin
    exists to prevent.
    """
    pin = Path(path) if path is not None else BABBLE_FILE
    try:
        data = json.loads(pin.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as failure:
        raise ValueError(f"{pin.name}: unreadable babble list: {failure}") from failure
    if not isinstance(data, dict):
        raise ValueError(f"{pin.name}: the babble list must be an object")
    note = data.get(BABBLE_NOTE_KEY)
    if not isinstance(note, str) or not note.strip():
        raise ValueError(f"{pin.name} holds no note")
    lists: dict[str, list[str]] = {}
    for key, value in data.items():
        if key == BABBLE_NOTE_KEY:
            continue
        if key not in NOISY_EXPECTED:
            raise ValueError(f"{pin.name}: {key!r} is not a noisy clip set")
        if not isinstance(value, list) or any(
            not isinstance(utterance, str) for utterance in value
        ):
            raise ValueError(f"{pin.name}: {key} must be a list of utterance ids")
        if not value:
            raise ValueError(f"{pin.name}: {key} holds no babble")
        if value != sorted(value):
            raise ValueError(f"{pin.name}: {key} is not sorted")
        if len(set(value)) != len(value):
            raise ValueError(f"{pin.name}: {key} holds an utterance twice")
        lists[key] = value
    if not lists:
        raise ValueError(f"{pin.name} holds no noisy clip set")
    return lists


def babble_union(babble: dict[str, list[str]]) -> list[str]:
    """Every babble utterance of the noisy sets: the sorted union, one id once.

    The flat list the training side excludes against: a corpus utterance id is
    in it exactly when some noisy set mixes it as babble.
    """
    return sorted(set().union(*babble.values()))


def _h(purpose: str, clip_id: str) -> int:
    """One deterministic integer from a purpose and a clip id (no random state)."""
    return int(hashlib.sha256(f"{purpose}-{clip_id}".encode()).hexdigest()[:16], 16)


def _condition(clip_id: str) -> str:
    """The acoustic condition of one clip of a noisy set: a hash of its id."""
    return CONDITIONS[_h("noisy-condition", clip_id) % len(CONDITIONS)]


def _source(name: str) -> str:
    if not name.startswith("noisy-") or name not in NOISY_EXPECTED:
        raise DownloadError(f"{name} is not a known noisy dataset")
    return name.removeprefix("noisy-")


def _prepared(dataset: str) -> Path:
    path = data_dir() / dataset
    if not (path / "manifest.jsonl").is_file():
        raise DownloadError(
            f"{dataset} is not prepared (no manifest.jsonl): run 'audiolab data fetch {dataset}'"
        )
    return path


def fetch_noisy(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare one noisy set: the source's panel clips, each in one condition."""
    source = _source(name)
    missing = [
        d
        for d in (*BABBLES[source], source, "musan-noise", "rirs-noises")
        if not (data_dir() / d / "manifest.jsonl").is_file()
    ]
    if missing:
        raise DownloadError(f"{name} needs the prepared datasets: {', '.join(missing)}")
    source_dir = _prepared(source)
    source_clips = {clip.id: clip for clip in read_manifest(source_dir)}
    panel_ids = _panel_union(source)
    unknown = [clip_id for clip_id in sorted(panel_ids) if clip_id not in source_clips]
    if unknown:
        raise DownloadError(f"{source} holds no clip {unknown[0]} that its panels name")
    if verbose:
        print(f"  {name}: {len(panel_ids)} panel clips of {source}")
    noise = _eval_half(_prepared("musan-noise"))
    rirs = _real_rirs(_prepared("rirs-noises"))
    babble = [_babble_pool(data_dir() / pool) for pool in BABBLES[source]]
    # The babble must be exactly this set's committed list: a pool manifest can
    # change (a pool is a training set), and a changed pool would change the
    # mix. The check runs before any output is written. The committed list
    # holds corpus utterance ids, so the dataset prefix of a manifest id is
    # stripped before the check (as in the babble choice itself, where the
    # pool entry's first field is what the committed list names).
    pool_ids = [
        [
            (clip.id.split(":", 1)[1], clip.group)
            for clip in sorted(read_manifest(data_dir() / pool), key=lambda clip: clip.id)
        ]
        for pool in BABBLES[source]
    ]
    check_babble(name, panel_ids, pool_ids)
    with ManifestBuilder(dataset_dir, verbose=verbose, total=len(panel_ids)) as builder:
        for clip_id in sorted(panel_ids):
            original = source_clips[clip_id]
            condition = _condition(clip_id)
            speech = load_audio(source_dir / original.audio)
            mixed, license_ = _mix(clip_id, condition, speech, noise, rirs, babble, original)
            builder.add(
                f"{name}:{clip_id.partition(':')[2]}",
                Audio(samples=mixed, sample_rate=SAMPLE_RATE),
                text=original.text,
                text_has_casing_punct=original.text_has_casing_punct,
                language=original.language,
                group=f"{name}:{original.group.partition(':')[2]}",
                group_kind=original.group_kind,
                source_split=original.source_split,
                license=license_,
                category=condition,
            )
        if verbose:
            per: dict[str, int] = {}
            for clip in builder.clips:
                per[clip.category] = per.get(clip.category, 0) + 1
            counts = ", ".join(f"{c} {per.get(c, 0)}" for c in CONDITIONS)
            print(f"  {name}: {len(builder.clips)} clips ({counts})")
        require_clips(name, len(builder.clips), NOISY_EXPECTED[name])
        builder.write()
    return len(builder.clips)


def _panel_union(source: str) -> set[str]:
    """The ids of the source's three panels; a derived set generates exactly those."""
    ids: set[str] = set()
    for panel_name in PANEL_NAMES:
        path = panel_path(f"{source}-{panel_name}", DEFAULT_PANELS_DIR)
        if not path.is_file():
            raise DownloadError(f"{path} is missing: the source panels must be built first")
        panel = read_panel(path)
        for clip_id in panel.ids:
            if not clip_id.startswith(f"{source}:"):
                raise DownloadError(f"{path} holds {clip_id!r}, not a {source} id")
            ids.add(clip_id)
    return ids


def _eval_half(dataset_dir: Path) -> list[tuple[Path, str]]:
    """The MUSAN noise files of the eval half, as (wav path, license)."""
    out: list[tuple[Path, str]] = []
    for clip in read_manifest(dataset_dir):
        if clip.source_split == "eval":
            out.append((dataset_dir / clip.audio, clip.license))
    return out


def _real_rirs(dataset_dir: Path) -> list[Path]:
    out: list[Path] = []
    for clip in read_manifest(dataset_dir):
        if clip.source_split == "eval" and clip.category == "real-rir":
            out.append(dataset_dir / clip.audio)
    return out


def _babble_pool(dataset_dir: Path) -> tuple[list[tuple[Path, str]], str]:
    """(wav path, group) for every clip of one training-side set, and its licence."""
    clips = sorted(read_manifest(dataset_dir), key=lambda clip: clip.id)
    licenses = {clip.license for clip in clips}
    if len(licenses) != 1:
        raise DownloadError(f"{dataset_dir.name}: a babble pool must hold one licence")
    return [(dataset_dir / clip.audio, clip.group) for clip in clips], licenses.pop()


def _mix(
    clip_id: str,
    condition: str,
    speech: Audio,
    noise: list[tuple[Path, str]],
    rirs: list[Path],
    babble: list[tuple[list[tuple[Path, str]], str]],
    original: Clip,
) -> tuple[np.ndarray, str]:
    samples = speech.samples
    parts: list[str] = []
    if condition.startswith("noise"):
        snr = int(condition.split("snr")[1])
        segment, license_ = _noise_segment(clip_id, len(samples), noise)
        samples = _add_noise(samples, segment, snr)
        parts.append(f"{license_} (noise)")
    elif condition == "babble-snr5":
        segment, license_ = _babble_segment(clip_id, len(samples), babble)
        samples = _add_noise(samples, segment, 5)
        parts.append(f"{license_} (babble)")
    elif condition.startswith("reverb"):
        samples = _reverb(samples, _rir(clip_id, rirs))
        parts.append(f"{RIRS_LICENSE} (reverb)")
        if condition.endswith("snr10"):
            segment, license_ = _noise_segment(clip_id, len(samples), noise)
            samples = _add_noise(samples, segment, 10)
            parts.append(f"{license_} (noise)")
    if np.abs(samples).max() > 0.99:
        samples = samples * (0.99 / np.abs(samples).max())
    license_ = f"{original.license} (speech) + " + " + ".join(parts) if parts else original.license
    return samples.astype("float32", copy=False), license_


def _noise_segment(
    clip_id: str, length: int, noise: list[tuple[Path, str]]
) -> tuple[np.ndarray, str]:
    """One eval-half noise file, at a hash-chosen offset, looped to the speech length."""
    path, license_ = noise[_h("noisy-noise", clip_id) % len(noise)]
    file = load_audio(path).samples
    offset = _h("noisy-noise-offset", clip_id) % len(file)
    segment = file[offset:]
    while len(segment) < length:
        segment = np.concatenate([segment, file[: length - len(segment)]])
    return segment[:length], license_


def _rir(clip_id: str, rirs: list[Path]) -> np.ndarray:
    return load_audio(rirs[_h("noisy-rir", clip_id) % len(rirs)]).samples


def _babble_segment(
    clip_id: str, length: int, babble: list[tuple[list[tuple[Path, str]], str]]
) -> tuple[np.ndarray, str]:
    """Four other speakers summed at equal power, at a hash-chosen offset each.

    With two pools, the speakers alternate between them, so a code-switched clip
    is buried in speech of both languages. The four groups are all different.
    """
    pools = [pool for pool, _ in babble]
    licenses = sorted({license_ for _, license_ in babble})
    mixed = np.zeros(length, dtype=np.float32)
    used: set[str] = set()
    for k in range(4):
        pool = pools[k % len(pools)]
        path, group = _choose_babble(clip_id, k, pool, used)
        file = load_audio(path).samples
        offset = _h(f"noisy-babble-offset-{k}", clip_id) % len(file)
        segment = file[offset:]
        while len(segment) < length:
            segment = np.concatenate([segment, file[: length - len(segment)]])
        segment = segment[:length]
        power = float(np.mean(segment.astype(np.float64) ** 2))
        if power > 0:
            # Equal power: every speaker enters with unit power before the sum.
            mixed = mixed + (segment / np.sqrt(power)).astype(np.float32)
    return mixed, " + ".join(licenses)


def _pick_babble_index(
    clip_id: str, k: int, pool: Sequence[tuple[object, str]], used: set[str]
) -> int:
    """The pool index of the k-th babble clip of a mix: chosen by hash, never a group already used.

    Only the index and the pool's groups are touched, so the choice depends on
    the pool's order and size alone: the mix passes (path, group) entries and
    the check of :func:`chosen_babble_utterances` passes (utterance id, group)
    entries, and both get the same index for the same pool.
    """
    for attempt in range(128):
        index = _h(f"noisy-babble-{k}-{attempt}", clip_id) % len(pool)
        if pool[index][1] not in used:
            return index
    raise DownloadError(f"{clip_id}: the babble pool has fewer than four free groups")


def _choose_babble(
    clip_id: str, k: int, pool: list[tuple[Path, str]], used: set[str]
) -> tuple[Path, str]:
    """The k-th babble clip of a mix: chosen by hash, never a group already used."""
    index = _pick_babble_index(clip_id, k, pool, used)
    path, group = pool[index]
    used.add(group)
    return path, group


def chosen_babble_utterances(
    clip_ids: Iterable[str], pools: Sequence[Sequence[tuple[str, str]]]
) -> set[str]:
    """The corpus utterances one noisy set's babble would mix, recomputed from the pools.

    The pure form of the babble choice of :func:`fetch_noisy`: for every clip
    whose condition is ``babble-snr5``, the four speakers are drawn from
    ``pools`` in the same order, with the same hash and the same group rule as
    the mix itself — the result is the mix's. ``clip_ids`` are the ids the
    fetcher hashes: the source's clip ids (its panel union, the ids of the
    ``noisy-``-set's source set), and ``pools`` the pool manifests as (corpus
    utterance id, group) lists, in the order of :data:`BABBLES`. Nothing is
    read from disk, so a preparation checks the pools it is about to mix
    against the committed list, and a test checks a synthetic one.
    """
    chosen: set[str] = set()
    for clip_id in sorted(clip_ids):
        if _condition(clip_id) != "babble-snr5":
            continue
        used: set[str] = set()
        for k in range(4):
            pool = pools[k % len(pools)]
            utterance, group = pool[_pick_babble_index(clip_id, k, pool, used)]
            used.add(group)
            chosen.add(utterance)
    return chosen


def check_babble(
    name: str, clip_ids: Iterable[str], pools: Sequence[Sequence[tuple[str, str]]]
) -> None:
    """Refuse a noisy preparation whose babble is no longer its committed list.

    A pool is a training set, so its manifest can change, and a changed pool
    changes the babble of the set being prepared: the choice is recomputed from
    the pools and compared with the set's committed list in
    :data:`BABBLE_FILE`. Any difference — and a set with no committed list,
    which cannot be checked — is refused, before any output is written.
    """
    expected = load_babble_utterances().get(name)
    if expected is None:
        raise DownloadError(
            f"{name}: no committed babble list in {BABBLE_FILE.name}: "
            f"the babble of a noisy set cannot be checked against nothing"
        )
    actual = chosen_babble_utterances(clip_ids, pools)
    if actual != set(expected):
        gained = len(actual - set(expected))
        lost = len(set(expected) - actual)
        raise DownloadError(
            f"{name}: the babble the pools would mix no longer equals the committed "
            f"list of {BABBLE_FILE.name} ({gained} utterances new, {lost} gone): "
            f"the pool manifests changed, and a prepared noisy set must keep its babble"
        )


def _add_noise(speech: np.ndarray, noise: np.ndarray, snr_db: int) -> np.ndarray:
    """The noise scaled so the target SNR holds on the active part of the speech."""
    speech_power = _active_speech_power(speech)
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    if noise_power <= 0 or speech_power <= 0:
        raise DownloadError("cannot mix: the speech or the noise is silent")
    scale = np.sqrt(speech_power / (noise_power * 10.0 ** (snr_db / 10)))
    return (speech + noise * scale).astype("float32", copy=False)


def _active_speech_power(samples: np.ndarray) -> float:
    """Mean power of the 20 ms frames within 35 dB of the loudest frame.

    Power measured over the whole clip would let the silence at the ends set the
    SNR; the active frames are what the listener hears.
    """
    frame = SAMPLE_RATE // 50
    if len(samples) < frame:
        return float(np.mean(samples.astype(np.float64) ** 2))
    powers = (
        (samples[: frame * (len(samples) // frame)].astype(np.float64) ** 2)
        .reshape(-1, frame)
        .mean(axis=1)
    )
    active = powers[powers >= powers.max() / 10.0 ** (35 / 10)]
    return float(active.mean())


def _reverb(speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """The speech convolved with the RIR, back to its original length.

    The samples before the RIR's largest value are cut, so the direct path sits
    at time zero and the text keeps its timing; the RIR is normalised to unit
    energy, so the level is set by the speech.
    """
    direct = int(np.abs(rir).argmax())
    rir = rir[direct:].astype(np.float64)
    rir = rir / np.sqrt(np.sum(rir**2))
    length = len(speech)
    n = 1
    while n < length + len(rir) - 1:
        n *= 2
    convolved = np.fft.irfft(np.fft.rfft(speech.astype(np.float64), n) * np.fft.rfft(rir, n), n)
    return convolved[:length].astype("float32", copy=False)


# The panels of each source, frozen with it: a noisy set holds exactly one mix
# per panel clip, so its size is the size of the source's panel union.
NOISY_EXPECTED: dict[str, int] = {
    "noisy-fleurs-fr": 676,
    "noisy-fleurs-en": 647,
    "noisy-servicenow-codeswitched-en-fr": 298,
}
