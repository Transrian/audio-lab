"""Shared helpers for turning downloaded files into a manifest."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

from audiolab.core.audio import Audio, pcm_sha256, write_wav
from audiolab.core.progress import progress, say
from audiolab.core.stop import raise_if_stopped
from audiolab.data.download import DownloadError
from audiolab.data.store import Clip, clip_from_dict, write_manifest

# What a preparation has written so far: a key line, then one manifest line per
# finished clip. It becomes manifest.jsonl when the preparation is complete.
PARTIAL_NAME = "manifest.partial.jsonl"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def file_name_for(clip_id: str) -> str:
    """A file name for a clip id: the id may contain ':' and '/', the file may not."""
    return _UNSAFE.sub("_", clip_id) + ".wav"


def require_clips(dataset: str, count: int, expected: int) -> None:
    """Fail loudly when a prepared split is not the size its source publishes.

    A short preparation is the quiet way a dataset changes under an evaluation:
    a moved archive, a truncated download or a renamed text file all give fewer
    clips and a number that looks usable. Tests pass a small ``expected``.
    """
    if count != expected:
        raise DownloadError(
            f"{dataset}: prepared {count} clips but {expected} were expected; "
            "the source is not the split this panel was built from"
        )


class ManifestBuilder:
    """Collects prepared clips, writes each WAV at once, and keeps what a stop leaves.

    Every clip is written to its WAV and then to ``manifest.partial.jsonl``, one
    manifest line at a time, so a preparation that is stopped (D-021) or killed
    keeps the clips it finished. The next preparation of the same dataset takes
    a clip from that file instead of decoding, writing and hashing it again, if
    its WAV is still there and every field it would get now is the field it got
    then; the audio itself is not compared, so the file is only read back by the
    code that wrote it (:func:`preparation_key`). :meth:`write` turns it into the
    manifest, which therefore still never advertises a clip that is not on disk.
    """

    def __init__(self, dataset_dir: Path, *, verbose: bool, total: int | None = None) -> None:
        self.dataset_dir = dataset_dir
        self.verbose = verbose
        self.clips: list[Clip] = []
        self.partial = dataset_dir / PARTIAL_NAME
        self.key = preparation_key()
        self.done = _read_partial(self.partial, self.key)
        if not self.done:
            # Nothing in it can be taken: a new one starts with this preparation.
            self.partial.unlink(missing_ok=True)
        self.bar = progress(desc=dataset_dir.name, unit="clip", total=total)

    def prepared(self, clip_id: str, **fields) -> Clip | None:
        """The clip a stopped preparation left with these fields, if its WAV is there."""
        done = self.done.get(clip_id)
        if done is None or not (self.dataset_dir / done.audio).is_file():
            return None
        audio = {"audio": done.audio, "sha256": done.sha256, "duration_s": done.duration_s}
        return done if Clip(id=clip_id, **audio, **fields) == done else None

    def add(self, clip_id: str, audio: Audio | Callable[[], Audio], **fields) -> None:
        """Store one clip's audio and remember its manifest line.

        ``audio`` can be a function that reads it, which is then called only
        when the clip is not already prepared. The preparation stops here, with
        the clip kept, when a stop was asked for.
        """
        clip = self.prepared(clip_id, **fields)
        if clip is None:
            relative = Path("audio") / file_name_for(clip_id)
            if callable(audio):
                audio = audio()
            write_wav(self.dataset_dir / relative, audio)
            clip = Clip(
                id=clip_id,
                audio=str(relative),
                sha256=pcm_sha256(self.dataset_dir / relative),
                duration_s=round(audio.duration_s, 3),
                **fields,
            )
            self._keep(clip)
        self.clips.append(clip)
        self.bar.update()
        if self.verbose and len(self.clips) % 500 == 0:
            say(f"  ... {len(self.clips)} clips prepared")
        raise_if_stopped(f"preparation of {self.dataset_dir.name} stopped")

    def write(self) -> Path:
        self.bar.close()
        path = write_manifest(self.dataset_dir, self.clips)
        self.partial.unlink(missing_ok=True)
        return path

    def _keep(self, clip: Clip) -> None:
        """Append one finished clip to the partial manifest, after its WAV is complete."""
        if not self.partial.is_file():
            self.partial.parent.mkdir(parents=True, exist_ok=True)
            self.partial.write_text(json.dumps({"key": self.key}) + "\n", encoding="utf-8")
        with self.partial.open("a", encoding="utf-8") as handle:
            handle.write(clip.manifest_line() + "\n")


def preparation_key() -> str:
    """What the audio of a prepared clip depends on besides its source: the code.

    The source files are pinned by their checksums and the fields of a clip are
    compared one by one, but the samples a WAV holds come from the decoders and
    the resampler, and from how this package cuts and converts. The key is the
    hash of the package's data code, of the audio core, and of the versions of
    the libraries that decode and resample, so any change to them prepares again
    from the start. It is broad on purpose: an edit to any ``data/*.py``, even
    one a given dataset does not run, throws its partial clips away, which costs
    time and never gives a wrong clip. It does not cover the version of
    ``ffmpeg``, which decodes the talk and ICSI sources outside Python.
    """
    digest = hashlib.sha256()
    package = Path(__file__).parent.parent
    for path in sorted([*package.glob("data/*.py"), *package.glob("data/*.json")]) + [
        package / "core" / "audio.py"
    ]:
        digest.update(f"{path.relative_to(package)}\0".encode())
        digest.update(path.read_bytes())
    for library in ("numpy", "soundfile", "soxr"):
        digest.update(f"{library} {version(library)}\0".encode())
    return digest.hexdigest()


def _read_partial(path: Path, key: str) -> dict[str, Clip]:
    """The clips a stopped preparation finished, by id; none if other code wrote them.

    A torn last line, from a process killed in the middle of writing it, is
    skipped: its clip is prepared again.
    """
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        if json.loads(lines[0]) != {"key": key}:
            return {}
    except (IndexError, json.JSONDecodeError):
        return {}
    done = {}
    for line in lines[1:]:
        try:
            clip = clip_from_dict(json.loads(line))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        done[clip.id] = clip
    return done
