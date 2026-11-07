"""Downloading and unpacking datasets.

Datasets are large and their upstream layout differs, so the only thing shared
here is the plumbing: a download that lands atomically and is skipped when it
already exists, and a helper that turns one member of an archive into a clip.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import httpx

from audiolab.core.audio import Audio, load_audio

HASH_CHUNK = 1 << 20


class DownloadError(RuntimeError):
    """Raised when a source cannot be downloaded or does not have the expected shape."""


def download(
    url: str, destination: Path, *, client: httpx.Client | None = None, sha256: str = ""
) -> Path:
    """Download ``url`` to ``destination``, skipping it when it is already there.

    The file appears under its final name only once it is complete, so an
    interrupted download never looks like a finished one. A ``url`` that is
    already a local file is copied instead of fetched, which is how the tests
    exercise preparation without network access. When ``sha256`` is given, a
    file that was just fetched or copied must match it; a file already in the
    cache was checked when it arrived.
    """
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    local = Path(url)
    if local.is_file():
        shutil.copyfile(local, destination)
        _check(destination, sha256, url)
        return destination
    own_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120.0)
    try:
        with http.stream("GET", url) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
                for block in response.iter_bytes(chunk_size=1 << 20):
                    temporary.write(block)
                staged = Path(temporary.name)
    except httpx.HTTPError as error:
        raise DownloadError(f"cannot download {url}: {error}") from error
    finally:
        if own_client:
            http.close()
    shutil.move(str(staged), str(destination))
    _check(destination, sha256, url)
    return destination


def sha256_file(path: Path) -> str:
    """sha256 of a file, read in blocks so a gigabyte stays a gigabyte."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(path: Path, expected: str, source: str) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual != expected:
        raise DownloadError(
            f"{source} does not have the expected sha256 "
            f"({actual[:12]}... instead of {expected[:12]}...): the source changed"
        )


def huggingface_sha256_map(
    repo: str, revision: str, path: str, *, client: httpx.Client | None = None
) -> dict[str, str]:
    """sha256 of every file under one Hub directory, as the Hub publishes it.

    A pinned revision makes a repository immutable, and the file listing carries
    the sha256 the Hub itself stores for large files, so a download can be
    checked against the same value anyone else would get.
    """
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{path}"
    own_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120.0)
    try:
        response = http.get(url)
        response.raise_for_status()
        entries = response.json()
    except (httpx.HTTPError, json.JSONDecodeError) as error:
        raise DownloadError(f"cannot list {url}: {error}") from error
    finally:
        if own_client:
            http.close()
    if not isinstance(entries, list):
        raise DownloadError(f"{url} did not answer with a file list")
    return {
        Path(entry["path"]).name: str(entry.get("lfs", {}).get("oid", ""))
        for entry in entries
        if isinstance(entry, dict) and entry.get("path")
    }


def open_archive(archive: Path) -> tarfile.TarFile:
    """Open a ``.tar.gz`` for streaming: it is read once, entry by entry."""
    return tarfile.open(archive, mode="r|gz")


def audio_from_archive(handle, member: tarfile.TarInfo) -> Audio:
    """Read one audio member of an open archive as 16 kHz mono float32."""
    data = handle.extractfile(member)
    if data is None:
        raise DownloadError(f"{member.name} is not a readable file")
    return load_audio(io.BytesIO(data.read()))
