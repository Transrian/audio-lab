"""Downloading and unpacking datasets.

Datasets are large and their upstream layout differs, so the only thing shared
here is the plumbing: a download that lands atomically and is skipped when it
already exists, and a helper that turns one member of an archive into a clip.
"""

from __future__ import annotations

import io
import shutil
import tarfile
import tempfile
from pathlib import Path

import httpx

from audiolab.core.audio import Audio, load_audio


class DownloadError(RuntimeError):
    """Raised when a source cannot be downloaded or does not have the expected shape."""


def download(url: str, destination: Path, *, client: httpx.Client | None = None) -> Path:
    """Download ``url`` to ``destination``, skipping it when it is already there.

    The file appears under its final name only once it is complete, so an
    interrupted download never looks like a finished one. A ``url`` that is
    already a local file is copied instead of fetched, which is how the tests
    exercise preparation without network access.
    """
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    local = Path(url)
    if local.is_file():
        shutil.copyfile(local, destination)
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
    return destination


def open_archive(archive: Path) -> tarfile.TarFile:
    """Open a ``.tar.gz`` for streaming: it is read once, entry by entry."""
    return tarfile.open(archive, mode="r|gz")


def audio_from_archive(handle, member: tarfile.TarInfo) -> Audio:
    """Read one audio member of an open archive as 16 kHz mono float32."""
    data = handle.extractfile(member)
    if data is None:
        raise DownloadError(f"{member.name} is not a readable file")
    return load_audio(io.BytesIO(data.read()))
