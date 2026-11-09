"""Downloading and unpacking datasets.

Datasets are large and their upstream layout differs, so the only thing shared
here is the plumbing: a download that lands atomically, is skipped when it
already exists and goes on from where a stopped one ended, and a helper that
turns one member of an archive into a clip.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path

import httpx

from audiolab.core.audio import Audio, load_audio
from audiolab.core.progress import progress
from audiolab.core.stop import raise_if_stopped

HASH_CHUNK = 1 << 20
# What a download is written under until it is complete and checked.
PART_SUFFIX = ".part"


class DownloadError(RuntimeError):
    """Raised when a source cannot be downloaded or does not have the expected shape."""


def download(
    url: str,
    destination: Path,
    *,
    client: httpx.Client | None = None,
    sha256: str = "",
    md5: str = "",
    git_blob_sha1: str = "",
) -> Path:
    """Download ``url`` to ``destination``, skipping it when it is already there.

    The file appears under its final name only once it is complete *and* it
    matched the checksum it was pinned by, so neither an interrupted download nor
    a changed source leaves something that looks finished: the next call fetches
    again instead of taking a bad file from the cache. A ``url`` that is already
    a local file is copied instead of fetched, which is how the tests exercise
    preparation without network access; it goes through the same staging name
    as a fetched file. When a checksum is given — a ``sha256``, the ``md5`` some
    hosts publish instead, or the ``git_blob_sha1`` a Hub publishes for a file it
    keeps as a git object — a file that was just fetched or copied must match it;
    a file already in the cache was checked when it arrived.

    A download lands in ``<destination>.part`` first, and a stop (D-021) or a
    broken connection leaves that part on disk. The next call asks the server for
    the rest only when the file is pinned by a checksum, which proves that the
    two halves are one file; an unpinned file starts again from the first byte.
    """
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    raise_if_stopped(f"download of {destination.name} stopped")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staged = destination.with_name(destination.name + PART_SUFFIX)
    local = Path(url)
    if local.is_file():
        shutil.copyfile(local, staged)
    else:
        own_client = client is None
        http = client or httpx.Client(follow_redirects=True, timeout=120.0)
        resumable = bool(sha256 or md5 or git_blob_sha1)
        try:
            _fetch(http, url, staged, resume=resumable)
        except httpx.HTTPError as error:
            raise DownloadError(f"cannot download {url}: {error}") from error
        finally:
            if own_client:
                http.close()
    try:
        _check(staged, sha256=sha256, md5=md5, git_blob_sha1=git_blob_sha1, source=url)
    except DownloadError:
        # A file that does not match what was pinned must not be left behind:
        # the next preparation would find it in the cache and take it as it is.
        staged.unlink(missing_ok=True)
        raise
    shutil.move(str(staged), str(destination))
    return destination


def _fetch(http: httpx.Client, url: str, staged: Path, *, resume: bool) -> None:
    """Write ``url`` into ``staged``, after the bytes already there when ``resume``.

    A server that ignores the range answers with the whole file, which then
    replaces the part; one that cannot serve that range (the part is not the
    start of the file it holds now) is asked for the whole file, unless it says
    the file is as long as the part: a stop after the last block leaves a whole
    file under the part's name, and the checksum then decides. The download
    stops between two blocks when a stop is asked for, and keeps what it wrote.
    """
    have = staged.stat().st_size if resume and staged.is_file() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with http.stream("GET", url, headers=headers) as response:
        if have and _refused_range_size(response) == have:
            return
        served_from = _range_start(response, have) if have else 0
        if served_from is None:
            whole_again = True
        else:
            whole_again = False
            response.raise_for_status()
            length = response.headers.get("content-length")
            total = served_from + int(length) if length else None
            with (
                staged.open("ab" if served_from else "wb") as handle,
                progress(desc=staged.stem, unit="B", total=total, initial=served_from) as bar,
            ):
                for block in response.iter_bytes(chunk_size=HASH_CHUNK):
                    handle.write(block)
                    bar.update(len(block))
                    raise_if_stopped(f"download of {staged.stem} stopped")
    if whole_again:
        _fetch(http, url, staged, resume=False)


def _range_start(response: httpx.Response, asked: int) -> int | None:
    """Where the body of an answer to a request for the bytes from ``asked`` on starts.

    ``asked`` when the server sent that range, 0 when it sent the whole file,
    ``None`` when it refused the range or sent another one.
    """
    if response.status_code == 416:
        return None
    if response.status_code != 206:
        return 0
    # Content-Range: bytes <first>-<last>/<size>
    first = response.headers.get("content-range", "").removeprefix("bytes ").partition("-")[0]
    return asked if first == str(asked) else None


def _refused_range_size(response: httpx.Response) -> int | None:
    """The size of the file, when a server refused a range past its end.

    It says so as ``416`` with ``Content-Range: bytes */<size>``.
    """
    if response.status_code != 416:
        return None
    size = response.headers.get("content-range", "").removeprefix("bytes */")
    return int(size) if size.isdecimal() else None


def sha256_file(path: Path) -> str:
    """sha256 of a file, read in blocks so a gigabyte stays a gigabyte."""
    return _digest_file(hashlib.sha256(), path)


def md5_file(path: Path) -> str:
    """md5 of a file, read in blocks like :func:`sha256_file`.

    Zenodo, which hosts one of the sources, publishes an md5 instead of a
    sha256. It is used to notice a changed file, not to prove who wrote it.
    """
    return _digest_file(hashlib.md5(usedforsecurity=False), path)


def _digest_file(digest, path: Path) -> str:
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1_file(path: Path) -> str:
    """The sha1 of a file as a git blob, which is how a Hub names a small file.

    A file the Hub does not store through its LFS is named by the sha1 of the git
    object ``blob <size>\0<content>``, so the digest covers the size as well as
    the bytes.
    """
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _check(path: Path, *, sha256: str, md5: str, git_blob_sha1: str, source: str) -> None:
    """Refuse a file that does not have the digest it was pinned by.

    The parameters are the expected digests, named after them so that a call site
    reads the same way as :func:`download`; one that is empty is not checked.
    """
    for digest_of, wanted, name in (
        (sha256_file, sha256, "sha256"),
        (md5_file, md5, "md5"),
        (git_blob_sha1_file, git_blob_sha1, "git blob sha1"),
    ):
        if not wanted:
            continue
        actual = digest_of(path)
        if actual != wanted:
            raise DownloadError(
                f"{source} does not have the expected {name} "
                f"({actual[:12]}... instead of {wanted[:12]}...): the source changed"
            )


def huggingface_sha256_map(
    repo: str, revision: str, path: str, *, client: httpx.Client | None = None
) -> dict[str, str]:
    """sha256 of every file under one Hub directory, as the Hub publishes it.

    A pinned revision makes a repository immutable, and the file listing carries
    the sha256 the Hub itself stores for large files, so a download can be
    checked against the same value anyone else would get. One answer holds a
    thousand files at most, so a bigger directory is followed page by page: a
    file left out of the map would look like a file without a checksum.
    """
    return {
        Path(entry["path"]).name: str((entry.get("lfs") or {}).get("oid", ""))
        for entry in huggingface_entries(repo, revision, path, recursive=False, client=client)
    }


def huggingface_entries(
    repo: str,
    revision: str,
    path: str,
    *,
    recursive: bool = True,
    client: httpx.Client | None = None,
) -> list[dict]:
    """One entry per file under one Hub directory, as the Hub lists it.

    With ``recursive`` the answer holds the files of every subdirectory as well,
    which is what a corpus keeps its meetings in; without it, only the files of
    the directory itself. The path of an entry is its full path inside the
    repository, so a caller can tell which file is which — and which entry is a
    large file, whose sha256 it carries under ``lfs``, rather than a small one it
    stores as a plain git object.
    """
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{path}"
    if recursive:
        url = f"{url}?recursive=true"
    own_client = client is None
    http = client or httpx.Client(follow_redirects=True, timeout=120.0)
    files: list[dict] = []
    try:
        while url:
            try:
                response = http.get(url)
                response.raise_for_status()
                entries = response.json()
            except (httpx.HTTPError, json.JSONDecodeError) as error:
                raise DownloadError(f"cannot list {url}: {error}") from error
            if not isinstance(entries, list):
                raise DownloadError(f"{url} did not answer with a file list")
            files.extend(
                entry for entry in entries if isinstance(entry, dict) and entry.get("path")
            )
            url = _next_page(response)
    finally:
        if own_client:
            http.close()
    return files


def _next_page(response: httpx.Response) -> str:
    """The URL of the next page of a Hub listing, or an empty string.

    The Hub pages a long listing and points at the following one with
    ``Link: <url>; rel="next"``.
    """
    for part in response.headers.get("link", "").split(","):
        url, _, attributes = part.partition(";")
        if 'rel="next"' in attributes:
            return url.strip().strip("<>")
    return ""


def open_archive(archive: Path) -> tarfile.TarFile:
    """Open a ``.tar.gz`` for streaming: it is read once, entry by entry."""
    return tarfile.open(archive, mode="r|gz")


def audio_from_archive(handle, member: tarfile.TarInfo) -> Audio:
    """Read one audio member of an open archive as 16 kHz mono float32."""
    data = handle.extractfile(member)
    if data is None:
        raise DownloadError(f"{member.name} is not a readable file")
    return load_audio(io.BytesIO(data.read()))
