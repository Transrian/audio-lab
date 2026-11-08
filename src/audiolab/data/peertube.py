"""The PeerTube videos that ``peertube-fr-train`` is cut from, and their audio.

A PeerTube instance is a live catalogue: videos are published, edited and
deleted, and its REST API answers with whatever exists on the day it is asked.
A training source has to be reproducible, so the list of usable videos is made
once, by ``audiolab data list-peertube``, and committed as
``peertube_listing.json``: metadata only (uuid, URL, title, duration, licence,
channel, account), with the date it was listed on. A preparation reads that
file and never the catalogue.

A video is usable when its licence allows a published model to be trained on it
and redistributed with attribution — CC BY or CC0, nothing share-alike or
non-commercial — when it is in French, and when the instance hosts it itself
rather than mirroring another instance.
"""

from __future__ import annotations

import email.utils
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

from audiolab.core.audio import SAMPLE_RATE
from audiolab.data.download import DownloadError, download

INSTANCE = "https://videos.capitoledulibre.org"
LISTING_PATH = Path(__file__).with_name("peertube_listing.json")
LANGUAGE = "fr"
# PeerTube licence ids and the license a clip of each is recorded under.
# PeerTube names the licence ("Attribution") without a Creative Commons version.
LICENSES = {1: "CC-BY", 7: "CC0-1.0"}

# The catalogue is read one page at a time, with a pause between requests: the
# instance answers 429 to a client that asks faster.
PAGE_SIZE = 100
PAUSE_S = 1.0
RETRIES = 6
TIMEOUT_S = 120.0


@dataclass(frozen=True)
class Video:
    """One listed video, with what CC BY requires a clip of it to carry."""

    uuid: str
    url: str
    title: str
    duration_s: int
    licence_id: int
    channel: str
    account: str

    @property
    def license(self) -> str:
        return LICENSES[self.licence_id]


def usable(row: dict, instance: str = INSTANCE) -> bool:
    """Whether one video of the API's answer may be used by this source."""
    licence = (row.get("licence") or {}).get("id")
    language = (row.get("language") or {}).get("id")
    hosted_here = bool(row.get("isLocal")) and str(row.get("url", "")).startswith(instance + "/")
    return licence in LICENSES and language == LANGUAGE and hosted_here


def video_of(row: dict) -> Video:
    """The listing entry of one video of the API's answer."""
    return Video(
        uuid=row["uuid"],
        url=row["url"],
        title=row["name"],
        duration_s=int(row["duration"]),
        licence_id=int(row["licence"]["id"]),
        channel=row["channel"]["displayName"],
        account=row["account"]["displayName"],
    )


def list_videos(http: httpx.Client, instance: str = INSTANCE) -> list[Video]:
    """Every usable video of the instance, by uuid.

    The API is asked to filter already; :func:`usable` checks every answer again,
    because a filter the server ignores would otherwise let anything in.
    """
    filters = {
        "languageOneOf": LANGUAGE,
        "licenceOneOf": list(LICENSES),
        "isLocal": "true",
        "sort": "publishedAt",
        "count": PAGE_SIZE,
    }
    rows: list[dict] = []
    total = None
    while total is None or len(rows) < total:
        page = get_json(http, f"{instance}/api/v1/videos", {**filters, "start": len(rows)})
        total = int(page["total"])
        if not page["data"]:
            break
        rows.extend(page["data"])
    return sorted((video_of(row) for row in rows if usable(row, instance)), key=lambda v: v.uuid)


def get_json(http: httpx.Client, url: str, params: dict | None = None) -> dict:
    """One API request, paced, waiting as long as the instance asks when it refuses."""
    for attempt in range(1, RETRIES + 1):
        time.sleep(PAUSE_S)
        try:
            response = http.get(url, params=params)
        except httpx.TransportError as failure:
            if attempt == RETRIES:
                raise DownloadError(f"cannot reach {url}: {failure}") from failure
            continue
        if response.status_code in (429, 503):
            time.sleep(retry_after_s(response.headers.get("retry-after"), 5.0 * attempt))
            continue
        if response.status_code >= 400:
            raise DownloadError(f"{url}: HTTP {response.status_code}")
        return response.json()
    raise DownloadError(f"{url}: still refused after {RETRIES} attempts")


def retry_after_s(value: str | None, default: float) -> float:
    """How long a ``Retry-After`` header asks to wait: seconds, or an HTTP date."""
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return default
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def write_listing(videos: list[Video], listed_on: str, path: Path = LISTING_PATH) -> None:
    listing = {
        "instance": INSTANCE,
        "listed_on": listed_on,
        "filter": {"licence_ids": list(LICENSES), "language": LANGUAGE, "local": True},
        "videos": [asdict(video) for video in videos],
    }
    path.write_text(json.dumps(listing, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def read_listing(path: Path = LISTING_PATH) -> tuple[str, list[Video]]:
    """The date of the pinned listing and its videos, their licences checked again."""
    listing = json.loads(path.read_text(encoding="utf-8"))
    videos = [Video(**row) for row in listing["videos"]]
    unusable = [video.uuid for video in videos if video.licence_id not in LICENSES]
    if unusable:
        raise DownloadError(f"{path}: videos under another licence: {unusable[:3]}")
    return listing["listed_on"], videos


def reading_order(videos: list[Video], seed: int) -> list[Video]:
    """The listed videos in a seeded order, the same on every machine.

    The instance publishes a conference year after another, so the order of the
    listing would spend the clip cap on one year; a shuffle with a fixed seed
    spreads it over the whole catalogue and stays reproducible.
    """
    ordered = sorted(videos, key=lambda video: video.uuid)
    random.Random(seed).shuffle(ordered)
    return ordered


def smallest_file_url(http: httpx.Client, video: Video, instance: str = INSTANCE) -> str:
    """The download URL of the smallest file the instance serves for a video.

    It is asked at download time rather than pinned: an instance may transcode a
    video again under new file names, and every file holds the same audio.
    """
    details = get_json(http, f"{instance}/api/v1/videos/{video.uuid}")
    files = list(details.get("files") or [])
    for playlist in details.get("streamingPlaylists") or []:
        files.extend(playlist.get("files") or [])
    if not files:
        raise DownloadError(f"{video.url} serves no downloadable file")
    return min(files, key=lambda file: file["size"])["fileUrl"]


def fetch_audio(http: httpx.Client, video: Video, destination: Path) -> Path:
    """Download one video and keep only its audio, as a 16 kHz mono WAV.

    The video file is downloaded first (reading it remotely through ffmpeg is
    much slower) and deleted as soon as its audio is out: most of its bytes are
    pixels.
    """
    if destination.is_file():
        return destination
    video_file = destination.with_suffix(".video")
    download(smallest_file_url(http, video), video_file, client=http)
    partial = destination.with_suffix(".partial.wav")
    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video_file),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(partial),
    ]  # fmt: skip
    try:
        finished = subprocess.run(command, capture_output=True, text=True)
    finally:
        video_file.unlink(missing_ok=True)
    if finished.returncode != 0:
        partial.unlink(missing_ok=True)
        raise DownloadError(f"ffmpeg failed on {video.url}: {finished.stderr.strip()[:200]}")
    partial.rename(destination)
    return destination


def http_client() -> httpx.Client:
    return httpx.Client(follow_redirects=True, timeout=TIMEOUT_S)
