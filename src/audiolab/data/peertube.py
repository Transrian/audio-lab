"""The PeerTube REST API: a live catalogue of videos, read into a pinned listing.

A PeerTube instance is a live catalogue: videos are published, edited and
deleted, and its API answers with whatever exists on the day it is asked. A
training source has to be reproducible, so the list of usable videos is made
once, by ``audiolab data list-talks``, and committed as a JSON file — metadata
only, with the date it was listed on. A preparation reads that file and never
the catalogue.

What may be listed is the caller's decision, written down as a :class:`Filter`:
the language, the licences that allow a published model to be trained on the
video (CC BY or CC0, nothing share-alike or non-commercial, D-019), how short a
video may be, and the channels or accounts to leave out. The API is asked to
filter already, and every answer is checked again here, because a filter an
instance ignores would otherwise let anything in.
"""

from __future__ import annotations

import email.utils
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from audiolab.data.download import DownloadError
from audiolab.data.talks import Talk

# PeerTube licence ids and the license a clip of each is recorded under.
# PeerTube names the licence ("Attribution") without a Creative Commons version.
LICENSES = {1: "CC-BY", 7: "CC0-1.0"}

# The catalogue is read one page at a time, with a pause between requests: an
# instance answers 429 to a client that asks faster.
PAGE_SIZE = 100
PAUSE_S = 1.0
RETRIES = 6
# What an instance answers when it is busy, not when the request is wrong: a 500
# from a large instance is a query that timed out under load, and the same page
# answers a minute later.
RETRY_STATUSES = (429, 500, 502, 503, 504)


@dataclass(frozen=True)
class Filter:
    """Which videos of one instance a listing of talks may hold.

    ``language_unset_ok`` keeps the videos an instance never tagged: some
    instances tag almost nothing, and the caller then has to know what language
    those videos are in from the collection they sit in. ``name_must_hold`` and
    ``name_must_not_hold`` are whole words of a channel or account display name
    ("debconf" keeps a conference, "musique" drops a music channel); a word
    never matches inside another one, so ``es`` drops ``DebConf ES`` and not
    ``Andreas``.
    """

    language: str
    licence_ids: tuple[int, ...] = (1, 7)
    language_unset_ok: bool = False
    min_duration_s: int = 0
    name_must_hold: str = ""
    name_must_not_hold: tuple[str, ...] = ()


def usable(row: dict, instance: str, filter: Filter) -> bool:
    """Whether one video of the API's answer may be used by this source."""
    if (row.get("licence") or {}).get("id") not in filter.licence_ids:
        return False
    language = (row.get("language") or {}).get("id")
    if language != filter.language and not (filter.language_unset_ok and language is None):
        return False
    if not row.get("isLocal") or not str(row.get("url", "")).startswith(instance + "/"):
        return False
    if int(row.get("duration") or 0) < filter.min_duration_s:
        return False
    words = words_of(row["channel"]["displayName"]) | words_of(row["account"]["displayName"])
    if filter.name_must_hold and filter.name_must_hold not in words:
        return False
    return not words.intersection(filter.name_must_not_hold)


def words_of(name: str) -> set[str]:
    """The words of a channel or account name, lowercased, accented letters kept.

    A word is a run of letters, digits and underscores, so ``théâtre`` stays one
    word and ``nikkojazz`` is not ``jazz``: the filter compares whole words.
    """
    return set(re.findall(r"\w+", name.lower()))


def license_of(row: dict) -> str:
    return LICENSES.get((row.get("licence") or {}).get("id"), "")


def talk_of(row: dict, instance: str) -> Talk:
    """The listing entry of one video of the API's answer."""
    return Talk(
        id=row["uuid"],
        url=row["url"],
        title=row["name"],
        duration_s=int(row["duration"]),
        license=license_of(row),
        author=row["account"]["displayName"],
        channel=row["channel"]["displayName"],
        instance=instance,
    )


def list_instance(http: httpx.Client, instance: str, filter: Filter) -> list[Talk]:
    """Every usable video one instance hosts itself, by uuid."""
    filters: dict[str, object] = {
        "licenceOneOf": list(filter.licence_ids),
        "isLocal": "true",
        "sort": "publishedAt",
        "count": PAGE_SIZE,
    }
    if not filter.language_unset_ok:
        filters["languageOneOf"] = filter.language
    rows: list[dict] = []
    total = None
    while total is None or len(rows) < total:
        page = get_json(http, f"{instance}/api/v1/videos", {**filters, "start": len(rows)})
        total = int(page["total"])
        if not page["data"]:
            break
        rows.extend(page["data"])
    return sorted(
        (talk_of(row, instance) for row in rows if usable(row, instance, filter)),
        key=lambda talk: talk.id,
    )


def list_instances(http: httpx.Client, instances: tuple[str, ...], filter: Filter) -> list[Talk]:
    """Every usable video of several instances, by uuid, each knowing its instance."""
    talks = [talk for instance in instances for talk in list_instance(http, instance, filter)]
    return sorted(talks, key=lambda talk: talk.id)


def smallest_file_url(http: httpx.Client, talk: Talk) -> str:
    """The download URL of the smallest file the instance serves for a talk.

    It is asked at download time rather than pinned: an instance may transcode a
    video again under new file names, and every file holds the same audio.

    A file is taken only once it has answered: PeerTube lists the files a video
    was transcoded into, and an instance that has moved to streaming may have
    deleted the plain files while still naming them, which is what
    ``test_a_file_the_instance_no_longer_serves_is_not_taken`` pins. Each
    candidate is asked for its first byte, so a talk costs one small request more
    and not a failed download.
    """
    details = get_json(http, f"{talk.instance}/api/v1/videos/{talk.id}")
    files = list(details.get("files") or [])
    for playlist in details.get("streamingPlaylists") or []:
        files.extend(playlist.get("files") or [])
    if not files:
        raise DownloadError(f"{talk.url} serves no downloadable file")
    for file in sorted(files, key=lambda file: file["size"]):
        if serves(http, file["fileUrl"]):
            return file["fileUrl"]
    raise DownloadError(f"{talk.url} serves no file the instance still has")


def serves(http: httpx.Client, url: str) -> bool:
    """Whether the instance answers for a file URL, asked for its first byte only."""
    try:
        response = http.get(url, headers={"Range": "bytes=0-0"})
    except httpx.TransportError:
        return False
    return response.status_code in (200, 206)


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
        if response.status_code in RETRY_STATUSES:
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
