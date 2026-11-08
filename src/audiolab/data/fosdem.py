"""FOSDEM talks: each year's schedule joined to the video files it published.

FOSDEM publishes a talk twice: in the schedule (one XML file per year, with an
``<event>`` per talk carrying its title, speakers, room, language and slot), and
as a recording on a plain HTTP tree of files (``/<year>/<room>/``, ``.mp4`` and
``.webm``). Neither side is enough on its own: the schedule lists talks whose
recording was never published, and a file name says nothing about who spoke. So
a talk is listed when the schedule points at a file that the tree actually
serves, and the schedule is where the metadata comes from.

The recordings are the conference's own, published under CC BY 2.0 BE: the
licence is stated on the tree's own pages, which link to
https://creativecommons.org/licenses/by/2.0/be/deed.en (checked 2026-11-08). CC
BY allows training a published model on the audio and redistributing it with
attribution, so a clip carries its speakers, title, URL and licence (D-019).

Only recent years are listed: the recordings go back to 2003, and a seeded set
spread over seven years already covers every devroom and topic the conference
has. Talks shorter than ten minutes are left out, as in the other talk sources:
a shorter recording cannot carry the clips a batch wants.
"""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET

import httpx

from audiolab.data.download import DownloadError
from audiolab.data.talks import Talk

VIDEO_TREE = "https://video.fosdem.org"
# The schedule of a year; fosdem.org redirects to its archive, which the client follows.
SCHEDULE = "https://fosdem.org/{year}/schedule/xml"

YEARS = (2020, 2021, 2022, 2023, 2024, 2025, 2026)
LANGUAGE = "en"
LICENSE = "CC-BY-2.0-BE"
# The shortest recording a listing may hold, as in the other talk sources.
MIN_TALK_SECONDS = 600

ROOM_LINK = re.compile(r"^[A-Za-z0-9._-]+/$")
VIDEO_FILE = re.compile(r"\.(mp4|webm)$")
HREF = re.compile(r'href="([^"]+)"')
CLOCK = re.compile(r"^(\d+):(\d{2})$")


def list_talks(http: httpx.Client, years: tuple[int, ...] = YEARS) -> list[Talk]:
    """Every listed talk of ``years``, by slug."""
    return sorted((talk for year in years for talk in list_year(http, year)), key=lambda t: t.id)


def list_year(http: httpx.Client, year: int) -> list[Talk]:
    """The talks of one year whose recording the tree actually serves."""
    present = video_files(http, year)
    talks = []
    for event in events(http, year):
        url = next((href for href in video_links(event) if href in present), "")
        if not url or not speaks_english(event):
            continue
        talk = talk_of(event, url)
        if worth_listing(talk):
            talks.append(talk)
    return talks


def video_files(http: httpx.Client, year: int) -> set[str]:
    """The URLs of every video file the tree serves for one year.

    The tree is a directory listing per room, and a room is one day of one
    track, so the rooms are asked one after the other.
    """
    index = f"{VIDEO_TREE}/{year}/"
    rooms = [href for href in links(get_text(http, index)) if ROOM_LINK.match(href)]
    files = {
        f"{index}{room}{href}"
        for room in rooms
        for href in links(get_text(http, f"{index}{room}"))
        if VIDEO_FILE.search(href)
    }
    if not files:
        raise DownloadError(f"{index} lists no video file: the tree changed shape")
    return files


def events(http: httpx.Client, year: int) -> list[dict]:
    """The ``<event>`` elements of one year's schedule."""
    return list(ET.fromstring(get_text(http, SCHEDULE.format(year=year))).iter("event"))


def video_links(event: ET.Element) -> list[str]:
    """The video URLs of one event, the MP4 first: it plays everywhere, an AV1 one may not."""
    hrefs = [
        link.get("href", "")
        for link in event.findall(".//links/link")
        if link.get("href", "").startswith(f"{VIDEO_TREE}/")
        and VIDEO_FILE.search(link.get("href", ""))
    ]
    return sorted(hrefs, key=lambda url: not url.endswith(".mp4"))


def talk_of(event: ET.Element, url: str) -> Talk:
    """One listed talk, with the attribution CC BY asks a clip of it to carry.

    The id starts with the year: FOSDEM publishes a devroom again the next year,
    keeps the slug, and two talks that share a slug would otherwise share a cache
    file and a manifest group.
    """
    slug = (event.findtext("slug") or "").strip()
    year, _ = recorded_in(url)
    return Talk(
        id=f"{year}-{slug}" if slug else "",
        url=url,
        title=" ".join((event.findtext("title") or "").split()),
        duration_s=slot_seconds(event.findtext("duration") or ""),
        license=LICENSE,
        author=", ".join(
            person.text.strip() for person in event.findall(".//persons/person") if person.text
        ),
        channel=(event.findtext("room") or "").strip(),
    )


def speaks_english(event: ET.Element) -> bool:
    """Whether an event is in English or was never tagged, which is the same here.

    Only the recent schedules carry a language; the older ones leave it empty,
    and FOSDEM is held in English.
    """
    return (event.findtext("language") or "").strip() in ("", LANGUAGE)


def slot_seconds(duration: str) -> int:
    """A schedule duration (``HH:MM``, the length of the slot) in seconds."""
    match = CLOCK.match(duration.strip())
    if not match:
        return 0
    return int(match.group(1)) * 3600 + int(match.group(2)) * 60


def worth_listing(talk: Talk) -> bool:
    return talk.duration_s >= MIN_TALK_SECONDS and bool(talk.title) and bool(talk.id)


def links(page: str) -> list[str]:
    """Every link of one HTML directory listing, entity-decoded."""
    return [html.unescape(href) for href in HREF.findall(page)]


def get_text(http: httpx.Client, url: str) -> str:
    """One page of the tree or of a schedule, as text."""
    response = http.get(url)
    if response.status_code >= 400:
        raise DownloadError(f"{url}: HTTP {response.status_code}")
    return response.text


def file_url(http: httpx.Client, talk: Talk) -> str:
    """The audio file of a listed talk: the listing pins the URL the tree serves."""
    return talk.url


def recorded_in(url: str) -> tuple[str, str]:
    """The year and the room a talk's file URL names."""
    _, _, _, year, room, _ = url.split("/", 5)
    return year, room


def spread_key(talk: Talk) -> str:
    """The year and the room a talk was recorded in: the two ways FOSDEM is uneven.

    A year holds a thousand talks and a devroom a handful of them, so a plain
    shuffle over the listing can spend a whole selection on one room of one
    year.
    """
    return "/".join(recorded_in(talk.url))
