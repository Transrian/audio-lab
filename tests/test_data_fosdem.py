"""FOSDEM: the schedule of a year joined to the video files the tree serves.

The tree and the schedule are replaced by fakes, so these tests check which
talks a listing keeps, what attribution a talk carries, and how the two sides
are joined.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx
import pytest

from audiolab.data import fosdem, talk_sources, talks
from audiolab.data.download import DownloadError
from audiolab.data.talks import Talk, read_listing

SOURCE = talk_sources.SOURCES["fosdem-en-train"]
YEAR = 2025


def index(rooms: list[str]) -> str:
    return "".join(f'<a href="{room}/">{room}</a>' for room in rooms)


def room_page(files: list[str]) -> str:
    return "".join(f'<a href="{name}">{name}</a>' for name in files)


def event(
    event_id: str,
    slug: str,
    *,
    title: str = "A talk",
    duration: str = "00:50",
    language: str = "",
    room: str = "H.1301 (Cornil)",
    persons: tuple[str, ...] = ("Ada Lovelace",),
    videos: tuple[str, ...] = (),
) -> str:
    people = "".join(f"<person>{name}</person>" for name in persons)
    links = "".join(f'<link href="{url}">Video</link>' for url in videos)
    return (
        f'<event id="{event_id}"><duration>{duration}</duration>'
        f"<room>{room}</room><slug>{slug}</slug>"
        f"<title>{title}</title><language>{language}</language>"
        f"<persons>{people}</persons><links>{links}</links></event>"
    )


def schedule(*events: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?><schedule><day>'
        + "".join(events)
        + "</day></schedule>"
    )


def tree_client(rooms: dict[str, list[str]], schedules: dict[int, str]) -> httpx.Client:
    """A client for a fake video tree and a fake per-year schedule."""

    def answer(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/schedule/xml"):
            year = int(url.split("/")[3])
            return httpx.Response(200, text=schedules[year])
        path = url.split(f"{fosdem.VIDEO_TREE}/")[1].strip("/").split("/")
        room = path[1] if len(path) > 1 else ""
        return httpx.Response(
            200, text=room_page(rooms.get(room, [])) if room else index(list(rooms))
        )

    return httpx.Client(transport=httpx.MockTransport(answer))


def test_a_talk_is_listed_only_when_the_tree_serves_the_file_the_schedule_names():
    slug = f"fosdem-{YEAR}-9001-a-talk"
    http = tree_client(
        {"h1301": [f"{slug}.mp4"]},
        {
            YEAR: schedule(
                event("9001", slug, videos=(f"{fosdem.VIDEO_TREE}/{YEAR}/h1301/{slug}.mp4",)),
                event(
                    "9002",
                    "fosdem-2025-9002-never-published",
                    videos=(
                        f"{fosdem.VIDEO_TREE}/{YEAR}/h1301/fosdem-2025-9002-never-published.mp4",
                    ),
                ),
                event("9003", "fosdem-2025-9003-no-recording"),
            )
        },
    )

    with http:
        talks = fosdem.list_year(http, YEAR)

    assert [talk.id for talk in talks] == [f"{YEAR}-{slug}"]
    assert talks[0].url == f"{fosdem.VIDEO_TREE}/{YEAR}/h1301/{slug}.mp4"


def test_the_mp4_is_preferred_over_the_webm_and_the_av1_one():
    slug = f"fosdem-{YEAR}-9001-a-talk"
    base = f"{fosdem.VIDEO_TREE}/{YEAR}/h1301/{slug}"
    root = ET.fromstring(
        schedule(event("9001", slug, videos=(f"{base}.webm", f"{base}.av1.webm", f"{base}.mp4")))
    )
    element = next(root.iter("event"))
    assert fosdem.video_links(element) == [f"{base}.mp4", f"{base}.webm", f"{base}.av1.webm"]

    http = tree_client(
        {"h1301": [f"{slug}.mp4", f"{slug}.webm", f"{slug}.av1.webm"]},
        {
            YEAR: schedule(
                event("9001", slug, videos=(f"{base}.webm", f"{base}.av1.webm", f"{base}.mp4"))
            )
        },
    )
    with http:
        listed = fosdem.list_year(http, YEAR)
    assert [one.url for one in listed] == [f"{base}.mp4"]


@pytest.mark.parametrize(
    ("fields", "kept"),
    [
        ({}, True),
        ({"language": "en"}, True),
        ({"language": "fr"}, False),
        ({"duration": "00:09"}, False),
        ({"duration": "00:10"}, True),
        ({"duration": "nonsense"}, False),
        ({"title": ""}, False),
    ],
)
def test_the_filter_of_the_source_says_which_talks_a_listing_may_hold(fields, kept):
    slug = f"fosdem-{YEAR}-9001-a-talk"
    mp4 = f"{fosdem.VIDEO_TREE}/{YEAR}/h1301/{slug}.mp4"
    http = tree_client(
        {"h1301": [f"{slug}.mp4"]}, {YEAR: schedule(event("9001", slug, videos=(mp4,), **fields))}
    )
    with http:
        talks = fosdem.list_year(http, YEAR)
    assert bool(talks) is kept


def test_a_listed_talk_carries_the_attribution_cc_by_asks_for():
    slug = f"fosdem-{YEAR}-9001-a-talk"
    mp4 = f"{fosdem.VIDEO_TREE}/{YEAR}/h1301/{slug}.mp4"
    http = tree_client(
        {"h1301": [f"{slug}.mp4"]},
        {
            YEAR: schedule(
                event(
                    "9001",
                    slug,
                    title="  A   talk about  Debian ",
                    persons=("Ada Lovelace", "Grace Hopper"),
                    videos=(mp4,),
                )
            )
        },
    )
    with http:
        talk = fosdem.list_year(http, YEAR)[0]

    assert talk.id == f"{YEAR}-{slug}"
    assert talk.title == "A talk about Debian"
    assert talk.author == "Ada Lovelace, Grace Hopper"
    assert talk.channel == "H.1301 (Cornil)"
    assert talk.duration_s == 3000 and talk.license == fosdem.LICENSE
    assert talk.url == mp4 and talk.instance == "", "the listed URL already is the audio file"


def test_a_year_without_a_single_video_file_says_so_instead_of_listing_nothing():
    http = tree_client({}, {YEAR: schedule()})
    with http, pytest.raises(DownloadError, match="lists no video file"):
        fosdem.video_files(http, YEAR)


@pytest.mark.parametrize(
    ("duration", "seconds"), [("00:10", 600), ("01:00", 3600), ("1:30", 5400), ("", 0), ("10", 0)]
)
def test_a_schedule_duration_is_read_as_hours_and_minutes(duration, seconds):
    assert fosdem.slot_seconds(duration) == seconds


def test_the_reading_order_of_the_source_spreads_over_years_and_rooms():
    rooms = ("h1301", "ua2114")
    listed = [
        Talk(
            id=f"fosdem-{year}-{index}",
            url=f"{fosdem.VIDEO_TREE}/{year}/{room}/{index}.mp4",
            title="A talk",
            duration_s=1800,
            license=fosdem.LICENSE,
            author="Someone",
            channel=room,
        )
        for year in (2023, 2024)
        for room in rooms
        for index in range(3)
    ]

    order = talks.reading_order(listed, seed=1, spread_by=fosdem.spread_key)

    assert len(order) == 12
    assert len({fosdem.spread_key(one) for one in order[:4]}) == 4, "one per year and room first"
    assert all(fosdem.spread_key(one).startswith(("2023/", "2024/")) for one in order)


def test_the_pinned_listing_holds_the_talks_the_tree_served():
    listed_on, talks = read_listing(SOURCE)

    assert listed_on and len(talks) == len({talk.id for talk in talks}) > 0
    assert all(talk.license == fosdem.LICENSE for talk in talks)
    assert all(talk.url.startswith(f"{fosdem.VIDEO_TREE}/") for talk in talks)
    assert all(talk.duration_s >= fosdem.MIN_TALK_SECONDS for talk in talks)
    assert all(talk.title and talk.channel for talk in talks)
    assert sum(1 for talk in talks if not talk.author) < 20, "a few talks name no speaker"
    assert all(talk.id.startswith(talk.url.split("/")[3] + "-") for talk in talks)
    years = {int(talk.url.split("/")[3]) for talk in talks}
    assert years <= set(fosdem.YEARS) and len(years) > 1, "several years, all listed ones"
    assert len({talk.channel for talk in talks}) > 10, "many rooms"
