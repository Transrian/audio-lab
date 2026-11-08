"""FOSDEM: the schedule of a year joined to the video files the tree serves.

The tree and the schedule are replaced by fakes, so these tests check which
talks a listing keeps, what attribution a talk carries, and how the two sides
are joined.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from urllib.parse import unquote

import httpx
import pytest

from audiolab.data import fosdem, talk_listing, talk_sources
from audiolab.data.download import DownloadError
from audiolab.data.talk_listing import Talk, read_listing

SOURCE = talk_sources.SOURCES["fosdem-en-train"]
YEAR = 2025


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(fosdem.time, "sleep", lambda seconds: None)


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
    page: str = "",
) -> str:
    people = "".join(f"<person>{name}</person>" for name in persons)
    links = "".join(f'<link href="{url}">Video</link>' for url in videos)
    return (
        f'<event id="{event_id}"><duration>{duration}</duration>'
        f"<room>{room}</room><slug>{slug}</slug><url>{page}</url>"
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
        room = unquote(path[1]) if len(path) > 1 else ""
        files = {unquote(name): listed for name, listed in rooms.items()}.get(room, [])
        return httpx.Response(200, text=room_page(files) if room else index(list(rooms)))

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

    assert [talk.id for talk in talks] == [f"{YEAR}-9001"]
    assert talks[0].file == f"{fosdem.VIDEO_TREE}/{YEAR}/h1301/{slug}.mp4"


def test_a_room_the_tree_percent_encodes_is_read_and_joined_to_the_schedule():
    """The tree encodes ``H.1301 (Cornil)``; the schedule links to it unencoded."""
    slug = "a_talk"
    http = tree_client(
        {"H.1301%20%28Cornil%29": [f"{slug}.mp4"], "Janson": []},
        {
            YEAR: schedule(
                event(
                    "9001", slug, videos=(f"{fosdem.VIDEO_TREE}/{YEAR}/H.1301 (Cornil)/{slug}.mp4",)
                )
            )
        },
    )

    with http:
        talks = fosdem.list_year(http, YEAR)

    assert [talk.file for talk in talks] == [
        f"{fosdem.VIDEO_TREE}/{YEAR}/H.1301%20%28Cornil%29/{slug}.mp4"
    ], "the URL kept is the one the tree serves"
    assert fosdem.spread_key(talks[0]) == f"{YEAR}/H.1301 (Cornil)"


def test_a_room_link_is_one_path_segment_and_not_the_parent_or_a_sort_link():
    page = (
        '<a href="../">up</a><a href="?C=N&amp;O=A">sort</a><a href="https://fosdem.org/">home</a>'
        + index(["Janson", "H.1301%20%28Cornil%29", "K.1.105%20%28La%20Fontaine%29"])
    )
    rooms = [href for href in fosdem.links(page) if fosdem.ROOM_LINK.match(href)]
    assert rooms == ["Janson/", "H.1301%20%28Cornil%29/", "K.1.105%20%28La%20Fontaine%29/"]


def test_a_page_the_server_failed_on_is_asked_again():
    answers = iter([httpx.Response(503), httpx.Response(200, text="ok")])
    with httpx.Client(transport=httpx.MockTransport(lambda request: next(answers))) as http:
        assert fosdem.get_text(http, f"{fosdem.VIDEO_TREE}/{YEAR}/") == "ok"


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
    assert [one.file for one in listed] == [f"{base}.mp4"]


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

    assert talk.id == f"{YEAR}-9001"
    assert talk.title == "A talk about Debian"
    assert talk.author == "Ada Lovelace, Grace Hopper"
    assert talk.channel == "H.1301 (Cornil)"
    assert talk.duration_s == 3000 and talk.license == fosdem.LICENSE
    assert talk.url == f"https://fosdem.org/{YEAR}/schedule/event/{slug}/", "the event's page"
    assert talk.file == mp4 and talk.instance == "", "the file is pinned, not asked for"
    assert fosdem.file_url(None, talk) == mp4


def test_the_page_a_schedule_names_is_the_one_a_talk_points_at():
    slug = "a_talk"
    page = f"https://fosdem.org/{YEAR}/schedule/event/fosdem-{YEAR}-9001-a-talk/"
    mp4 = f"{fosdem.VIDEO_TREE}/{YEAR}/h1301/{slug}.mp4"
    http = tree_client(
        {"h1301": [f"{slug}.mp4"]}, {YEAR: schedule(event("9001", slug, videos=(mp4,), page=page))}
    )
    with http:
        assert fosdem.list_year(http, YEAR)[0].url == page


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
            id=f"{year}-{room}{index}",
            url=f"https://fosdem.org/{year}/schedule/event/{index}/",
            file=f"{fosdem.VIDEO_TREE}/{year}/{room}/{index}.mp4",
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

    order = talk_listing.reading_order(listed, seed=1, spread_by=fosdem.spread_key)

    assert len(order) == 12
    assert len({fosdem.spread_key(one) for one in order[:4]}) == 4, "one per year and room first"
    assert all(fosdem.spread_key(one).startswith(("2023/", "2024/")) for one in order)


def test_the_pinned_listing_holds_the_talks_the_tree_served():
    listed_on, talks = read_listing(SOURCE)

    assert listed_on and len(talks) == len({talk.id for talk in talks}) > 0
    assert all(talk.license == fosdem.LICENSE for talk in talks)
    assert all(talk.file.startswith(f"{fosdem.VIDEO_TREE}/") for talk in talks)
    assert all(talk.url.startswith("https://fosdem.org/") for talk in talks), "the event pages"
    assert all(talk.duration_s >= fosdem.MIN_TALK_SECONDS for talk in talks)
    assert all(talk.title and talk.channel for talk in talks)
    assert sum(1 for talk in talks if not talk.author) < 20, "a few talks name no speaker"
    assert all(re.fullmatch(talk.file.split("/")[3] + r"-\d+", talk.id) for talk in talks)
    years = {int(talk.file.split("/")[3]) for talk in talks}
    assert years <= set(fosdem.YEARS) and len(years) > 1, "several years, all listed ones"
    assert len({talk.channel for talk in talks}) > 10, "many rooms"
