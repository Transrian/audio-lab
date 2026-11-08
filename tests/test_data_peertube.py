"""The PeerTube API of a talk source, without the network.

The API is replaced by a fake transport, so these tests check which videos a
listing keeps, how an instance is paged, and how the file of a talk is found.
"""

from __future__ import annotations

import email.utils
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from audiolab.data import peertube, talk_sources
from audiolab.data.download import DownloadError
from audiolab.data.peertube import Filter
from audiolab.data.talk_listing import read_listing

INSTANCE = "https://videos.capitoledulibre.org"
FR = Filter(language="fr")


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    """The API client paces its requests; a test has nobody to be polite to."""
    monkeypatch.setattr(peertube.time, "sleep", lambda seconds: None)


def api_row(uuid: str, *, licence: int = 1, language: str | None = "fr", **fields) -> dict:
    """One video as the PeerTube API lists it, with only the fields this source reads."""
    row = {
        "uuid": uuid,
        "url": f"{INSTANCE}/videos/watch/{uuid}",
        "name": f"Talk {uuid}",
        "duration": 600,
        "licence": {"id": licence, "label": "Attribution"},
        "language": {"id": language, "label": "Français"},
        "isLocal": True,
        "channel": {"displayName": "Capitole du Libre"},
        "account": {"displayName": "Capitole du Libre"},
    }
    row.update(fields)
    return row


def instance_client(rows: list[dict], pages: int = 3) -> tuple[httpx.Client, list]:
    """An HTTP client for a fake instance serving ``rows``, and the requests it will get."""
    asked = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.params)
        start = int(request.url.params["start"])
        return httpx.Response(200, json={"total": len(rows), "data": rows[start : start + pages]})

    return httpx.Client(transport=httpx.MockTransport(answer)), asked


def test_only_the_videos_a_source_may_use_are_listed_and_every_page_is_read():
    rows = [
        api_row("c"),
        api_row("a", licence=7),
        api_row("share-alike", licence=2),
        api_row("non-commercial", licence=4),
        api_row("english", language="en"),
        api_row("mirrored", isLocal=False),
        api_row("elsewhere", url="https://other.example/videos/watch/elsewhere"),
        api_row("b"),
    ]
    http, asked = instance_client(rows)
    with http:
        talks = peertube.list_instance(http, INSTANCE, FR)

    assert [talk.id for talk in talks] == ["a", "b", "c"]
    assert [params["start"] for params in asked] == ["0", "3", "6"]
    assert asked[0].get_list("licenceOneOf") == ["1", "7"]
    assert asked[0]["languageOneOf"] == "fr"
    assert talks[0].license == "CC0-1.0" and talks[1].license == "CC-BY"
    assert talks[2].title == "Talk c" and talks[2].author == "Capitole du Libre"
    assert talks[2].instance == INSTANCE and talks[2].url == f"{INSTANCE}/videos/watch/c"


def test_a_filter_says_what_may_be_listed():
    rows = {
        "kept": api_row("kept"),
        "short": api_row("short", duration=60),
        "untagged": api_row("untagged", language=None),
        "in-musique": api_row(
            "in-musique",
            channel={"displayName": "Education Musicale"},
            account={"displayName": "Education Musicale"},
        ),
        "in-danse": api_row(
            "in-danse",
            channel={"displayName": "Atelier Danse"},
            account={"displayName": "Atelier Danse"},
        ),
        "other-licence": api_row("other-licence", licence=3),
    }

    def listed(filter: Filter) -> list[str]:
        return [name for name in rows if peertube.usable(rows[name], INSTANCE, filter)]

    assert listed(Filter(language="fr")) == ["kept", "short", "in-musique", "in-danse"]
    assert listed(Filter(language="fr", min_duration_s=120)) == ["kept", "in-musique", "in-danse"]
    assert listed(Filter(language="en", language_unset_ok=True)) == ["untagged"], "tagged fr"
    assert listed(Filter(language="fr", name_must_hold="capitole")) == ["kept", "short"]
    assert listed(Filter(language="fr", name_must_not_hold=("musicale", "danse"))) == [
        "kept",
        "short",
    ]


def test_a_word_of_the_filter_never_matches_inside_another_word():
    rows = {
        # "es" is the Spanish edition, and the word must not drop an Andreas.
        "spanish": api_row("spanish", language=None, channel={"displayName": "DebConf ES"}),
        "andreas": api_row("andreas", language=None, account={"displayName": "Andreas @ DebConf"}),
    }
    filter = Filter(language="en", language_unset_ok=True, name_must_not_hold=("es",))
    assert [row for row in rows if peertube.usable(rows[row], INSTANCE, filter)] == ["andreas"]


def test_a_request_the_instance_refuses_is_asked_again_after_the_wait_it_names(monkeypatch):
    waits = []
    monkeypatch.setattr(peertube.time, "sleep", waits.append)
    answers = iter(
        [httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200, json={})]
    )
    with httpx.Client(transport=httpx.MockTransport(lambda request: next(answers))) as http:
        assert peertube.get_json(http, f"{INSTANCE}/api/v1/videos") == {}
    assert 7.0 in waits


def test_a_server_error_is_asked_again_and_survives_a_large_instance(monkeypatch):
    waits = []
    monkeypatch.setattr(peertube.time, "sleep", waits.append)
    answers = iter(
        [
            httpx.Response(500),
            httpx.Response(502),
            httpx.Response(200, json={"total": 5827}),
        ]
    )
    with httpx.Client(transport=httpx.MockTransport(lambda request: next(answers))) as http:
        assert peertube.get_json(http, f"{INSTANCE}/api/v1/videos")["total"] == 5827
    assert [wait for wait in waits if wait > 1.0] == [5.0, 10.0], "backing off, then answering"


def test_an_instance_that_keeps_failing_is_reported_and_not_read_on(monkeypatch):
    monkeypatch.setattr(peertube.time, "sleep", lambda seconds: None)
    with httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as http:
        with pytest.raises(DownloadError, match="still refused after"):
            peertube.get_json(http, f"{INSTANCE}/api/v1/videos")


def test_the_smallest_file_of_a_talk_is_asked_for_at_download_time():
    details = {
        "files": [{"size": 900, "fileUrl": f"{INSTANCE}/big.mp4"}],
        "streamingPlaylists": [{"files": [{"size": 100, "fileUrl": f"{INSTANCE}/small.mp4"}]}],
    }

    def answer(request):
        if request.url.path.startswith("/api/v1/videos"):
            return httpx.Response(200, json=details)
        return httpx.Response(206, content=b"x")

    with httpx.Client(transport=httpx.MockTransport(answer)) as http:
        talk = peertube.talk_of(api_row("v1"), INSTANCE)
        assert peertube.smallest_file_url(http, talk) == f"{INSTANCE}/small.mp4"
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
    ) as http:
        with pytest.raises(DownloadError, match="no downloadable file"):
            peertube.smallest_file_url(http, peertube.talk_of(api_row("v2"), INSTANCE))


def test_a_file_the_instance_no_longer_serves_is_not_taken():
    """An instance that moved to streaming still names the files it deleted."""
    details = {
        "files": [{"size": 100, "fileUrl": f"{INSTANCE}/deleted.mp4"}],
        "streamingPlaylists": [{"files": [{"size": 900, "fileUrl": f"{INSTANCE}/kept.mp4"}]}],
    }

    def answer(request):
        if request.url.path.startswith("/api/v1/videos"):
            return httpx.Response(200, json=details)
        if request.url.path == "/deleted.mp4":
            return httpx.Response(404, json={"detail": "Not Found"})
        assert request.headers["Range"] == "bytes=0-0"
        return httpx.Response(206, content=b"x")

    with httpx.Client(transport=httpx.MockTransport(answer)) as http:
        talk = peertube.talk_of(api_row("v1"), INSTANCE)
        assert peertube.smallest_file_url(http, talk) == f"{INSTANCE}/kept.mp4"
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda r: (
                httpx.Response(200, json=details)
                if r.url.path.startswith("/api/v1/videos")
                else httpx.Response(404)
            )
        )
    ) as http:
        talk = peertube.talk_of(api_row("v1"), INSTANCE)
        with pytest.raises(DownloadError, match="no file the instance still has"):
            peertube.smallest_file_url(http, talk)


def test_a_server_that_ignores_the_range_is_not_read_to_the_end():
    """A server that answers the whole video to a one-byte range is taken on its status."""

    class WholeVideo(httpx.SyncByteStream):
        def __iter__(self):
            raise AssertionError("the body of a probed file is never read")

    answer = httpx.MockTransport(lambda request: httpx.Response(200, stream=WholeVideo()))
    with httpx.Client(transport=answer) as http:
        assert peertube.serves(http, f"{INSTANCE}/whole.mp4")


def test_the_committed_listing_of_the_capitole_du_libre_is_readable_and_usable():
    source = talk_sources.SOURCES["peertube-fr-train"]
    listed_on, talks = read_listing(source)
    assert listed_on and len(talks) == len({talk.id for talk in talks}) > 0
    assert all(talk.url.startswith("https://videos.capitoledulibre.org/") for talk in talks)
    assert all(talk.instance == "https://videos.capitoledulibre.org" for talk in talks)


def test_the_pinned_listing_holds_the_videos_of_the_day_it_was_listed():
    """One run of ``list-talks`` wrote these 519 videos on 2026-11-08; a refresh is a change."""
    source = talk_sources.SOURCES["peertube-fr-train"]
    listed_on, talks = read_listing(source)
    assert listed_on == "2026-11-08"
    assert len(talks) == 519
    assert Path(source.listing_path).name == "peertube_listing.json"


def test_retry_after_reads_seconds_or_an_http_date():
    assert peertube.retry_after_s("7", 5.0) == 7.0
    assert peertube.retry_after_s(None, 5.0) == 5.0
    assert peertube.retry_after_s("soon", 5.0) == 5.0
    assert peertube.retry_after_s("Wed, 21 Oct 2015 07:28:00 GMT", 5.0) == 0.0, "a past date"
    later = email.utils.format_datetime(datetime.now(UTC) + timedelta(seconds=60))
    assert 50 < peertube.retry_after_s(later, 5.0) <= 60
