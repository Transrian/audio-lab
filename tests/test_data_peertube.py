"""The PeerTube listing of ``peertube-fr-train``, without the network or ffmpeg.

The API and ffmpeg are replaced by fakes, so these tests check which videos the
listing keeps, the order they are read in, and how a talk becomes audio.
"""

from __future__ import annotations

import email.utils
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import write_tone  # noqa: E402

from audiolab.data import peertube  # noqa: E402
from audiolab.data.download import DownloadError  # noqa: E402

INSTANCE = peertube.INSTANCE


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    """The API client paces its requests; a test has nobody to be polite to."""
    monkeypatch.setattr(peertube.time, "sleep", lambda seconds: None)


def api_row(uuid: str, *, licence: int = 1, language: str = "fr", **fields) -> dict:
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


def video(uuid: str = "v1", licence_id: int = 1) -> peertube.Video:
    return peertube.Video(
        uuid=uuid,
        url=f"{INSTANCE}/videos/watch/{uuid}",
        title=f"Talk {uuid}",
        duration_s=600,
        licence_id=licence_id,
        channel="Capitole du Libre",
        account="Capitole du Libre",
    )


def test_list_videos_reads_every_page_and_keeps_only_usable_videos():
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
    asked = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.params)
        start = int(request.url.params["start"])
        return httpx.Response(200, json={"total": len(rows), "data": rows[start : start + 3]})

    with httpx.Client(transport=httpx.MockTransport(answer)) as http:
        videos = peertube.list_videos(http)

    assert [v.uuid for v in videos] == ["a", "b", "c"]
    assert [params["start"] for params in asked] == ["0", "3", "6"]
    assert asked[0].get_list("licenceOneOf") == ["1", "7"]
    assert asked[0]["languageOneOf"] == "fr"
    assert videos[0].license == "CC0-1.0" and videos[1].license == "CC-BY"
    assert videos[2].title == "Talk c" and videos[2].account == "Capitole du Libre"


def test_a_request_the_instance_refuses_is_asked_again_after_the_wait_it_names(monkeypatch):
    waits = []
    monkeypatch.setattr(peertube.time, "sleep", waits.append)
    answers = iter(
        [httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200, json={})]
    )
    with httpx.Client(transport=httpx.MockTransport(lambda request: next(answers))) as http:
        assert peertube.get_json(http, f"{INSTANCE}/api/v1/videos") == {}
    assert 7.0 in waits


def test_the_listing_is_written_and_read_back_with_its_date(tmp_path):
    path = tmp_path / "listing.json"
    peertube.write_listing([video("a"), video("b", licence_id=7)], "2026-11-08", path)

    listed_on, videos = peertube.read_listing(path)

    assert listed_on == "2026-11-08"
    assert videos == [video("a"), video("b", licence_id=7)]
    assert json.loads(path.read_text())["filter"]["licence_ids"] == [1, 7]


def test_a_listing_holding_another_licence_is_refused(tmp_path):
    path = tmp_path / "listing.json"
    peertube.write_listing([video("a", licence_id=2)], "2026-11-08", path)
    with pytest.raises(DownloadError, match="another licence"):
        peertube.read_listing(path)


def test_the_committed_listing_is_readable_and_usable():
    listed_on, videos = peertube.read_listing()
    assert listed_on and len(videos) == len({v.uuid for v in videos}) > 0
    assert all(v.url.startswith(INSTANCE + "/") for v in videos)


def test_the_reading_order_depends_on_the_seed_and_not_on_the_listing_order():
    videos = [video(f"v{index}") for index in range(20)]
    order = peertube.reading_order(videos, seed=1)
    assert order == peertube.reading_order(list(reversed(videos)), seed=1)
    assert order != peertube.reading_order(videos, seed=2)
    assert sorted(v.uuid for v in order) == sorted(v.uuid for v in videos)


def test_fetch_audio_keeps_the_audio_and_deletes_the_video(tmp_path, monkeypatch):
    source = tmp_path / "remote.mp4"
    source.write_bytes(b"pixels and sound")
    monkeypatch.setattr(peertube, "smallest_file_url", lambda http, video: str(source))
    commands = []

    def fake_ffmpeg(command, **options):
        commands.append(command)
        assert Path(command[command.index("-i") + 1]).is_file()
        write_tone(Path(command[-1]), seconds=2.0)
        return type("Finished", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(peertube.subprocess, "run", fake_ffmpeg)
    audio = peertube.fetch_audio(None, video(), tmp_path / "cache" / "v1.wav")

    assert audio.is_file() and audio.name == "v1.wav"
    assert sorted(p.name for p in audio.parent.iterdir()) == ["v1.wav"]
    assert ["-ac", "1", "-ar", "16000"] == commands[0][8:12]
    peertube.fetch_audio(None, video(), audio)
    assert len(commands) == 1, "an extracted talk is not downloaded again"


def test_fetch_audio_reports_ffmpeg_failing_and_leaves_nothing_behind(tmp_path, monkeypatch):
    source = tmp_path / "remote.mp4"
    source.write_bytes(b"pixels and sound")
    monkeypatch.setattr(peertube, "smallest_file_url", lambda http, video: str(source))
    failed = type("Finished", (), {"returncode": 1, "stderr": "no audio stream"})()
    monkeypatch.setattr(peertube.subprocess, "run", lambda command, **options: failed)

    with pytest.raises(DownloadError, match="no audio stream"):
        peertube.fetch_audio(None, video(), tmp_path / "cache" / "v1.wav")
    assert list((tmp_path / "cache").iterdir()) == []


def test_retry_after_reads_seconds_or_an_http_date():
    assert peertube.retry_after_s("7", 5.0) == 7.0
    assert peertube.retry_after_s(None, 5.0) == 5.0
    assert peertube.retry_after_s("soon", 5.0) == 5.0
    assert peertube.retry_after_s("Wed, 21 Oct 2015 07:28:00 GMT", 5.0) == 0.0, "a past date"
    later = email.utils.format_datetime(datetime.now(UTC) + timedelta(seconds=60))
    assert 50 < peertube.retry_after_s(later, 5.0) <= 60
