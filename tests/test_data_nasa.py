"""NASA's podcasts: the feeds, the episode pages, and a preparation without a server.

The feeds and the pages are written here, shaped like the ones nasa.gov serves,
and the two speech models are fakes. What is checked is which episodes a listing
keeps, which text of a page is the transcript, and that a preparation labels its
clips with that text, after every Whisper request of the batch (D-020).
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import write_tone  # noqa: E402

from audiolab.data import nasa, talks  # noqa: E402
from audiolab.data.aligned_clips import Line  # noqa: E402
from audiolab.data.download import DownloadError  # noqa: E402
from audiolab.data.store import read_manifest  # noqa: E402
from audiolab.data.talk_listing import Talk, write_listing  # noqa: E402

SHOW = "gravity-assist"
PAGES = f"https://www.nasa.gov/podcasts/{SHOW}"


def item(slug: str, *, title: str = "", link: str = "", audio: str = "", duration: str = "60"):
    link = link if link else f"{PAGES}/{slug}/"
    audio = audio if audio else f"https://traffic.megaphone.fm/{slug}.mp3"
    enclosure = f'<enclosure url="{audio}" length="1" type="audio/mpeg"/>' if audio != "-" else ""
    return (
        f"<item><title>{title or slug}</title><link>{link}</link>{enclosure}"
        f"<itunes:duration>{duration}</itunes:duration></item>"
    )


def feed(*items: str, copyright: str = "NASA") -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">'
        f"<channel><title>Gravity Assist</title><copyright>{copyright}</copyright>"
        f"{''.join(items)}</channel></rss>"
    )


def page(*paragraphs: str) -> str:
    """An episode page: the site's menu, the episode's content, the site's footer."""
    body = "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
    return (
        "<html><body><nav><p>Missions</p></nav>"
        f'<div class="hds-podcast-content hds-entry-content">{body}</div>'
        "<footer><p>National Aeronautics and Space Administration</p>"
        "<p>Copyright policy</p></footer></body></html>"
    )


def episode(slug: str, **fields) -> Talk:
    values = {
        "id": slug,
        "url": f"{PAGES}/{slug}/",
        "title": slug,
        "duration_s": 60,
        "license": nasa.LICENSE,
        "author": "NASA, Gravity Assist",
        "channel": SHOW,
        "file": f"https://traffic.megaphone.fm/{slug}.mp3",
    }
    return Talk(**(values | fields))


# The feeds.


def test_a_feed_episode_carries_its_page_its_mp3_and_its_attribution():
    [listed] = nasa.feed_episodes(feed(item("jupiter", title="  Jupiter  with Jared ")), SHOW)

    assert listed == episode("jupiter", title="Jupiter with Jared")
    assert nasa.SOURCE.audio_url(None, listed) == "https://traffic.megaphone.fm/jupiter.mp3"


def test_a_feed_item_without_a_podcast_page_or_an_mp3_is_not_listed():
    listed = nasa.feed_episodes(
        feed(
            item("kept"),
            item("no-page", link=" "),
            item("elsewhere", link="https://appel.nasa.gov/podcast/episode-149/"),
            item("show-page", link=f"{PAGES}/"),
            item("no-audio", audio="-"),
        ),
        SHOW,
    )

    assert [one.id for one in listed] == ["kept"]


def test_a_feed_that_names_another_copyright_holder_is_refused():
    with pytest.raises(DownloadError, match="Caltech"):
        nasa.feed_episodes(feed(item("a"), copyright="NASA/JPL/Caltech"), SHOW)


def test_a_repeat_is_listed_once_under_the_copy_aired_first():
    oldest_first = [
        episode("a-day-in-space", title="A Day in Space"),
        episode("encore-a-day-in-space", title="Encore: A Day In Space"),
        episode("same-mp3", file="https://chrt.fm/track/X/traffic.megaphone.fm/a-day-in-space.mp3"),
        episode("posted-twice", title="Mars Is Hard. Here's Why."),
        episode("posted-twice-again", title="Mars is hard: here’s why"),
        episode("suit-up-for-mars", title="Suit Up for Mars"),
        episode("mars-ep-10-suit-up-for-mars", title="Mars Ep. 10: Suit Up for Mars"),
        episode("another", title="Another episode"),
    ]  # fmt: skip

    kept = nasa.without_repeats(oldest_first)

    assert [one.id for one in kept] == [
        "a-day-in-space",
        "posted-twice",
        "suit-up-for-mars",
        "another",
    ]


def test_the_listing_reads_each_feed_oldest_first(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOLAB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(nasa, "SHOWS", {SHOW: "Gravity Assist"})
    transcript = page("<strong>Transcript</strong>", " ".join(["word"] * 70))
    newest_first = feed(item("encore-moon", title="Encore: Moon"), item("moon", title="Moon"))

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=newest_first if "/feeds/" in request.url.path else transcript
        )

    with httpx.Client(transport=httpx.MockTransport(answer)) as http:
        listed = nasa.list_episodes(http)

    assert [one.id for one in listed] == ["moon"]


# The episode pages.


def test_the_transcript_starts_after_its_heading_and_reads_every_kind_of_label():
    lines = nasa.transcript_lines(
        page(
            "An episode description nobody reads aloud.",
            "<strong>Transcript:</strong>",
            "<strong>Jim Green </strong>",
            "Our solar system is a wondrous place.",
            "<strong>Andres Almeida (Host)</strong>: Welcome to the show.",
            "<strong>[0:09] Narrator:</strong>The space race began.",
            "NICKY FOX: Three, two, one.",
            "Host:",
            "Today: the Moon, and what comes next.",
            "<strong>[ Music ]</strong>",
            "Volcanoes <strong>have</strong> been here for ages.",
        )
    )

    assert lines == [
        Line("Jim Green", ""),
        Line("", "Our solar system is a wondrous place."),
        Line("Andres Almeida", "Welcome to the show."),
        Line("Narrator", "The space race began."),
        Line("NICKY FOX", "Three, two, one."),
        Line("Host", ""),
        Line("", "Today: the Moon, and what comes next."),
        Line("", "[ Music ]"),
        Line("", "Volcanoes have been here for ages."),
    ]


def test_the_text_after_a_list_inside_a_list_item_is_read():
    body = (
        '<div class="hds-podcast-content"><ul><li><strong>Joe Westlake</strong>'
        "<ul><li>(Joe appeared in Episode 1)</li><li>Totality changes you.</li></ul>"
        "and then he thanked the listeners.</li></ul><p>Next.</p></div>"
    )

    assert [block.text for block in nasa.blocks_of(body)] == [
        "Joe Westlake",
        "(Joe appeared in Episode 1)",
        "Totality changes you.",
        "and then he thanked the listeners.",
        "Next.",
    ]


def test_a_plain_name_on_its_own_line_is_a_label_when_it_recurs():
    lines = nasa.transcript_lines(
        page("Miles Hatfield", "Hello there.", "Peter Elstner", "Hi.", "Miles Hatfield", "Ohh")
    )

    assert [line.speaker for line in lines] == ["Miles Hatfield", "", "", "", "Miles Hatfield", ""]


def test_a_page_with_a_copyright_notice_or_without_a_transcript_is_unusable(tmp_path):
    words = " ".join(["word"] * 70)
    pages = {
        "spoken": page("<strong>Transcript</strong>", words),
        "noticed": page(words, "Song courtesy of a label. © 2024 Some Records"),
        "credited": page(words, "Clips from a 1993 interview are used courtesy of an archive."),
        "said": page(words, "Host: So it's not a copyright of the government? Courtesy of Nate."),
        "live": page("Video with captions:", "A short caption."),
        "rerun": page(words, "This episode originally aired May 1, 2020, as Episode 142."),
        "hubble": page(words, "The image Hubble originally released back in 1995."),
    }
    for slug, text in pages.items():
        (tmp_path / f"{slug}.html").write_text(text, encoding="utf-8")

    reasons = {slug: nasa.why_unusable(None, episode(slug), tmp_path) for slug in pages}

    # The footer's "Copyright policy" link is the site's, not the episode's.
    assert reasons == {
        "spoken": "",
        "noticed": "page with a copyright notice",
        "credited": "page with a copyright notice",
        "said": "",
        "live": "page without a transcript",
        "rerun": "page of an episode aired before",
        "hubble": "",
    }


def test_the_listing_reads_each_feed_and_caches_each_page_once(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOLAB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(nasa, "SHOWS", {SHOW: "Gravity Assist", "on-air": "On Air"})
    asked: list[str] = []
    transcript = page("<strong>Transcript</strong>", " ".join(["word"] * 70))

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        if "/feeds/" in request.url.path:
            show = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, text=feed(item(f"{show}-one"), item(f"{show}-live")))
        if request.url.path.endswith("-live/"):
            return httpx.Response(200, text=page("Video with captions:"))
        return httpx.Response(200, text=transcript)

    with httpx.Client(transport=httpx.MockTransport(answer)) as http:
        listed = nasa.list_episodes(http)
        nasa.list_episodes(http)

    assert [one.id for one in listed] == [f"{SHOW}-one", "on-air-one"]
    pages_asked = [path for path in asked if not path.startswith("/feeds/")]
    assert len(pages_asked) == 4, "each page is fetched once"
    assert (tmp_path / "cache" / nasa.NAME / "pages" / f"{SHOW}-one.html").is_file()


# The preparation.


class FakeModels:
    """Whisper and Parakeet, each answering the same text for every request, logged."""

    def __init__(self, whisper_rows: list[dict], parakeet_text: str):
        self.log: list[str] = []
        self.answers = {
            nasa.SEGMENT_MODEL: {"segments": whisper_rows, "text": ""},
            nasa.BASE_MODEL: {"text": parakeet_text},
        }

    def serve(self, monkeypatch) -> None:
        """Stand in for the two model clients ``nasa`` opens."""
        monkeypatch.setattr(nasa, "_client", lambda model, request=None: nullcontext(model))
        monkeypatch.setattr(nasa, "_answer", lambda model, language: partial(self.hear, model))

    def hear(self, model: str, wav: bytes) -> dict:
        self.log.append(model)
        return self.answers[model]


def test_a_preparation_labels_its_clips_with_the_transcript_after_every_whisper_request(
    tmp_path, monkeypatch, capsys
):
    episodes = [episode("one", title="Episode One"), episode("two", title="Episode Two")]
    source = replace(nasa.SOURCE, listing_path=tmp_path / "listing.json")
    write_listing(source, "2026-11-08", episodes)
    monkeypatch.setattr(nasa, "SOURCE", source)
    cache = tmp_path / "cache"
    for one in episodes:
        transcript = page(
            "<strong>Transcript</strong>",
            "<strong>Jim Green:</strong> Welcome to Gravity Assist, Episode 7! (laughs)",
            "<strong>Jared Espley:</strong> Thanks, Jim. Mars has 2 moons.",
        )
        (cache / nasa.NAME / "pages").mkdir(parents=True, exist_ok=True)
        (cache / nasa.NAME / "pages" / f"{one.id}.html").write_text(transcript, encoding="utf-8")
    models = FakeModels(
        [
            {"start": 0.0, "end": 5.0, "text": " welcome to gravity assist episode seven."},
            {"start": 6.5, "end": 11.0, "text": " Thanks Jim, Mars has two moons."},
        ],
        parakeet_text="thanks jim mars has two moons",
    )
    models.serve(monkeypatch)
    monkeypatch.setattr(
        talks, "fetch_audio", lambda http, one, path, src: write_tone(path, seconds=12.0)
    )

    count = nasa.fetch_nasa_en_train(nasa.NAME, cache, tmp_path / "dataset", verbose=True)

    assert count == 4
    assert models.log == [nasa.SEGMENT_MODEL] * 2 + [nasa.BASE_MODEL] * 4
    clips = read_manifest(tmp_path / "dataset")
    assert {clip.text for clip in clips} == {
        "Welcome to Gravity Assist, Episode 7!",
        "Thanks, Jim. Mars has 2 moons.",
    }
    clip = next(clip for clip in clips if clip.source_url == f"{PAGES}/one/" and clip.base_wer == 0)
    assert clip.text == "Thanks, Jim. Mars has 2 moons."
    assert (clip.source_author, clip.source_title) == ("NASA, Gravity Assist", "Episode One")
    assert clip.license == nasa.LICENSE and clip.text_has_casing_punct
    assert "4 of 4 candidate clips (100.0%) labelled" in capsys.readouterr().out
