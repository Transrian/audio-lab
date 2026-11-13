"""NASA's podcasts: the feeds, the episode pages, and a preparation without a server.

The feeds and the pages are written here, shaped like the ones nasa.gov serves,
and the two speech models are fakes. What is checked is which episodes a listing
keeps, which text of a page is the transcript, and that a preparation labels its
clips with that text, after every Whisper request of the batch (D-020).
"""

from __future__ import annotations

import hashlib
import io
import sys
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path

import httpx
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from helpers import make_clip, write_tone  # noqa: E402

from audiolab.core.stop import Stopped, request_stop  # noqa: E402
from audiolab.data import nasa, talks  # noqa: E402
from audiolab.data.aligned_clips import Line, labelled_clips  # noqa: E402
from audiolab.data.download import DownloadError  # noqa: E402
from audiolab.data.store import read_manifest  # noqa: E402
from audiolab.data.talk_clips import Segment  # noqa: E402
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


def transcript_of(html: str) -> list[Line]:
    return nasa.transcript_lines(nasa.blocks_of(html))


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


def test_a_feed_duration_is_read_in_seconds_or_as_a_time():
    listed = nasa.feed_episodes(
        feed(item("seconds", duration="1448"), item("time", duration="1:02:03")), SHOW
    )

    assert [one.duration_s for one in listed] == [1448, 3723]


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
    lines = transcript_of(
        page(
            "An episode description nobody reads aloud.",
            "<strong>Transcript:</strong>",
            "<strong>Jim Green </strong>",
            "Our solar system is a wondrous place.",
            "<strong>Andres Almeida (Host)</strong>: Welcome to the show.",
            "<strong>[0:09] Narrator:</strong>The space race began.",
            "NICKY FOX: Three, two, one.",
            "Host:",
            "Willie Lyles: Oh, thank you, Gary.",
            "Steve Altemus 11:32",
            "<strong>[ Music ]</strong>",
            "Volcanoes <strong>have</strong> been here for ages.",
            "Today: the Moon, and what comes next.",
            "and we said: no.",
        )
    )

    assert lines == [
        Line("Jim Green", ""),
        Line("", "Our solar system is a wondrous place."),
        Line("Andres Almeida", "Welcome to the show."),
        Line("Narrator", "The space race began."),
        Line("NICKY FOX", "Three, two, one."),
        Line("Host", ""),
        Line("Willie Lyles", "Oh, thank you, Gary."),
        Line("Steve Altemus", ""),
        Line("", "[ Music ]"),
        Line("", "Volcanoes have been here for ages."),
        # Read as a label, a capitalised phrase before a colon costs a clip at most.
        Line("Today", "the Moon, and what comes next."),
        Line("", "and we said: no."),
    ]


def test_a_plain_label_before_a_colon_keeps_a_clip_from_running_across_two_speakers():
    host = "So thank you very much for coming on the show today, it was a real pleasure."
    lines = transcript_of(
        page(f"<strong>Host:</strong> {host}", "Willie Lyles: Oh, thank you, Gary. Appreciate it.")
    )
    # Read as speech, the two words of the name would be few enough to label the clip.
    clips = [Segment(0.0, 9.0, f"{host} Oh, thank you, Gary. Appreciate it.")]
    tally: Counter[str] = Counter()

    assert labelled_clips(clips, lines, "en", tally) == []
    assert tally["clips left out: a change of speaker or a note inside"] == 1


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
    lines = transcript_of(
        page("Miles Hatfield", "Hello there.", "Peter Elstner", "Hi.", "Miles Hatfield", "Ohh")
    )

    assert [line.speaker for line in lines] == ["Miles Hatfield", "", "", "", "Miles Hatfield", ""]


def test_a_page_with_a_notice_a_music_credit_or_no_transcript_is_unusable():
    words = " ".join(["word"] * 70)  # the episodes last one minute
    pages = {
        "spoken": page("<strong>Transcript</strong>", words),
        "noticed": page(words, "Song courtesy of a label. © 2024 Some Records"),
        "credited": page(words, "Clips from a 1993 interview are used courtesy of an archive."),
        "provided": page(words, "Audio from the 'Life on Mars' talk provided by a society."),
        "said": page(words, "Host: So it's not a copyright of the government? Courtesy of Nate."),
        "live": page("Video with captions:", "A short caption."),
        "rerun": page(words, "This episode originally aired May 1, 2020, as Episode 142."),
        "hubble": page(words, "The image Hubble originally released back in 1995."),
        "twice": page(words, words, words, words),
        "archive": page(words, "[ARCHIVAL Apollo 11 Broadcast: The Eagle has landed.]"),
        "radio": page(words, "<strong>UTTR radio broadcast:</strong> Capsule on the ground."),
        "announced": page(words, "Launch announcer: Three, two, one, liftoff."),
        "speech": page(words, "[John F. Kennedy Archive: We choose to go to the Moon.]"),
        "music": page(words, "[ Music ]", "(Distributed Active Archive Center)"),
        "library": page(words, "Music for the series comes from Universal Production Music."),
        "theme": page(words, "[Song: Ants Instrumental by Taylor Welsh]"),
    }
    reasons = {slug: nasa.why_page_unusable(text, duration_s=60) for slug, text in pages.items()}

    # The footer's "Copyright policy" link is the site's, not the episode's.
    assert reasons == {
        "spoken": "",
        "noticed": "page with a copyright notice",
        "credited": "page with a copyright notice",
        "provided": "page with a copyright notice",
        "said": "",
        "live": "page without a transcript",
        "rerun": "page of an episode aired before",
        "hubble": "",
        "twice": "page with more text than its episode",
        "archive": "episode that plays recordings",
        "radio": "episode that plays recordings",
        "announced": "episode that plays recordings",
        "speech": "episode that plays recordings",
        "music": "",
        "library": "page crediting music of a third party",
        "theme": "page crediting music of a third party",
    }


def test_a_transcript_that_writes_its_notes_as_speech_is_left_out_without_a_fetch(tmp_path):
    reason = nasa.why_unusable(None, episode("a-seismic-journey"), tmp_path)

    assert reason == "transcript with its notes written as speech"


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


def test_a_stopped_listing_reads_no_further_page(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOLAB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(nasa, "PAGE_WORKERS", 1)
    items = [item(f"{SHOW}-{n}") for n in range(5)]
    transcript = page("<strong>Transcript</strong>", " ".join(["word"] * 70))
    pages_asked: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        if "/feeds/" in request.url.path:
            return httpx.Response(200, text=feed(*items))
        pages_asked.append(request.url.path)
        request_stop()  # Ctrl-C while the first page is read
        return httpx.Response(200, text=transcript)

    with (
        httpx.Client(transport=httpx.MockTransport(answer)) as http,
        pytest.raises(Stopped, match="listing stopped"),
    ):
        nasa.list_episodes(http)
    assert len(pages_asked) == 1


# The preparation.


class FakeModels:
    """Whisper and Parakeet, logged: Whisper gives the same segments, Parakeet its texts in turn."""

    def __init__(self, whisper_rows: list[dict], parakeet_texts: list[str]):
        self.log: list[str] = []
        self.whisper = {"segments": whisper_rows, "text": ""}
        self.parakeet_texts = list(parakeet_texts)

    def serve(self, monkeypatch) -> None:
        """Stand in for the two model clients ``nasa`` opens."""
        monkeypatch.setattr(nasa, "client", lambda model, request=None: nullcontext(model))
        monkeypatch.setattr(nasa, "served", lambda model, language: self.served(model))

    def served(self, model: str) -> talks.Served:
        return talks.Served(
            partial(self.hear, model), warm_up=lambda: self.log.append(f"warm-up {model}")
        )

    def hear(self, model: str, wav: bytes) -> dict:
        self.log.append(model)
        if model == nasa.SEGMENT_MODEL:
            return self.whisper
        return {"text": self.parakeet_texts.pop(0)}


def test_a_preparation_labels_its_clips_with_the_transcript_after_every_whisper_request(
    tmp_path, monkeypatch, capsys
):
    episodes = [
        episode("one", title="Episode One"),
        episode("two", title="Episode Two"),
        episode("noticed", title="Episode Three"),
    ]
    source = replace(nasa.SOURCE, listing_path=tmp_path / "listing.json")
    write_listing(source, "2026-11-08", episodes)
    monkeypatch.setattr(nasa, "SOURCE", source)
    monkeypatch.setattr(nasa, "MIN_WORDS_PER_MINUTE", 10)  # a one-minute episode of 13 words
    transcript = page(
        "<strong>Transcript</strong>",
        "<strong>Jim Green:</strong> Welcome to Gravity Assist, Episode 7! (laughs)",
        "<strong>Jared Espley:</strong> Thanks, Jim. Mars has 2 moons.",
    )
    # The page of the third episode took a copyright notice after the listing.
    noticed = page("Transcript", "Jim Green: Hello. © 2026 Some Studio")
    pages = tmp_path / "cache" / nasa.NAME / "pages"
    pages.mkdir(parents=True)
    for one in episodes:
        (pages / f"{one.id}.html").write_text(transcript, encoding="utf-8")

    def answer(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=noticed if "/noticed/" in request.url.path else transcript)

    monkeypatch.setattr(
        nasa, "http_client", lambda: httpx.Client(transport=httpx.MockTransport(answer))
    )
    models = FakeModels(
        [
            {"start": 0.0, "end": 5.0, "text": " welcome to gravity assist episode seven."},
            {"start": 6.5, "end": 11.0, "text": " Thanks Jim, Mars has two moons."},
        ],
        [
            "welcome to gravity assist episode seven",
            "thanks jim mars has two moons",
            "to gravity assist episode seven",  # the cut took the first word
            "thanks jim mars has to moons",
        ],
    )
    models.serve(monkeypatch)
    monkeypatch.setattr(
        talks, "fetch_audio", lambda http, one, path, src: write_tone(path, seconds=12.0)
    )

    count = nasa.fetch_nasa_en_train(
        nasa.NAME, tmp_path / "cache", tmp_path / "dataset", verbose=True
    )

    assert count == 3
    # One Parakeet request per clip: the base model's WER reuses the edge check's answer.
    whisper, parakeet = nasa.SEGMENT_MODEL, nasa.BASE_MODEL
    each_pass_warmed = [f"warm-up {whisper}", whisper, whisper, f"warm-up {parakeet}"]
    assert models.log == each_pass_warmed + [parakeet] * 4
    clips = read_manifest(tmp_path / "dataset")
    assert sorted((clip.source_url, clip.text) for clip in clips) == [
        (f"{PAGES}/one/", "Thanks, Jim. Mars has 2 moons."),
        (f"{PAGES}/one/", "Welcome to Gravity Assist, Episode 7!"),
        (f"{PAGES}/two/", "Thanks, Jim. Mars has 2 moons."),
    ]
    clip = next(clip for clip in clips if clip.source_url == f"{PAGES}/one/" and clip.base_wer == 0)
    assert (clip.source_author, clip.source_title) == ("NASA, Gravity Assist", "Episode One")
    assert clip.license == nasa.LICENSE and clip.text_has_casing_punct
    assert [clip.base_wer > 0 for clip in clips if clip.source_url == f"{PAGES}/two/"] == [True]
    out = capsys.readouterr().out
    assert "4 of 4 candidate clips (100.0%) labelled" in out
    assert f"1 {nasa.EDGES_UNHEARD}" in out
    assert "1 episodes left out at preparation: page with a copyright notice" in out


def test_the_room_under_the_cap_is_shared_over_the_episodes_in_turn():
    assert nasa.shares([5, 1, 3], room=6, cap=10) == [3, 1, 2]
    assert nasa.shares([5, 1, 3], room=100, cap=2) == [2, 1, 2]
    assert nasa.shares([5, 1, 3], room=2, cap=10) == [1, 1, 0]
    assert nasa.shares([], room=5, cap=2) == []


# The reservation of the evaluation set.


def test_the_reservation_reserves_60_episodes_by_a_constant_rule():
    # Four shows of a hundred episodes; two of them are in the training sets.
    episodes = [episode(f"{show}-{i:03d}", channel=show) for show in "abcd" for i in range(100)]
    used = frozenset({episodes[3].id, episodes[150].id})
    reserved = nasa.reserved_test_episodes(episodes, used)

    assert len(reserved) == nasa.TEST_EPISODES
    per_show = Counter(e.channel for e in reserved)
    assert all(count <= nasa.TEST_EPISODES_PER_SHOW for count in per_show.values())
    assert all(e.id not in used for e in reserved)
    # The rule is a constant: the same episodes, whatever the listing's order.
    again = nasa.reserved_test_episodes(list(reversed(episodes)), used)
    assert [e.id for e in again] == [e.id for e in reserved]
    # And it is the first 60 of the constant hash, at most 15 per show.
    free = [e for e in episodes if e.id not in used]
    expected, per = [], Counter()
    for one in sorted(free, key=lambda e: hashlib.sha256(f"nasa-test-{e.id}".encode()).hexdigest()):
        if per[one.channel] >= 15:
            continue
        expected.append(one.id)
        per[one.channel] += 1
        if len(expected) == 60:
            break
    assert [e.id for e in reserved] == expected


def test_the_reservation_never_reads_an_excluded_show():
    episodes = [episode(f"cu-{i:03d}", channel="curious-universe") for i in range(100)]
    assert nasa.reserved_test_episodes(episodes, frozenset()) == []


# The exclusion of the evaluation set from the training set.


def test_a_preparation_of_the_training_set_skips_the_episodes_of_the_evaluation_set(
    tmp_path, monkeypatch, capsys
):
    # The evaluation set is prepared and holds the episode "one".
    test_dir = tmp_path / nasa.NAME_TEST
    test_dir.mkdir()
    (test_dir / "manifest.jsonl").write_text(
        make_clip(f"{nasa.NAME_TEST}:one", language="en", group=f"{nasa.NAME_TEST}:one").manifest_line()
        + "\n",
        encoding="utf-8",
    )
    episodes = [episode("one"), episode("two")]
    source = replace(nasa.SOURCE, listing_path=tmp_path / "listing.json")
    write_listing(source, "2026-11-08", episodes)
    monkeypatch.setattr(nasa, "SOURCE", source)
    monkeypatch.setattr(nasa, "MIN_WORDS_PER_MINUTE", 10)
    transcript = page(
        "<strong>Transcript</strong>",
        "<strong>Jim Green:</strong> Welcome to Gravity Assist, Episode 7! (laughs)",
        "<strong>Jared Espley:</strong> Thanks, Jim. Mars has 2 moons.",
    )
    asked = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        return httpx.Response(200, text=transcript)

    monkeypatch.setattr(nasa, "http_client", lambda: httpx.Client(transport=httpx.MockTransport(answer)))
    models = FakeModels(
        [
            {"start": 0.0, "end": 5.0, "text": " welcome to gravity assist episode seven."},
            {"start": 6.5, "end": 11.0, "text": " Thanks Jim, Mars has two moons."},
        ],
        ["welcome to gravity assist episode seven", "thanks jim mars has two moons"],
    )
    models.serve(monkeypatch)
    monkeypatch.setattr(talks, "fetch_audio", lambda http, one, path, src: write_tone(path, seconds=12.0))

    count = nasa.fetch_nasa_en_train(nasa.NAME, tmp_path / "cache", tmp_path / "dataset", verbose=True)

    assert count == 2
    clips = read_manifest(tmp_path / "dataset")
    assert {clip.source_url for clip in clips} == {f"{PAGES}/two/"}
    # The reserved episode is never downloaded, and its page is never read.
    assert all("/one/" not in path for path in asked)
    assert "1 episodes reserved for nasa-en-test are left out" in capsys.readouterr().out


# The preparation of the evaluation set.


class FakeTestModels(FakeModels):
    """Whisper and Parakeet, logged: Whisper gives the chunks' segments, then its
    per-clip texts in turn, and Parakeet its texts in turn."""

    def __init__(self, whisper_rows: list[dict], whisper_clip_texts: list[str], parakeet_texts: list[str]):
        super().__init__(whisper_rows, parakeet_texts)
        self.whisper_clip_texts = list(whisper_clip_texts)

    def hear(self, model: str, wav: bytes) -> dict:
        self.log.append(model)
        if model == nasa.SEGMENT_MODEL:
            # The chunk of a 60 s episode is 60 s; a clip is at most 20 s.
            if sf.info(io.BytesIO(wav)).duration >= 30:
                return self.whisper
            return {"text": self.whisper_clip_texts.pop(0)}
        return {"text": self.parakeet_texts.pop(0)}


def test_a_preparation_of_the_evaluation_set_checks_the_edges_with_whisper(
    tmp_path, monkeypatch, capsys
):
    episodes = [episode("one"), episode("two"), episode("curious", channel="curious-universe")]
    source = replace(nasa.SOURCE, listing_path=tmp_path / "listing.json")
    write_listing(source, "2026-11-08", episodes)
    monkeypatch.setattr(nasa, "SOURCE", source)
    monkeypatch.setattr(nasa, "MIN_WORDS_PER_MINUTE", 10)
    transcript = page(
        "<strong>Transcript</strong>",
        "<strong>Jim Green:</strong> Welcome to Gravity Assist, Episode 7! (laughs)",
        "<strong>Jared Espley:</strong> Thanks, Jim. Mars has 2 moons.",
    )
    monkeypatch.setattr(
        nasa,
        "http_client",
        lambda: httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, text=transcript))
        ),
    )
    models = FakeTestModels(
        [
            {"start": 0.0, "end": 5.0, "text": " welcome to gravity assist episode seven."},
            {"start": 6.5, "end": 11.0, "text": " Thanks Jim, Mars has two moons."},
        ],
        [
            "welcome to gravity assist episode seven",
            "to gravity assist episode seven",  # the cut took the first word
            "welcome to gravity assist episode seven",
            "to gravity assist episode seven",  # the cut took the first word
        ],
        ["welcome to gravity assist episode seven", "welcome to gravity assist episode seven"],
    )
    models.serve(monkeypatch)
    monkeypatch.setattr(talks, "fetch_audio", lambda http, one, path, src: write_tone(path, seconds=60.0))

    count = nasa.fetch_nasa_en_test(nasa.NAME_TEST, tmp_path / "cache", tmp_path / "dataset", verbose=True)

    assert count == 2
    whisper, parakeet = nasa.SEGMENT_MODEL, nasa.BASE_MODEL
    # Every Whisper request of the batch — the chunks and the edge check — is sent
    # before the first Parakeet one (D-020).
    assert models.log == [
        f"warm-up {whisper}",
        whisper,
        whisper,
        whisper,
        whisper,
        whisper,
        whisper,
        f"warm-up {parakeet}",
        parakeet,
        parakeet,
    ]
    clips = read_manifest(tmp_path / "dataset")
    # Each episode keeps its first clip, whose edge words Whisper confirms.
    assert [clip.text for clip in clips] == [
        "Welcome to Gravity Assist, Episode 7!",
        "Welcome to Gravity Assist, Episode 7!",
    ]
    assert {clip.group for clip in clips} == {f"{nasa.NAME_TEST}:one", f"{nasa.NAME_TEST}:two"}
    assert all(len([c for c in clips if c.group == clip.group]) == 1 for clip in clips)
    assert all(clip.group_kind == "recording" for clip in clips)
    assert all(clip.base_wer == 0 for clip in clips)
    out = capsys.readouterr().out
    assert "3 episodes; 2 reserved for nasa-en-test" in out
    assert "4 of 4 candidate clips (100.0%) labelled" in out
    assert f"2 {nasa.EDGES_UNHEARD_TEST}" in out
