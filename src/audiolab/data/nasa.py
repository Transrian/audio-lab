"""NASA's podcasts: episodes labelled by the transcripts NASA publishes with them.

NASA publishes each episode twice: as an MP3 in the show's RSS feed, and as a
page on nasa.gov whose body is a near-verbatim transcript, cased and punctuated,
with the speakers named. The labels of ``nasa-en-train`` are that human text,
not a model's, and :mod:`audiolab.data.aligned_clips` is what places it in time.

- ``audiolab data list-talks nasa-en-train`` reads the feeds of :data:`SHOWS`,
  drops the repeats (:func:`without_repeats`), fetches every episode page once
  into the cache, and pins the episodes whose page holds a transcript and no
  copyright notice. The listing holds metadata only; the raw pages stay in
  ``$AUDIOLAB_CACHE/nasa-en-train/pages``.
- ``audiolab data fetch nasa-en-train`` (:func:`fetch_nasa_en_train`) reads the
  listing in a seeded order spread over the shows and runs, batch by batch, the
  Whisper pass of the other talk sources (:func:`audiolab.data.talks.heard_talk`:
  chunks, raw answers cached), the alignment of each episode's transcript to its
  candidate clips (no model), then the Parakeet pass on the kept clips, which
  records ``base_wer`` (:func:`audiolab.data.talks.offer_talk`). Every Whisper
  request of a batch is sent before the first Parakeet one (D-020).

The licence is public domain: a work of the US government is not protected by
copyright in the United States, and D-019 allows public domain. The evidence and
the rules that follow from it are on the source's card in `docs/stt/datasets.md`.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from html.parser import HTMLParser
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse

import httpx

from audiolab.core.paths import cache_dir
from audiolab.data.aligned_clips import NOTE, Line, labelled_clips
from audiolab.data.build import ManifestBuilder
from audiolab.data.download import DownloadError
from audiolab.data.talk_clips import Answers, Transcribe, candidates
from audiolab.data.talk_listing import Talk, TalkSource, read_listing, reading_order
from audiolab.data.talks import (
    BASE_MODEL,
    BASE_WRONG,
    KEPT,
    SEED,
    SEGMENT_MODEL,
    _answer,
    _audio_file,
    _client,
    batch_size,
    clips_file,
    heard_talk,
    http_client,
    offer_talk,
    prefetched,
)
from audiolab.data.train import Take

NAME = "nasa-en-train"
LANGUAGE = "en"
LICENSE = "public domain (US government work)"
LISTING_PATH = Path(__file__).parent / "nasa_listing.json"
FEED = "https://www.nasa.gov/feeds/podcasts/{show}"
# The shows this source reads, by the slug of their feed, with their names.
# On a Mission is left out: it is made by JPL, which Caltech runs for NASA, and
# its feed names "NASA/JPL/Caltech" as the copyright holder, so it is not a
# work of the US government.
SHOWS = {
    "houston-we-have-a-podcast": "Houston We Have a Podcast",
    "small-steps-giant-leaps": "Small Steps, Giant Leaps",
    "nasa-in-silicon-valley": "NASA in Silicon Valley",
    "curious-universe": "NASA's Curious Universe",
    "gravity-assist": "Gravity Assist",
}
# What the feed of a show may name as its copyright holder: NASA, or NASA's own
# knowledge programme (APPEL). A feed that names anyone else is refused, so a
# change of hands is seen at the next listing rather than trained on.
FEED_COPYRIGHTS = frozenset({"NASA", "NASA/APPEL Knowledge Services"})
ITUNES = "{http://www.itunes.com/dtds/podcast-1.0.dtd}"
# Where an episode page is: www.nasa.gov/podcasts/<section>/<slug>/.
PAGE_HOST, PAGE_SECTION = "www.nasa.gov", "podcasts"
# A page has a transcript when its text holds at least this many words for each
# minute of audio; people say about 150. A "Live" episode's page holds a
# caption and a video link, an older page only its description.
MIN_WORDS_PER_MINUTE = 60
# NASA's media guidelines: third-party material NASA uses is "identified as
# copyright protected with the name of the copyright holder". An episode whose
# page carries such a notice is left out whole, since the notice does not say
# which minutes of the episode it covers. A notice, not a word: guests say
# "copyright" and "licensed" in passing, and a credit names its source as
# "used courtesy of" or "used with permission". The credit of the shows' music
# library ("Music … comes from Universal Production Music") is not a notice: the
# music is not what a clip's label transcribes.
NOTICE = re.compile(
    r"©|\ball rights reserved\b|\bcopyright(?:ed)? (?:by |\d{4})"
    r"|\bused (?:courtesy of|with permission)\b",
    re.IGNORECASE,
)
# What a title adds when an episode is aired again: "Encore: X", "Mars Ep. 3: X".
RERUN_TITLE = re.compile(r"\bencore\b|^.*?\b(?:ep|episode)\.? ?\d+ ?:", re.IGNORECASE)
# What the page of an episode aired again says of it: "This episode originally
# aired May 1, 2020, as Episode 142", "the rebroadcast of a conversation".
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
RERUN_PAGE = re.compile(
    rf"\boriginally (?:aired|released)(?: on)? (?:as )?(?:episode\b|(?:{MONTHS}) \d)"
    r"|\brebroadcast of\b",
    re.IGNORECASE,
)
# Pages come from one public website: three at a time, as the audio downloads.
PAGE_WORKERS = 3
# The counts of the labelling report, beside those of the Parakeet pass.
CANDIDATES = "candidate clips"
LABELLED = "clips labelled from the transcript"


# The feeds.


def feed_episodes(feed: str, show: str) -> list[Talk]:
    """The episodes of one show's RSS feed that have a page and an MP3.

    A few items link to no page, or to a page outside the podcast section of
    nasa.gov, where no transcript is found: they are left out.
    """
    channel = ET.fromstring(feed).find("channel")
    holder = (channel.findtext("copyright") or "").strip()
    if holder not in FEED_COPYRIGHTS:
        raise DownloadError(f"the feed of {show} names {holder!r} as its copyright holder")
    episodes = []
    for item in channel.iter("item"):
        page = (item.findtext("link") or "").strip()
        enclosure = item.find("enclosure")
        audio = enclosure.get("url", "").strip() if enclosure is not None else ""
        if not episode_slug(page) or not audio:
            continue
        episodes.append(
            Talk(
                id=episode_slug(page),
                url=page,
                title=" ".join((item.findtext("title") or "").split()),
                duration_s=int(item.findtext(f"{ITUNES}duration") or 0),
                license=LICENSE,
                author=f"NASA, {SHOWS[show]}",
                channel=show,
                file=audio,
            )
        )
    return episodes


def episode_slug(url: str) -> str:
    """The last part of an episode page's path, or nothing for a URL that is not one."""
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc != PAGE_HOST or len(parts) != 3 or parts[0] != PAGE_SECTION:
        return ""
    return parts[2]


def without_repeats(episodes: list[Talk]) -> list[Talk]:
    """Each episode once: the first copy of it in the order given.

    A show airs an episode again under a new page and a new MP3, as an "Encore"
    or in a series ("Mars Ep. 3: X"), and once in a while posts it twice. Two
    episodes are the same when they share a page, an MP3 file or a title once
    :data:`RERUN_TITLE` and punctuation are set aside. The listing passes the
    episodes oldest first, so the copy kept is the one aired first.
    """
    seen: set[tuple[str, str]] = set()
    kept = []
    for episode in episodes:
        keys = {
            ("page", episode.url),
            ("audio", urlparse(episode.file).path.rsplit("/", 1)[-1]),
            ("title", title_key(episode.title)),
        }
        if not keys & seen:
            kept.append(episode)
        seen |= keys
    return kept


def title_key(title: str) -> str:
    """What two copies of one episode's title share: its words, without a rerun's prefix."""
    return " ".join(re.findall(r"[a-z0-9]+", RERUN_TITLE.sub(" ", title.lower())))


# The episode pages.


@dataclass(frozen=True)
class Block:
    """One paragraph of a page: its text, and the bold text it starts with."""

    text: str
    bold_start: str


class _Blocks(HTMLParser):
    """The paragraphs, list items and headings of a stretch of HTML, in order."""

    BLOCKS = frozenset({"p", "li", "h1", "h2", "h3", "h4", "h5", "h6"})
    BOLD = frozenset({"strong", "b"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.parts: list[str] | None = None
        self.bold_parts: list[str] = []
        self.starting = False  # still in the bold text the paragraph opens with
        self.bold = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.BLOCKS:
            self.handle_endtag(tag)  # a paragraph left open ends where the next starts
            self.parts, self.bold_parts, self.starting = [], [], True
        elif tag in self.BOLD:
            self.bold += 1
        elif tag == "br":
            self.handle_data(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BOLD:
            self.bold = max(0, self.bold - 1)
        elif tag in self.BLOCKS and self.parts is not None:
            self.blocks.append(Block(squash("".join(self.parts)), squash("".join(self.bold_parts))))
            self.parts = None

    def handle_data(self, data: str) -> None:
        if self.parts is None:
            return
        self.parts.append(data)
        if self.starting and self.bold:
            self.bold_parts.append(data)
        elif data.strip():
            self.starting = False


def squash(text: str) -> str:
    return " ".join(text.split())


def episode_body(page: str) -> str:
    """The HTML of the episode itself: from its content block to the site's footer."""
    start = page.find('<div class="hds-podcast-content')
    if start < 0:
        return ""
    end = page.find("<footer", start)
    return page[start : end if end > 0 else len(page)]


def blocks_of(page: str) -> list[Block]:
    """The paragraphs of an episode's content that hold any text."""
    parser = _Blocks()
    parser.feed(episode_body(page))
    return [block for block in parser.blocks if block.text]


def transcript_lines(page: str) -> list[Line]:
    """The transcript of an episode page, as lines with their speakers.

    The transcript starts after a "Transcript" heading when the page has one, and
    with the episode's content otherwise; a description that comes before it on
    such a page is text nobody says, which the alignment finds no audio for.
    """
    blocks = blocks_of(page)
    headings = [n for n, block in enumerate(blocks) if is_transcript_heading(block.text)]
    start = headings[0] + 1 if headings else 0
    lines = [line_of(block) for block in blocks[start:]]
    return with_name_lines(lines)


def is_transcript_heading(text: str) -> bool:
    return text.rstrip(":").strip().lower() == "transcript"


def line_of(block: Block) -> Line:
    """One paragraph as a line, its speaker label taken off the speech.

    The shows write a label three ways: bold and alone on its line ("Gary
    Jordan"), bold or in capitals before a colon ("Host: Welcome…", "NICKY FOX:
    Three, two, one"), or alone with a colon ("Host:"). A note in a label, such
    as a timestamp or "(Host)", is not part of the name. A colon in plain speech
    ("Today: the Moon") is not a label.
    """
    text, bold = block.text, NOTE.sub("", block.bold_start).strip()
    # The first colon outside a note: "[0:09] Narrator: …" is labelled Narrator.
    masked = NOTE.sub(lambda note: "#" * len(note.group()), text)
    before, colon, _ = masked.partition(":")
    name = squash(NOTE.sub("", text[: len(before)]))
    speech = text[len(before) + 1 :].strip()
    if colon and is_name(name) and (bold.startswith(name) or name.isupper() or not speech):
        return Line(name, speech)
    if bold and text == block.bold_start and is_name(bold):
        return Line(bold, "")
    return Line("", text)


def is_name(text: str) -> bool:
    """Whether a label can be a speaker's name: a few words, not a sentence."""
    return 0 < len(text.split()) <= 6 and not text.endswith((".", "?", "!", ",", ";"))


def with_name_lines(lines: list[Line]) -> list[Line]:
    """The lines, with plain names on a line of their own read as labels.

    Some pages write the speaker in plain text above what they say ("Miles
    Hatfield"). Such a line is a label when it looks like a name (two to four
    capitalised words, no punctuation) and recurs: a line of speech that short
    rarely comes back word for word.
    """
    counts = Counter(line.text for line in lines if not line.speaker)
    names = {
        text
        for text, count in counts.items()
        if count >= 2 and re.fullmatch(r"[A-Z][\w'’.-]*(?: [A-Z][\w'’.-]*){1,3}", text)
    }
    return [Line(line.text, "") if line.text in names else line for line in lines]


def words_in(lines: list[Line]) -> int:
    """How many words of speech a transcript holds."""
    return sum(len(NOTE.sub(" ", line.text).split()) for line in lines)


def notice_in(page: str) -> str:
    """The copyright notice an episode's content carries, with its context, or nothing."""
    for block in blocks_of(page):
        if NOTICE.search(block.text):
            return block.text[:200]
    return ""


def page_text(http: httpx.Client, episode: Talk, pages: Path) -> str:
    """An episode's page, fetched once and kept raw in ``pages``.

    The raw page is kept rather than what was read from it, so a change to the
    reading never reuses an old reading.
    """
    path = pages / f"{episode.id}.html"
    if not path.is_file():
        response = http.get(episode.url)
        if response.status_code != 200:
            raise DownloadError(f"{episode.url}: HTTP {response.status_code}")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = path.with_suffix(".partial")
        partial_path.write_text(response.text, encoding="utf-8")
        partial_path.rename(path)
    return path.read_text(encoding="utf-8")


def why_unusable(http: httpx.Client, episode: Talk, pages: Path) -> str:
    """Why an episode cannot be trained on, or an empty string when it can."""
    try:
        page = page_text(http, episode, pages)
    except (DownloadError, httpx.HTTPError):
        return "page not fetched"
    if notice_in(page):
        return "page with a copyright notice"
    if RERUN_PAGE.search(" ".join(block.text for block in blocks_of(page))):
        return "page of an episode aired before"
    minutes = max(episode.duration_s / 60, 1.0)
    if words_in(transcript_lines(page)) < MIN_WORDS_PER_MINUTE * minutes:
        return "page without a transcript"
    return ""


# The listing.


def list_episodes(http: httpx.Client) -> list[Talk]:
    """The episodes ``audiolab data list-talks`` pins: every show, repeats out, pages read."""
    episodes: list[Talk] = []
    for show in SHOWS:
        response = http.get(FEED.format(show=show))
        response.raise_for_status()
        # A feed lists the newest episode first; a repeat is newer than its original.
        episodes += reversed(feed_episodes(response.text, show))
    unique = without_repeats(episodes)
    slugs = Counter(episode.id for episode in unique)
    if max(slugs.values(), default=0) > 1:
        raise DownloadError(f"two episode pages share a slug: {slugs.most_common(1)}")
    pages = cache_dir() / NAME / "pages"
    with ThreadPoolExecutor(max_workers=PAGE_WORKERS) as fetches:
        reasons = list(fetches.map(partial(why_unusable, http, pages=pages), unique))
    left_out = Counter(reason for reason in reasons if reason)
    print(
        f"  {len(episodes)} episodes in the feeds, {len(episodes) - len(unique)} repeats; "
        f"left out: {dict(left_out) or 'none'}"
    )
    return [episode for episode, reason in zip(unique, reasons, strict=True) if not reason]


SOURCE = TalkSource(
    name=NAME,
    language=LANGUAGE,
    allowed_licenses=frozenset({LICENSE}),
    listing_path=LISTING_PATH,
    listing_filter={
        "feeds": [FEED.format(show=show) for show in SHOWS],
        "feed_copyright_holders": sorted(FEED_COPYRIGHTS),
        "page": f"https://{PAGE_HOST}/{PAGE_SECTION}/<section>/<slug>/",
        "repeats": "the first aired of the episodes sharing a page, an MP3 file or a title "
        "without 'Encore' or a series' 'Ep. N:'; no page saying it aired before",
        "min_transcript_words_per_minute": MIN_WORDS_PER_MINUTE,
        "copyright_notice": "an episode whose page carries one is left out",
    },
    lister=list_episodes,
    audio_url=lambda http, episode: episode.file,
    spread_by=lambda episode: episode.channel,
)


# The preparation.


def fetch_nasa_en_train(name: str, cache: Path, dataset_dir: Path, *, verbose: bool) -> int:
    """Prepare ``nasa-en-train`` up to the clip cap, returning the clip count."""
    listed_on, episodes = read_listing(SOURCE)
    cache = cache / NAME
    builder = ManifestBuilder(dataset_dir, verbose=verbose)
    take = Take(
        name=NAME,
        builder=builder,
        language=LANGUAGE,
        source_split="listing",
        license=LICENSE,
        text_has_casing_punct=True,
        group_kind="recording",
    )
    tally: Counter[str] = Counter()
    if verbose:
        print(f"  listing of {listed_on}: {len(episodes)} episodes")
    order = reading_order(episodes, SEED, SOURCE.spread_by)
    with (
        http_client() as http,
        _client(SEGMENT_MODEL, {"response_format": "verbose_json"}) as whisper_client,
        _client(BASE_MODEL) as parakeet_client,
        closing(prefetched(order, partial(_audio_file, http, cache, SOURCE))) as audio_files,
    ):
        whisper = _answer(whisper_client, LANGUAGE)
        parakeet = _answer(parakeet_client, LANGUAGE)
        episodes_read = 0
        while not take.full():
            batch = list(islice(audio_files, batch_size(take, episodes_read)))
            if not batch:
                break
            label_batch(batch, http, cache, whisper, parakeet, take, tally)
            episodes_read += len(batch)
            if verbose:
                print(f"  {episodes_read} episodes read, {take.kept} clips so far")
    if verbose:
        print(f"  {report(tally)}")
        print(f"  {take.report()}")
    builder.write()
    return take.kept


def label_batch(
    batch: list[tuple[Talk, Path | None]],
    http: httpx.Client,
    cache: Path,
    whisper: Transcribe,
    parakeet: Transcribe,
    take: Take,
    tally: Counter[str],
) -> None:
    """Whisper on every episode of the batch, then the alignments, then Parakeet.

    An episode stops at its share of the cap, and the batch at the room left in
    it, so no Parakeet request is sent for a clip the cap cannot take.
    """
    heard = [heard_talk(episode, path, cache, whisper, tally) for episode, path in batch]
    room = take.cap - take.kept
    for episode in heard:
        if not episode.segments:
            continue
        try:
            lines = transcript_lines(page_text(http, episode.talk, cache / "pages"))
        except (DownloadError, httpx.HTTPError):
            tally["episodes whose page could not be fetched"] += 1
            continue
        clips = candidates(episode.segments, LANGUAGE, tally)
        labelled = labelled_clips(clips, lines, LANGUAGE, tally)
        tally[CANDIDATES] += len(clips)
        tally[LABELLED] += len(labelled)
        episode.agreed = labelled[: min(take.cap_per_group, room)]
        room -= len(episode.agreed)
    for episode in heard:
        answers = Answers(cache / clips_file(episode.talk, BASE_MODEL))
        offer_talk(episode, answers, parakeet, take, LANGUAGE, tally)


def report(tally: Counter[str]) -> str:
    """One line on the labelling: the share the transcripts label, the base model, the rest."""
    seen, labelled = tally[CANDIDATES], tally[LABELLED]
    kept, wrong = tally[KEPT], tally[BASE_WRONG]
    shown = {CANDIDATES, LABELLED, KEPT, BASE_WRONG}
    others = [f"{count} {reason}" for reason, count in sorted(tally.items()) if reason not in shown]
    return (
        f"{labelled} of {seen} candidate clips ({labelled / max(seen, 1):.1%}) "
        f"labelled from the transcripts; the base model gets {wrong} of the {kept} kept clips "
        f"wrong ({wrong / max(kept, 1):.1%}); also {', '.join(others) or 'nothing else'}"
    )
