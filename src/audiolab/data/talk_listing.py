"""The pinned listing of a talk source: which talks it may use, and in what order.

A catalogue is live: videos are published, edited and deleted, and a training
source has to be reproducible. So the list of usable talks is made once, by
``audiolab data list-talks``, and committed as a JSON file: metadata only, with
the date it was listed on and the filter it was made with. A preparation reads
that file and never the catalogue, in the seeded order :func:`reading_order`
gives it.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from audiolab.data.download import DownloadError


@dataclass(frozen=True)
class Talk:
    """One talk of a pinned listing: where it is, what it is, and its licence.

    ``channel`` is the channel, the room or the collection the source files it
    under, and is empty when the source has none. ``instance`` is the PeerTube
    instance whose API serves the video files, and is empty when ``url`` already
    is the audio file: only a PeerTube file URL changes under a talk (an
    instance transcodes again), which is why it is asked for at download time
    rather than pinned.
    """

    id: str
    url: str
    title: str
    duration_s: int
    license: str
    author: str
    channel: str = ""
    instance: str = ""


@dataclass(frozen=True)
class TalkSource:
    """One source of talks, with what the pipeline must know about it.

    ``lister`` reads the live catalogue and returns the talks a listing may hold;
    it is used by ``audiolab data list-talks`` only. ``audio_url`` says how to
    find the file to download for one listed talk. ``listing_filter`` is the
    record of what ``lister`` keeps, written into the listing so a reader can see
    what was asked for and when. ``spread_by`` is what a seeded selection spreads
    over (a conference, an instance, a room), and is what keeps the first talks
    of a source from all coming from the same corner of it.
    """

    name: str
    language: str
    allowed_licenses: frozenset[str]
    listing_path: Path
    listing_filter: dict[str, object]
    lister: Callable[[httpx.Client], list[Talk]]
    audio_url: Callable[[httpx.Client, Talk], str]
    spread_by: Callable[[Talk], str] | None = None


def write_listing(source: TalkSource, listed_on: str, talks: list[Talk]) -> None:
    """Pin the usable talks of one source, with the date and the filter they came from."""
    listing = {
        "source": source.name,
        "listed_on": listed_on,
        "filter": source.listing_filter,
        "talks": [asdict(talk) for talk in talks],
    }
    source.listing_path.write_text(
        json.dumps(listing, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


def read_listing(source: TalkSource) -> tuple[str, list[Talk]]:
    """The date of the pinned listing of ``source`` and its talks.

    A talk whose licence is not one this source may train on is refused rather
    than skipped: the file was written by :func:`write_listing` from talks that
    passed the filter, so a licence that does not is a wrong file (D-019).
    """
    listing = json.loads(source.listing_path.read_text(encoding="utf-8"))
    talks = [Talk(**row) for row in listing["talks"]]
    unusable = [talk.id for talk in talks if talk.license not in source.allowed_licenses]
    if unusable:
        raise DownloadError(f"{source.listing_path}: talks under another licence: {unusable[:3]}")
    return listing["listed_on"], talks


def reading_order(
    talks: list[Talk], seed: int, spread_by: Callable[[Talk], str] | None = None
) -> list[Talk]:
    """The listed talks in a seeded order, the same on every machine.

    A catalogue publishes one conference, instance or year after another, so the
    order of the listing would spend the clip cap on a handful of them. The
    talks are shuffled with a fixed seed, which is reproducible, and then taken
    one per group in turn (:func:`spread`), so that the first talks read cover
    every group the source has.
    """
    shuffled = sorted(talks, key=lambda talk: talk.id)
    random.Random(seed).shuffle(shuffled)
    return spread(shuffled, spread_by) if spread_by else shuffled


def spread(talks: list[Talk], group_of: Callable[[Talk], str]) -> list[Talk]:
    """The talks reordered so that each group appears once before any appears twice.

    The groups are the ones of the shuffle's first appearance, and the talks
    inside a group keep their shuffled order, so the result is as reproducible as
    the shuffle it is made from.
    """
    groups: dict[str, list[Talk]] = {}
    for talk in talks:
        groups.setdefault(group_of(talk), []).append(talk)
    longest = max((len(group) for group in groups.values()), default=0)
    return [
        group[position]
        for position in range(longest)
        for group in groups.values()
        if position < len(group)
    ]
