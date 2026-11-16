"""Duplicate talks: classes of re-uploads a preparation excludes whole.

Nothing a model is scored on may be trained on (decision 0002). The rule is
not on the id alone: the same recording often comes back under another id —
a video re-uploaded on the same instance, an episode re-released, a
compilation that replays an episode — and an exclusion of ids misses those
copies. The rule is on the recording: the talks of a corpus fall into
*duplicate classes* (:func:`duplicate_classes`), a preparation leaves out
every class that holds an evaluation group (:func:`excluded_talks`), and
reads each other class at most once, the first talk of its reading order
standing for it.

Two talks are in the same class when either of two separate signals says so,
each one alone enough:

- the same title and length: their normalized titles are equal (case,
  accents and punctuation set aside) and their durations differ by at most
  :data:`DURATION_TOLERANCE_S`; a re-upload keeps the title and changes the
  cut by seconds at most, while a different cut is another recording;
- a confirmed pair: an entry of the committed ``duplicates.json`` beside
  this module, where two ids were checked against their prepared clips to
  hold the same speech, with the evidence that confirmed them.

The classes are closed transitively: a twin of a twin of an evaluation talk
is one. A shared author, channel or host never puts two talks in the same
class: the rule takes only the title, the duration and the confirmed pairs,
so the same host on both sides is not a leak — only the same speech is.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path

# How far apart the durations of two copies of one recording may be, in
# seconds: a re-upload is trimmed or padded at the edges, a different cut
# lasts seconds to minutes.
DURATION_TOLERANCE_S = 2.0

# The confirmed pairs, one line each: the domain of the listing they live in
# ("peertube" the union of the three PeerTube listings, "nasa" the NASA
# listing), then the two ids. The evidence that confirmed each pair is the
# "reason" beside them in duplicates.json.
CONFIRMED_PAIRS: tuple[tuple[str, str, str], ...] = tuple(
    (entry["domain"], entry["ids"][0], entry["ids"][1])
    for entry in json.loads(Path(__file__).with_name("duplicates.json").read_text(encoding="utf-8"))
)


def _pair_ids(domain: str) -> list[tuple[str, str]]:
    return [(first, second) for dom, first, second in CONFIRMED_PAIRS if dom == domain]


def record_fields(record: Mapping | object) -> tuple[str, str, float]:
    """The (id, title, duration_s) of one talk record.

    A talk record is a mapping with the keys ``id``, ``title`` and
    ``duration_s`` — a row of a pinned listing read back — or an object with
    those attributes, like the ``Talk`` of :mod:`audiolab.data.talk_listing`.
    Nothing else of a talk — author, channel, host, licence — is a class
    member: the rule takes the three, and no shared field other than them
    ever merges two talks.
    """
    if isinstance(record, Mapping):
        return str(record["id"]), str(record["title"]), float(record["duration_s"])
    return str(record.id), str(record.title), float(record.duration_s)


def normalized_title(title: str) -> str:
    """What two copies of one title share: its letters and digits, one space between runs.

    The title is lowercased, its accents stripped (NFKD decomposition without
    the combining marks), and every run of anything that is not a letter or a
    digit is gone, one space where it was, trimmed: "Écrire son propre Rust
    linter !" and "ecrire son propre rust linter" are the same title.
    """
    text = unicodedata.normalize("NFKD", title).lower()
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(re.findall(r"[^\W_]+", text, re.UNICODE))


def duplicate_classes(
    talks: Iterable[Mapping | object],
    domain: str = "peertube",
    pairs: Iterable[tuple[str, str]] | None = None,
) -> dict[str, str]:
    """The duplicate class of every talk: its id mapped to the class's id.

    ``talks`` are talk records (a list of mappings or objects with ``id``,
    ``title`` and ``duration_s``). Two talks are in the same class when their
    normalized titles are equal and their durations differ by at most
    :data:`DURATION_TOLERANCE_S`, or when they form one of ``pairs`` — by
    default the confirmed pairs of ``domain`` from
    ``src/audiolab/data/duplicates.json``; the classes are closed
    transitively. The class's id is the smallest id of its members, so a
    preparation can group its talks by class with no other id. A shared
    author, channel or host never puts two talks in the same class.

    The function reads nothing but its arguments — the confirmed pairs are
    the committed file read at import, like the reserved NASA episodes — so
    a fetcher and the measurement of what a prepared set would hold use the
    same rule.
    """
    if pairs is None:
        pairs = _pair_ids(domain)

    parent: dict[str, str] = {}

    def find(talk_id: str) -> str:
        root = talk_id
        while parent[root] != root:
            root = parent[root]
        while parent[talk_id] != root:
            parent[talk_id], talk_id = root, parent[talk_id]
        return root

    def union(first: str, second: str) -> None:
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    info: dict[str, tuple[str, float]] = {}
    for record in talks:
        talk_id, title, duration = record_fields(record)
        info.setdefault(talk_id, (normalized_title(title), duration))
        parent.setdefault(talk_id, talk_id)

    # Same title, close length: within a title the talks are sorted by
    # duration and every talk is joined to its predecessor, so a run merged
    # over gaps of at most the tolerance is one class, whatever its ends.
    by_title: dict[str, list[str]] = {}
    for talk_id, (title, _) in info.items():
        by_title.setdefault(title, []).append(talk_id)
    for ids in by_title.values():
        ids.sort(key=lambda talk_id: (info[talk_id][1], talk_id))
        for first, second in zip(ids, ids[1:], strict=False):
            if info[second][1] - info[first][1] <= DURATION_TOLERANCE_S:
                union(first, second)

    for first, second in pairs:
        if first in info and second in info:
            union(first, second)

    members: dict[str, list[str]] = {}
    for talk_id in info:
        members.setdefault(find(talk_id), []).append(talk_id)
    return {talk_id: min(group) for group in members.values() for talk_id in group}


def excluded_talks(classes: Mapping[str, str], reserved: Iterable[str]) -> dict[str, str]:
    """The talks a preparation leaves out before it reads any, and why.

    ``classes`` maps a talk id to its duplicate class; ``reserved`` are the
    ids of the evaluation groups of the corpus. A talk is left out when its
    id is reserved — reason ``"evaluation talk"`` — or when its class holds
    a reserved id — reason ``"duplicate of an evaluation talk"``. The
    re-uploads of a talk the preparation then reads are not here: they
    depend on the reading order, which the preparation counts on its own.
    """
    reserved_ids = frozenset(reserved)
    blocked = frozenset(classes[talk_id] for talk_id in reserved_ids if talk_id in classes)
    left_out: dict[str, str] = {}
    for talk_id, class_id in classes.items():
        if talk_id in reserved_ids:
            left_out[talk_id] = "evaluation talk"
        elif class_id in blocked:
            left_out[talk_id] = "duplicate of an evaluation talk"
    return left_out


# The three listings the PeerTube sources are one corpus of: the evaluation
# talks and the training talks come from different ones, and a re-upload can
# straddle them, so the classes of the PeerTube sources span the union.
PEERTUBE_LISTINGS = (
    "peertube_listing.json",
    "peertube_edu_listing.json",
    "peertube_tech_listing.json",
)


def listing_records(path: str | Path) -> list[dict]:
    """The talk records of a pinned listing: one mapping per talk.

    The listing is what a preparation reads, never the live catalogue, so a
    row holds everything a duplicate class needs — the id, the title, the
    duration — in the order its ``fields`` names them.
    """
    listing = json.loads(Path(path).read_text(encoding="utf-8"))
    return [dict(zip(listing["fields"], row, strict=True)) for row in listing["talks"]]


def peertube_talks() -> list[dict]:
    """The PeerTube talks of the three pinned listings, one record per id.

    An id in two listings is the same talk (the tech listing is the union of
    the other two, kept by title), so the first record of it stands for it.
    """
    records: dict[str, dict] = {}
    for name in PEERTUBE_LISTINGS:
        for record in listing_records(Path(__file__).with_name(name)):
            records.setdefault(record["id"], record)
    return list(records.values())
