"""The ministry's PeerTube instances, their filter, and their pinned listing.

The instances are replaced by a fake transport, so these tests check which
instances a listing asks, which videos it keeps, and that a talk remembers the
instance it came from.
"""

from __future__ import annotations

import httpx
import pytest

from audiolab.data import peertube, peertube_edu, talk_sources
from audiolab.data.talks import read_listing

SOURCE = talk_sources.SOURCES["peertube-edu-fr-train"]
HOST = "tube-cycle-2.apps.education.fr"
INSTANCE = f"https://{HOST}"


@pytest.fixture(autouse=True)
def no_pause(monkeypatch):
    monkeypatch.setattr(peertube.time, "sleep", lambda seconds: None)


def api_row(uuid: str, host: str = HOST, **fields) -> dict:
    """One video as an instance lists it, with only the fields this source reads."""
    row = {
        "uuid": uuid,
        "url": f"https://{host}/videos/watch/{uuid}",
        "name": f"Video {uuid}",
        "duration": 1200,
        "licence": {"id": 1, "label": "Attribution"},
        "language": {"id": "fr", "label": "Français"},
        "isLocal": True,
        "channel": {"displayName": "Main une.enseignante channel"},
        "account": {"displayName": "une.enseignante"},
    }
    row.update(fields)
    return row


def test_the_instance_list_is_the_ministry_s_instances():
    assert len(peertube_edu.INSTANCES) == len(set(peertube_edu.INSTANCES)) == 11
    assert all(host.startswith("https://tube-") for host in peertube_edu.INSTANCES)
    assert all(host.endswith(".apps.education.fr") for host in peertube_edu.INSTANCES)
    assert "https://tube-numerique-educatif.apps.education.fr" in peertube_edu.INSTANCES
    assert not any("test" in host for host in peertube_edu.INSTANCES)


@pytest.mark.parametrize(
    ("name", "kept"),
    [
        ("Main une.enseignante channel", True),
        ("Radio-Doué", True),
        ("Main chantallayoussaint channel", True),  # "chantal" is not "chant"
        ("Education Musicale Ac-Créteil", False),
        ("La Chorale de Jean Zay !", False),
        ("Les comptines de Maitresse Charlotte", False),
        ("Creil fait son théâtre 2026", False),
        ("Éducation musicale et chant choral", False),
        ("Ville en choeur 2026", False),
        ("Creil Ville en chœur 2025", False),
        ("Karaoké du CM2", False),
        ("nikkojazz YT", True),  # a word glued into a longer one escapes the filter
    ],
)
def test_channels_that_are_not_speech_are_left_out(name, kept):
    row = api_row("v1", channel={"displayName": name}, account={"displayName": name})
    assert peertube.usable(row, INSTANCE, peertube_edu.FILTER) is kept


@pytest.mark.parametrize(
    ("title", "kept"),
    [
        ("Le professeur de musique", True),
        ("Interview radio théâtre", True),
        ("Concert Chorale 2 Juin 2026", False),
        ("SPECTACLE DE NOEL V2", False),
        ("Comédie musicale - Voyage en Enfer", False),
        ("Concert du 26 mai 2026 - En choeur, plus fort", False),
    ],
)
def test_a_title_that_names_a_performance_is_left_out(title, kept):
    row = api_row("v1", name=title)
    assert peertube.usable(row, INSTANCE, peertube_edu.FILTER) is kept


@pytest.mark.parametrize(
    ("fields", "kept"),
    [
        ({}, True),
        ({"duration": 599}, False),
        ({"licence": {"id": 2}}, False),
        ({"language": {"id": "en"}}, False),
        ({"language": {"id": None}}, False),
        ({"isLocal": False}, False),
    ],
)
def test_the_filter_of_the_source_says_what_a_listing_may_hold(fields, kept):
    row = api_row("v1", **fields)
    assert peertube.usable(row, INSTANCE, peertube_edu.FILTER) is kept


def test_a_listing_asks_every_instance_and_a_talk_remembers_its_own():
    hosts = {
        HOST: [api_row("a"), api_row("short", duration=60)],
        "tube-cycle-3.apps.education.fr": [api_row("b", host="tube-cycle-3.apps.education.fr")],
        "tube-maternelle.apps.education.fr": [],
    }
    asked: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        asked.append(host)
        rows = hosts[host]
        start = int(request.url.params["start"])
        return httpx.Response(200, json={"total": len(rows), "data": rows[start : start + 2]})

    instances = tuple(f"https://{host}" for host in hosts)
    with httpx.Client(transport=httpx.MockTransport(answer)) as http:
        talks = peertube_edu.list_talks(http, instances)

    assert sorted(asked) == sorted(hosts), "every instance of the source is asked"
    assert [talk.id for talk in talks] == ["a", "b"]
    assert [talk.instance for talk in talks] == list(instances[:2])
    assert all(talk.url.startswith(talk.instance + "/") for talk in talks)
    assert talks[0].duration_s == 1200


def test_the_pinned_listing_holds_the_ministry_s_own_long_french_videos():
    listed_on, talks = read_listing(SOURCE)

    assert listed_on and len(talks) == len({talk.id for talk in talks}) > 0
    assert all(talk.instance in peertube_edu.INSTANCES for talk in talks)
    assert all(talk.url.startswith(talk.instance + "/") for talk in talks)
    assert all(talk.duration_s >= peertube_edu.MIN_TALK_SECONDS for talk in talks)
    assert all(talk.license in SOURCE.allowed_licenses for talk in talks)
    assert all(talk.title and talk.author and talk.channel for talk in talks)
    assert len({talk.instance for talk in talks}) == len(peertube_edu.INSTANCES)
    left_out = set(peertube_edu.NON_SPEECH_WORDS)
    assert not any(peertube.words_of(f"{talk.channel} {talk.author}") & left_out for talk in talks)
    performances = set(peertube_edu.PERFORMANCE_WORDS)
    assert not any(peertube.words_of(talk.title) & performances for talk in talks)
    assert SOURCE.listing_filter["channel_words_left_out"] == list(peertube_edu.NON_SPEECH_WORDS)
