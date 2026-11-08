"""What the DebConf listing of the Debian PeerTube instance keeps, and why.

Almost every video of that instance is untagged, so the filter cannot be about
the language tag alone: these tests check the two rules that stand in for it —
the channel must be a DebConf one, and the editions that speak Portuguese or
Spanish are dropped by name.
"""

from __future__ import annotations

import httpx
import pytest

from audiolab.data import debconf, peertube, talk_sources
from audiolab.data.talks import read_listing

SOURCE = talk_sources.SOURCES["debconf-en-train"]
INSTANCE = debconf.INSTANCE


def api_row(uuid: str, *, channel: str, account: str = "DebConf Videos", **fields) -> dict:
    """One video as the Debian instance lists it: most of them carry no language."""
    row = {
        "uuid": uuid,
        "url": f"{INSTANCE}/w/{uuid}",
        "name": f"Talk {uuid}",
        "duration": 2400,
        "licence": {"id": 1, "label": "Attribution"},
        "language": None,
        "isLocal": True,
        "channel": {"displayName": channel},
        "account": {"displayName": account},
    }
    row.update(fields)
    return row


@pytest.mark.parametrize(
    ("row", "kept"),
    [
        (api_row("a", channel="DebConf 25"), True),
        (api_row("b", channel="DebConf ES"), False),
        (api_row("c", channel="mini-DebConf Brazil"), False),
        (api_row("d", channel="miniDebConf Argentina"), False),
        (api_row("e", channel="MiniDebConf Portugal"), False),
        (api_row("f", channel="Andreas @ DebConf", account="Andreas Tille"), True),
        (api_row("g", channel="Main debianbrazilteam channel", account="Debian Brasil"), False),
        (api_row("h", channel="Phosh.mobi", account="Phosh.mobi"), False),
        (api_row("i", channel="mini-DebConf Germany"), True),
        (api_row("o", channel="miniDebConf India 2021"), False),
        (api_row("p", channel="mini-DebConf France"), False),
        (api_row("q", channel="Debian-edu Gatherings"), False),
        (api_row("j", channel="DebConf 15", language={"id": "pt"}), False),
        (api_row("k", channel="DebConf 15", language={"id": "en"}), True),
        (api_row("l", channel="DebConf 15", licence={"id": 2}), False),
        (api_row("m", channel="DebConf 15", duration=300), False),
        (api_row("n", channel="DebConf 15", isLocal=False), False),
    ],
)
def test_only_english_speaking_debconf_talks_of_the_instance_are_listed(row, kept):
    assert peertube.usable(row, INSTANCE, debconf.FILTER) is kept


@pytest.mark.parametrize(
    ("title", "english"),
    [
        ("Bits from the DPL", True),
        ("piuparts.d.o: testing that .deb packages behave", True),
        ("How to E-Mail - sending and receiving email properly", True),
        ("De-vendoring Gnulib in Debian", True),
        ("Shell ZSH en Debian", False),
        ("Introducción al Software Libre", False),
        ("Comunidades: o bom, o ruim e o maravilhoso", False),
        ("Bisection de l'Archive avec snapshot", False),
        ("Mein Leben mit Luca", False),
        ("എന്താണ് ഡെബിയൻ ?", False),
        ("跨社群合作：LibreOffice 宣傳動畫幕後花絮", False),
        ("Debian in Retro Computers [పాత కంప్యూటర్లలో డెబియన్]", False),
    ],
)
def test_a_talk_whose_title_is_not_english_is_not_listed(title, english):
    assert debconf.english_title(title) is english


def test_a_listing_keeps_only_the_talks_with_an_english_title(monkeypatch):
    monkeypatch.setattr(peertube.time, "sleep", lambda seconds: None)
    rows = [
        api_row("a", channel="DebConf 26", name="Debian LTS BoF"),
        api_row("b", channel="DebConf 26", name="Licencia para ignorar: Por qué ya nadie lee"),
    ]
    answer = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"total": len(rows), "data": rows})
    )
    with httpx.Client(transport=answer) as http:
        assert [talk.id for talk in debconf.list_talks(http)] == ["a"]


def test_the_pinned_listing_holds_debconf_talks_of_the_debian_instance():
    listed_on, talks = read_listing(SOURCE)

    assert listed_on and len(talks) == len({talk.id for talk in talks}) > 0
    assert all(talk.instance == INSTANCE and talk.url.startswith(INSTANCE + "/") for talk in talks)
    assert all(talk.duration_s >= debconf.MIN_TALK_SECONDS for talk in talks)
    assert all(talk.license in SOURCE.allowed_licenses for talk in talks)
    assert all(
        "debconf" in talk.channel.lower() or "debconf" in talk.author.lower() for talk in talks
    )
    assert not any(
        set(debconf.NON_ENGLISH_EDITIONS) & peertube.words_of(talk.channel) for talk in talks
    ), "the editions held in another language are not listed"
    assert all(debconf.english_title(talk.title) for talk in talks)
    assert len({talk.channel for talk in talks}) > 5, "several conferences, not one"
