"""What the DebConf listing of the Debian PeerTube instance keeps, and why.

Almost every video of that instance is untagged, so the filter cannot be about
the language tag alone: these tests check the two rules that stand in for it —
the channel must be a DebConf one, and the editions that speak Portuguese or
Spanish are dropped by name.
"""

from __future__ import annotations

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
        (api_row("j", channel="DebConf 15", language={"id": "pt"}), False),
        (api_row("k", channel="DebConf 15", language={"id": "en"}), True),
        (api_row("l", channel="DebConf 15", licence={"id": 2}), False),
        (api_row("m", channel="DebConf 15", duration=300), False),
        (api_row("n", channel="DebConf 15", isLocal=False), False),
    ],
)
def test_only_english_speaking_debconf_talks_of_the_instance_are_listed(row, kept):
    assert peertube.usable(row, INSTANCE, debconf.FILTER) is kept


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
        {"brazil", "brasil", "portugal", "argentina", "es"} & peertube.words_of(talk.channel)
        for talk in talks
    ), "the Portuguese- and Spanish-speaking editions are not English"
    assert len({talk.channel for talk in talks}) > 5, "several conferences, not one"
