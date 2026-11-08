"""DebConf talks on the Debian PeerTube instance (``debconf-en-train``).

The Debian project publishes the recordings of DebConf and of its
mini-conferences on ``peertube.debian.social``, from DebConf 4 (2004) to the
current year. The talks are in English — the conference is held in English
wherever it takes place — on technical subjects full of project and package
names.

Two things about that instance decide the filter:

- it leaves the language of almost every video untagged: of the 2 779 local
  videos under those licences, 121 carry the English tag and 2 343 no tag at all.
  Filtering on the tag alone would keep 4 % of the corpus, so an untagged video is
  kept and a video tagged in another language is dropped; the channel is what says
  an untagged video is a DebConf talk at all, so a channel or account name must
  hold the word "debconf".
- the Portuguese- and Spanish-speaking editions are untagged as well (mini-
  DebConf Brazil, DebConf ES, miniDebConf Argentina, MiniDebConf Portugal), so
  they are dropped by name instead: their talks are not English, which is the
  language this source is labelled in.

The licence filter is the usual one: CC BY or CC0 only (D-019).
"""

from __future__ import annotations

import httpx

from audiolab.data import peertube
from audiolab.data.talks import Talk

INSTANCE = "https://peertube.debian.social"

# The shortest video a listing may hold, as for the other talk sources.
MIN_TALK_SECONDS = 600

# Whole words of a channel or account name of an edition that does not speak
# English: DebConf ES is Spanish, miniDebConf Argentina too, and the Brazilian
# and Portuguese editions are in Portuguese.
NON_ENGLISH_EDITIONS = ("argentina", "brazil", "brasil", "es", "portugal")

FILTER = peertube.Filter(
    language="en",
    language_unset_ok=True,
    min_duration_s=MIN_TALK_SECONDS,
    name_must_hold="debconf",
    name_must_not_hold=NON_ENGLISH_EDITIONS,
)


def list_talks(http: httpx.Client, instance: str = INSTANCE) -> list[Talk]:
    """Every usable DebConf talk the instance hosts itself."""
    return peertube.list_instance(http, instance, FILTER)


def spread_key(talk: Talk) -> str:
    """The conference a talk belongs to: one channel per DebConf or mini-DebConf.

    Some editions published a hundred talks and others a handful, so a plain
    shuffle over the listing can spend a whole selection on one year.
    """
    return talk.channel
