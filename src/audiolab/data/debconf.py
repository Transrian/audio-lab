"""DebConf talks on the Debian PeerTube instance (``debconf-en-train``).

The Debian project publishes the recordings of DebConf and of its
mini-conferences on ``peertube.debian.social``, from DebConf 4 (2004) to the
current year. Most talks are in English — the conference is held in English
wherever it takes place — on technical subjects full of project and package
names.

Three things about that instance decide the filter:

- it leaves the language of almost every video untagged: of the 2 779 local
  videos under those licences, 121 carry the English tag and 2 343 no tag at all.
  Filtering on the tag alone would keep 4 % of the corpus, so an untagged video is
  kept and a video tagged in another language is dropped; the channel is what says
  an untagged video is a DebConf talk at all, so a channel or account name must
  hold the word "debconf".
- the editions that are not held in English are untagged as well (mini-DebConf
  Brazil, DebConf ES, miniDebConf Argentina, MiniDebConf Portugal, the
  mini-DebConf France, the Indian miniDebConf of 2021, whose talks are in Hindi,
  Malayalam, Marathi and Telugu, and the German Debian-edu gatherings), so they
  are dropped by name instead.
- an edition held in English still has talks in the language of its host
  country, and those are untagged too: Spanish at DebConf 9, 12, 20 and 26,
  Portuguese at DebConf 19, Chinese at DebConf 18, Malayalam and Telugu at
  DebConf 20 and 21. Their titles are in that language, so a talk is dropped
  when its title is not written in English (:func:`english_title`). A talk given
  in another language under an English title escapes this.

A talk that is not in English must not be listed at all: the source is labelled
in English, and Whisper told English translates what it hears, so the two
labelling models can agree on a translation and keep it as a label.

The licence filter is the usual one: CC BY or CC0 only (D-019).
"""

from __future__ import annotations

import unicodedata

import httpx

from audiolab.data import peertube
from audiolab.data.talk_listing import Talk

INSTANCE = "https://peertube.debian.social"

# The shortest video a listing may hold, as for the other talk sources.
MIN_TALK_SECONDS = 600

# Whole words of a channel or account name of an edition that does not speak
# English: DebConf ES and miniDebConf Argentina are Spanish, the Brazilian and
# Portuguese editions Portuguese, the mini-DebConf France French, the Debian-edu
# gatherings German, and the miniDebConf India of 2021 was held in four Indian
# languages.
NON_ENGLISH_EDITIONS = (
    "argentina",
    "brazil",
    "brasil",
    "es",
    "france",
    "gatherings",
    "india",
    "portugal",
)

# Words that a title in Spanish, Portuguese, French or German uses and a title in
# English does not, and the letters of those languages that English does not
# write. One of either is enough: "Shell ZSH en Debian" is a Spanish talk.
NON_ENGLISH_WORDS = frozenset(
    "al como cómo con del el en la las los mi para por qué una y"  # Spanish
    " com da das dos em na nas o ou um uma"  # Portuguese
    " avec dans des du et les pour sur une"  # French
    " der eine für mein mit und von".split()  # German
)
NON_ENGLISH_LETTERS = frozenset("áâãçêíñóõú¿¡")
# What is cut off a word of a title before it is compared.
PUNCTUATION = ".,:;!?¿¡()[]\"'|"

FILTER = peertube.Filter(
    language="en",
    language_unset_ok=True,
    min_duration_s=MIN_TALK_SECONDS,
    name_must_hold="debconf",
    name_must_not_hold=NON_ENGLISH_EDITIONS,
)


def list_talks(http: httpx.Client, instance: str = INSTANCE) -> list[Talk]:
    """Every usable DebConf talk the instance hosts itself."""
    return [
        talk for talk in peertube.list_instance(http, instance, FILTER) if english_title(talk.title)
    ]


def english_title(title: str) -> bool:
    """Whether a title is written in English, as far as its words and letters say.

    A title in another script (Chinese, Malayalam, Telugu, Devanagari) is not, and
    neither is one holding a word or a letter of :data:`NON_ENGLISH_WORDS` and
    :data:`NON_ENGLISH_LETTERS`. A word is cut at spaces only, so the ``o`` of
    ``ftp.d.o`` is not a word of its own. The rule costs a few English titles that
    name a place (``Brasília``) and drops every Spanish, Portuguese, French and
    German title of the listing.
    """
    if any(char.isalpha() and not unicodedata.name(char, "").startswith("LATIN") for char in title):
        return False
    if NON_ENGLISH_LETTERS.intersection(title.lower()):
        return False
    words = {word.strip(PUNCTUATION) for word in title.lower().split()}
    return not words & NON_ENGLISH_WORDS


def spread_key(talk: Talk) -> str:
    """The conference a talk belongs to: one channel per DebConf or mini-DebConf.

    Some editions published a hundred talks and others a handful, so a plain
    shuffle over the listing can spend a whole selection on one year.
    """
    return talk.channel
