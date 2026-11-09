"""The French education ministry's PeerTube instances (``peertube-edu-fr-train``).

The ministry runs eleven PeerTube instances on ``apps.education.fr``, one per
teaching area or school level, each with its own catalogue and its own REST API.
Their videos are the ministry's and its teachers' own: lectures, webinars,
conferences and classroom recordings, in French. The list below is committed
here so that a listing reads the same instances on every machine; each video
keeps its instance, which is what the card counts the hours per instance with.

The filter keeps what a training set may hold (D-019): the instance's own videos
under CC BY or CC0, in French, at least ten minutes long, from channels that are
not music or stage performances. Ten minutes is the floor the other talk sources
use too: at the yield the pipeline expects of a talk
(:data:`audiolab.data.talks.EXPECTED_CLIPS_PER_TALK`), a video much shorter than
that cannot fill a batch, and every talk costs a download whether it is long or
short.

The music and stage channels are dropped by name because their videos are of
music, dance or theatre, not of anyone explaining something: the licence and the
language say nothing about that, and the instances publish songs, choirs, dances
and stage performances next to their lessons. The words are matched as whole
words of a channel or account display name, so "Education Musicale" is dropped
and "Main chantallayoussaint channel" is not. A teacher's own channel holds
lessons and the school concert alike, so a title that names a performance
("Concert Chorale 2 Juin 2026", "Spectacle de Noël") is dropped as well; the
title words are fewer, because a lesson is often about music or theatre ("Le
professeur de musique", "Interview radio théâtre"). A channel that glues a music
word into a longer name ("nikkojazz") escapes the filter; ten minutes of it is
rare, and a music video yields no clips anyway, because the two labelling models
do not agree on music.
"""

from __future__ import annotations

import httpx

from audiolab.data import peertube
from audiolab.data.talk_listing import Talk

# The ministry's instances, in the order the PeerTube index lists them. A video
# of one is only usable if that instance hosts it itself (peertube.Filter).
INSTANCES = (
    "https://tube-action-educative.apps.education.fr",
    "https://tube-arts-lettres-sciences-humaines.apps.education.fr",
    "https://tube-cycle-2.apps.education.fr",
    "https://tube-cycle-3.apps.education.fr",
    "https://tube-education-physique-et-sportive.apps.education.fr",
    "https://tube-enseignement-professionnel.apps.education.fr",
    "https://tube-institutionnel.apps.education.fr",
    "https://tube-langues-vivantes.apps.education.fr",
    "https://tube-maternelle.apps.education.fr",
    "https://tube-numerique-educatif.apps.education.fr",
    "https://tube-sciences-technologies.apps.education.fr",
)

# The shortest video a listing may hold.
MIN_TALK_SECONDS = 600

# Whole words of a channel or account name that mean "this is not a talk".
NON_SPEECH_WORDS = (
    "chant",
    "chants",
    "choeur",
    "choeurs",
    "chœur",
    "chœurs",
    "chorale",
    "chorales",
    "comptine",
    "comptines",
    "concert",
    "concerts",
    "danse",
    "jazz",
    "karaoke",
    "karaoké",
    "musicale",
    "music",
    "musique",
    "opera",
    "opéra",
    "orchestre",
    "rap",
    "spectacle",
    "theatre",
    "théâtre",
)

# Whole words of a video title that name a performance rather than a lesson.
PERFORMANCE_WORDS = (
    "choeur",
    "chœur",
    "chorale",
    "chorales",
    "concert",
    "concerts",
    "karaoke",
    "karaoké",
    "musicale",
    "spectacle",
    "spectacles",
)

FILTER = peertube.Filter(
    language="fr",
    min_duration_s=MIN_TALK_SECONDS,
    name_must_not_hold=NON_SPEECH_WORDS,
    title_must_not_hold=PERFORMANCE_WORDS,
)


def list_talks(http: httpx.Client, instances: tuple[str, ...] = INSTANCES) -> list[Talk]:
    """Every usable video of the ministry's instances."""
    return peertube.list_instances(http, instances, FILTER)


def spread_key(talk: Talk) -> str:
    """The instance a talk comes from: the way this source is most uneven.

    One instance hosts several thousand videos and another a few hundred, so a
    plain shuffle over the listing can spend a whole selection on one of them.
    """
    return talk.instance
