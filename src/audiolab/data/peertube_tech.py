"""The tech talks of the two French PeerTube listings, as one source.

The talks of ``peertube-tech-fr-train`` are the union of the pinned listings
of the ``peertube-fr-train`` and the ``peertube-edu-fr-train`` sources, kept
when the title carries a word of the technology of the model the campaign
trains — the list of words is the source's own definition, in
:data:`TECH_TITLE`, so the listing it pins stays re-derivable from what the
two parents pinned.
"""

from __future__ import annotations

import re

from audiolab.data.talk_listing import read_listing

#: The words a talk title must carry for the talk to be a tech talk.
TECH_TITLE = re.compile(
    r"informatique|programm|python|algorithm|\bnsi\b|numérique|réseau|linux|donnée"
    r"|\bcode|logiciel|\bweb\b|intelligence artificielle|\bia\b|cyber|robot"
    r"|base de données|javascript|html|tableur|développ|open ?source|libre",
    re.IGNORECASE,
)


def list_union(fr, edu):
    """The talks of both listings, kept by :data:`TECH_TITLE`, in one order.

    The two listings are pinned files, so this reads them rather than the
    network. A talk present under both listings keeps its seat in the first.
    """
    talks = [talk for listing in (fr, edu) for talk in read_listing(listing)[1]]
    seen: set[str] = set()
    return [
        talk
        for talk in talks
        if talk.id not in seen and not seen.add(talk.id) and TECH_TITLE.search(talk.title or "")
    ]
