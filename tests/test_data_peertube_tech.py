"""The tech talks of the two French PeerTube listings: the title filter, the union.

The two parents' listings are pinned files, so the union is tested offline,
against them and against tiny listings of its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from test_data_talks import make_source, talk  # noqa: E402

from audiolab.data import peertube_tech  # noqa: E402
from audiolab.data.talk_listing import read_listing  # noqa: E402
from audiolab.data.talk_sources import SOURCES  # noqa: E402


def test_the_words_a_title_must_carry():
    for title in (
        "Python : les bases",
        "Intelligence Artificielle",
        "Développement Web",
        "Open source en pratique",
        "L'IA au service du réseau",
        "Linux et le code",
        "NSI au lycée",
        "Les données de la ville",
        "Le web en HTML",
        "JavaScript et tableur",
        "Cyberattaque",
        "Robotique",
        "Bases de données",
        "Numérique et citoyenneté",
        "Algorithmique",
        "Logiciel libre",
    ):
        assert peertube_tech.TECH_TITLE.search(title), title
    for title in (
        "L'été au Capitole",
        "Cuisine au feu de bois",
        "Histoire de la musique",
        "Le sport à l'école",
        "Voyage en Provence",
    ):
        assert not peertube_tech.TECH_TITLE.search(title), title


def test_the_union_keeps_the_tech_titles_of_both_parents_the_first_seat_winning(
    tmp_path,
):
    fr = make_source(
        tmp_path,
        [
            talk("a", title="Python : les bases"),
            talk("b", title="Cuisine au feu de bois"),
            talk("c", title="Le développement web"),
        ],
        name="fr",
        listing_path=tmp_path / "fr-listing.json",
    )
    edu = make_source(
        tmp_path,
        [
            talk("c", title="Le développement web", channel="Salle 12"),
            talk("d", title="Musique et culture"),
            talk("e", title="Introduction à l'IA"),
        ],
        name="edu",
        listing_path=tmp_path / "edu-listing.json",
    )

    union = peertube_tech.list_union(fr, edu)

    assert [one.id for one in union] == ["a", "c", "e"], (
        "the tech titles of both parents, the shared id once, the rest left out"
    )
    assert union[1].channel == "Talks", "the first seat is the Capitole's"


def test_the_source_is_registered_on_the_parents_pinned_listings():
    source = SOURCES["peertube-tech-fr-train"]
    assert source.listing_path.name == "peertube_tech_listing.json"
    assert source.evaluation_datasets == ("peertube-edu-captions-fr",)
    assert source.spread_by(talk("a")) == talk("a").instance

    listed_on, union = read_listing(source)
    assert (listed_on, len(union)) == ("2026-11-13", 421), "the pinned union of the two parents"
    assert union == source.lister(None), "the pin is what the two parents' pins give again"
    fr, edu = (
        read_listing(SOURCES[name])[1] for name in ("peertube-fr-train", "peertube-edu-fr-train")
    )
    is_tech = lambda one: peertube_tech.TECH_TITLE.search(one.title or "")  # noqa: E731
    fr_ids = {one.id for one in fr}
    ids = [one.id for one in union]
    assert set(ids) == {one.id for one in fr if is_tech(one)} | {
        one.id for one in edu if is_tech(one)
    }, "exactly the tech titles of the two parents"
    assert ids == [one.id for one in fr if is_tech(one)] + [
        one.id for one in edu if is_tech(one) and one.id not in fr_ids
    ], "the Capitole's seats first, then the ministry's"
