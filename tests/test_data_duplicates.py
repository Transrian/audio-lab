"""Duplicate classes: the re-uploads a preparation leaves out, whole, on CPU.

A recording comes back under another id — a video re-uploaded, an episode
re-released, a compilation replaying an episode — and an exclusion of ids
alone misses the copy. The rule is on the recording: the talks of a corpus
fall into duplicate classes, a class that holds an evaluation group is left
out whole, and each other class is read once. Synthetic listings feed the
rule itself here; the preparations that apply it are checked in
``test_data_talks.py`` and ``test_data_nasa.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from audiolab.data import duplicates  # noqa: E402


def record(talk_id: str, title: str = "", duration_s: int | float = 600) -> dict:
    """One talk record: what a duplicate class sees of a talk."""
    return {
        "id": talk_id,
        "title": title if title else f"Talk {talk_id}",
        "duration_s": duration_s,
    }


# The rule itself.


def test_the_normalized_title_keeps_the_letters_and_digits_only():
    norm = duplicates.normalized_title
    assert norm("Écrire son propre Rust linter !") == "ecrire son propre rust linter"
    assert norm("ecrire   son propre rust linter...") == "ecrire son propre rust linter"
    assert norm("C++98") == "c 98"
    assert norm("« to be continuous », mon pipeline GitLab CI") == (
        "to be continuous mon pipeline gitlab ci"
    )
    # U+2011, the non-breaking hyphen: not a letter, not a digit.
    assert norm("Ergo‑L : le reboot de Bépo") == "ergo l le reboot de bepo"


def test_a_twin_of_an_evaluation_talk_is_left_out_and_a_differently_cut_one_is_not():
    talks = [
        record("eval", "Regards croisés sur l'IA", 3910),
        record("twin", "Regards croisés sur l'IA", 3910),
        record("other", "Regards croisés sur l'IA", 3920),
    ]
    left_out = duplicates.excluded_talks(
        duplicates.duplicate_classes(talks, domain="none"), {"eval"}
    )

    assert left_out == {
        "eval": "evaluation talk",
        "twin": "duplicate of an evaluation talk",
    }, "the same title within 2 s is the recording; 10 s is another cut"


def test_a_confirmed_pair_is_left_out_even_when_the_titles_differ():
    talks = [
        record("eval", "The evaluation talk", 3910),
        record("reupload", "The same talk, re-titled", 3905),
        record("unrelated", "Another talk", 1000),
    ]
    classes = duplicates.duplicate_classes(talks, domain="none", pairs=[("eval", "reupload")])
    left_out = duplicates.excluded_talks(classes, {"eval"})

    assert left_out == {
        "eval": "evaluation talk",
        "reupload": "duplicate of an evaluation talk",
    }, "the confirmed pair links the re-upload to the evaluation talk"


def test_a_class_is_closed_transitively_over_pairs():
    # a~b and b~c: c is the twin of a twin of the evaluation talk a.
    talks = [record(talk_id, f"Title {talk_id}", 1000) for talk_id in "abc"]
    classes = duplicates.duplicate_classes(talks, domain="none", pairs=[("a", "b"), ("b", "c")])
    left_out = duplicates.excluded_talks(classes, {"a"})

    assert classes["a"] == classes["b"] == classes["c"], "the class is closed transitively"
    assert left_out == {
        "a": "evaluation talk",
        "b": "duplicate of an evaluation talk",
        "c": "duplicate of an evaluation talk",
    }


def test_a_title_link_and_a_confirmed_pair_are_one_chain():
    # c~b by title and length, b~a by confirmed pair: one class, id "a".
    talks = [
        record("a", "First title", 1000),
        record("b", "Second title", 1000),
        record("c", "Second title", 1001),
    ]

    classes = duplicates.duplicate_classes(talks, domain="none", pairs=[("a", "b")])

    assert classes == {"a": "a", "b": "a", "c": "a"}, (
        "the title link and the pair close into one class"
    )


def test_the_domain_filter_keeps_the_pairs_of_another_corpus_out(monkeypatch):
    # The default pairs are filtered by domain: a confirmed pair of the nasa
    # corpus must not merge two talks of the peertube corpus, even if the ids
    # of one corpus were the slugs of the other.
    monkeypatch.setattr(duplicates, "CONFIRMED_PAIRS", (("nasa", "one", "two"),))
    talks = [record("one", "First talk", 1000), record("two", "Second talk", 1000)]

    assert duplicates.duplicate_classes(talks, domain="peertube") == {
        "one": "one",
        "two": "two",
    }, "a pair of the nasa domain does not merge two talks of the peertube corpus"
    assert duplicates.duplicate_classes(talks, domain="nasa") == {
        "one": "one",
        "two": "one",
    }, "the pair applies in its own domain"


def test_a_title_chain_merges_over_more_than_the_tolerance():
    # 1000, 1001.5 and 1003 s: each within 2 s of its neighbour, the ends 3 s apart.
    talks = [
        record("a", "Same title", 1000),
        record("b", "Same title", 1001.5),
        record("c", "Same title", 1003),
    ]

    classes = duplicates.duplicate_classes(talks, domain="none")

    assert classes == {"a": "a", "b": "a", "c": "a"}, "the chain is one class"


def test_the_class_id_is_the_smallest_id_of_the_class():
    talks = [record("zzz", "Same talk", 1000), record("aaa", "Same talk", 1001)]

    classes = duplicates.duplicate_classes(talks, domain="none")

    assert classes == {"zzz": "aaa", "aaa": "aaa"}


def test_a_shared_author_or_channel_never_merges_two_talks():
    # A record holds no author, channel or host: the rule cannot use them, so
    # the same host or channel never puts two talks in the same class.
    talks = [record("one", "First talk", 1000), record("two", "Second talk", 1000)]

    classes = duplicates.duplicate_classes(talks, domain="none")

    assert classes == {"one": "one", "two": "two"}


def test_a_talk_record_may_be_an_object_with_the_same_attributes():
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Talk:
        id: str
        title: str
        duration_s: int

    talks = [
        Talk("eval", "Same talk", 1000),
        Talk("twin", "Same talk", 1000),
    ]

    classes = duplicates.duplicate_classes(talks, domain="none")

    assert classes == {"eval": "eval", "twin": "eval"}


# The committed data, read-only.


def test_the_peertube_population_is_one_record_per_id_over_the_three_listings():
    data = Path(duplicates.__file__).parent
    expected = set()
    for name in duplicates.PEERTUBE_LISTINGS:
        expected.update(record["id"] for record in duplicates.listing_records(data / name))

    population = duplicates.peertube_talks()
    ids = [record["id"] for record in population]

    assert len(ids) == len(set(ids)), "an id in two listings is the same talk"
    assert set(ids) == expected, "one record per id, no talk of a listing lost"


def test_every_confirmed_pair_holds_two_ids_of_its_own_listing():
    data = Path(duplicates.__file__).parent
    peertube_ids = {record["id"] for record in duplicates.peertube_talks()}
    nasa_ids = {record["id"] for record in duplicates.listing_records(data / "nasa_listing.json")}

    for domain, first, second in duplicates.CONFIRMED_PAIRS:
        assert first != second, f"{domain}: a pair holds two ids"
        population = peertube_ids if domain == "peertube" else nasa_ids
        assert first in population, f"{domain}: {first} is not in the listing"
        assert second in population, f"{domain}: {second} is not in the listing"
