"""The trap set: pairs the normalizer must join, and pairs it must keep apart."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402

from audiolab.eval.normalize import NORMALIZER_VERSION, VIEWS, normalize  # noqa: E402

TRAP_SET = Path(__file__).parent / "data" / "normalizer_traps.jsonl"

# What each equal pair is meant to prove. Only the pairs whose intent is not
# obvious from the text are listed; the file itself stays free of explanations.
RULES = {
    "fr-elision-one-token": "an elision is one token, so dropping the apostrophe splits the word",
    "en-contraction": "contractions are not expanded",
    "strict-percent-spacing": "the strict view does not verbalize units",
    "strict-hyphen": "the strict view keeps a hyphen inside a word",
    "fr-only-filler": "a filler word alone normalizes to nothing",
    "en-homophone-their": "homophones are not folded",
    "formatted-comma-kept": "the formatted view keeps punctuation",
    "fr-dash-range": "a dash between two numbers reads as a space",
    "en-decimal-french-style": "either decimal mark is read as a decimal",
    "fr-plural-kept": "number spelling with a plural s is not a number here",
    "enmore-1234": "a four digit number in the year range reads as a year pair",
    "fr-mister": '"m" is not expanded to "monsieur": it is also the metre symbol',
    "strict-thousands-en": "in the strict view a comma between digits is a decimal mark",
    "m4-decimal-leading-zero": "3,05 and 3,5 are not the same number",
    "m4-en-and-between-numbers": 'an "and" between plain numbers is a real word',
    "m4-filler-ben-kept": '"ben" is a name as often as a hesitation, so it is kept',
    "m4-en-year-round": "a year ending in 00 keeps its cardinal reading",
    "m4-metre-not-monsieur": "a metre is not a person",
}


def load_pairs() -> list[dict]:
    """The pairs of the trap set; the header line carries no pair."""
    rows = [json.loads(line) for line in TRAP_SET.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if "id" in row]


PAIRS = load_pairs()


def test_trap_set_header_says_it_is_unreviewed():
    header = json.loads(TRAP_SET.read_text(encoding="utf-8").splitlines()[0])
    assert "not yet reviewed by the owner" in header["note"]


def test_trap_set_is_large_and_balanced():
    assert len(PAIRS) >= 200
    equal = [pair for pair in PAIRS if pair["expect_equal"]]
    assert len(equal) >= 120
    assert len(PAIRS) - len(equal) >= 25
    identifiers = [pair["id"] for pair in PAIRS]
    assert len(identifiers) == len(set(identifiers))
    assert {pair["language"] for pair in PAIRS} == {"fr", "en"}
    assert {pair["view"] for pair in PAIRS} <= set(VIEWS)


@pytest.mark.parametrize("pair", PAIRS, ids=[pair["id"] for pair in PAIRS])
def test_trap_set_pair(pair: dict):
    reference = normalize(pair["ref"], pair["language"], pair["view"])
    hypothesis = normalize(pair["hyp"], pair["language"], pair["view"])
    assert (reference == hypothesis) is pair["expect_equal"], (
        f"{pair['id']}: {pair['ref']!r} -> {reference!r}, {pair['hyp']!r} -> {hypothesis!r}"
    )


def test_rule_notes_describe_real_pairs():
    assert set(RULES) <= {pair["id"] for pair in PAIRS}


def test_version_is_recorded():
    assert NORMALIZER_VERSION == 1
    assert VIEWS == ("normalized", "strict", "formatted")


def test_unknown_view_or_language_is_refused():
    with pytest.raises(ValueError):
        normalize("bonjour", "fr", "leetspeak")
    with pytest.raises(ValueError):
        normalize("bonjour", "de")


def test_a_normalized_text_stays_as_it_is():
    """Normalizing a normalized text changes nothing, with one documented exception.

    "2.000" is ambiguous once the punctuation is gone: it is the strict form of
    an English "2,000" *and* a dot-grouped 2000. Reading it again takes the
    grouped reading, so that one string does not round-trip; nothing else does.
    """
    for pair in PAIRS:
        once = normalize(pair["ref"], pair["language"], pair["view"])
        if once == "2.000":
            assert normalize(once, pair["language"], "strict") == "2000"
            continue
        assert normalize(once, pair["language"], pair["view"]) == once


def test_no_token_of_the_trap_set_keeps_a_punctuation_character():
    """Tokens hold letters, digits, ``'``/``-`` and the unit symbols, nothing else.

    This is the property the punctuation fix is about: a quote, bracket or slash
    left inside a token would be scored as a word of its own.
    """
    # The canonical decimal point and the clock colon are part of a number.
    allowed = set("'-%€$£°&.:")
    for pair in PAIRS:
        for view in ("strict", "normalized"):
            for token in normalize(pair["ref"], pair["language"], view).split():
                extra = {
                    character
                    for character in token
                    if not character.isalnum() and character not in allowed
                }
                assert not extra, f"{pair['id']} leaves {extra} in {token!r}"
