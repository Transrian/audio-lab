"""The trap set: pairs the normalizer must join, and pairs it must keep apart."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402

from audiolab.eval.normalize import NORMALIZER_VERSION, VIEWS, normalize, token_count  # noqa: E402

TRAP_SET = Path(__file__).parent / "data" / "trap_set.jsonl"

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
}


def load_pairs() -> list[dict]:
    return [json.loads(line) for line in TRAP_SET.read_text(encoding="utf-8").splitlines()]


PAIRS = load_pairs()


def test_trap_set_is_large_and_balanced():
    assert len(PAIRS) >= 200
    equal = [pair for pair in PAIRS if pair["equal"]]
    assert len(equal) >= 120
    assert len(PAIRS) - len(equal) >= 25
    identifiers = [pair["id"] for pair in PAIRS]
    assert len(identifiers) == len(set(identifiers))
    assert {pair["language"] for pair in PAIRS} == {"fr", "en"}
    assert {pair["view"] for pair in PAIRS} <= set(VIEWS)


@pytest.mark.parametrize("pair", PAIRS, ids=[pair["id"] for pair in PAIRS])
def test_trap_set_pair(pair: dict):
    reference = normalize(pair["reference"], pair["language"], pair["view"])
    hypothesis = normalize(pair["hypothesis"], pair["language"], pair["view"])
    assert (reference == hypothesis) is pair["equal"], (
        f"{pair['id']}: {pair['reference']!r} -> {reference!r}, "
        f"{pair['hypothesis']!r} -> {hypothesis!r}"
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


def test_normalizing_twice_changes_nothing():
    for pair in PAIRS:
        once = normalize(pair["reference"], pair["language"], pair["view"])
        assert normalize(once, pair["language"], pair["view"]) == once


def test_token_count_matches_the_normalized_text():
    assert token_count("Bonjour, le monde !", "fr") == 3
    assert token_count("", "fr") == 0
    assert token_count("vingt-deux chats", "fr") == 3
