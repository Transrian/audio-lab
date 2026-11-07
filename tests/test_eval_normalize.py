"""The trap set: pairs the normalizer must join, and pairs it must keep apart."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402

from audiolab.eval.normalize import (  # noqa: E402
    NORMALIZER_VERSION,
    STAND_IN_CHARACTERS,
    VIEWS,
    normalize,
)

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
    "strict-thousands-en": "the strict view groups thousands with an English comma too",
    "n1-en-decimal-not-thousands": "a comma that does not group three digits is the decimal mark",
    "n1-fr-comma-is-decimal": "the same string in French is one and a half, not fifteen",
    "n1-fr-cent-mille-plural-refused": "cent loses its plural s before mille",
    "n1-en-year-measurement": "a quantity is not a year",
    "n1-en-year-2000s-oh-not-folded": 'the spoken "twenty oh five" is not folded',
    "n1-en-round-hundred-not-a-year": "a year ending in 00 keeps its cardinal reading",
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
    """Normalizing a normalized text changes nothing.

    This used to have one exception, the English "2.000" that stands for both a
    comma-grouped 2000 and a decimal with three digits. Since a mark that groups
    three digits is read as a thousands separator in either language, the strict
    view of "2,000" is "2000" and the exception is gone.
    """
    for pair in PAIRS:
        once = normalize(pair["ref"], pair["language"], pair["view"])
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


# A transcript is not a controlled input: whatever a model answers has to come
# back as a score, never as a traceback. These two tests pin that, on the shapes
# the verifier of this phase found and on random text over the same alphabet.
FUZZ_SEED = 20261107
FUZZ_CASES = 3000
FUZZ_ALPHABET = list("0123456789,.$€£%°h:/-") + list("abcdefghijklmnopqrstuvwxyz") + [" "]
# Fragments a transcript really contains, so the corpus is not only noise.
FUZZ_WORDS = [
    "bonjour",
    "le",
    "chat",
    "de",
    "euh",
    "l'homme",
    "90",
    "1,500",
    "3.5",
    "21",
    "%",
    "$",
    "€",
    "£",
    "°C",
    "1 000",
    "14:30",
]


def _random_text(rng) -> str:
    """One plausible or implausible line: noise, words, or a written number."""
    chance = rng.random()
    if chance < 0.45:
        return "".join(rng.choice(FUZZ_ALPHABET) for _ in range(rng.randint(1, 14)))
    if chance < 0.8:
        return " ".join(rng.choice(FUZZ_WORDS) for _ in range(rng.randint(1, 8)))
    digits = "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 8)))
    separator = rng.choice([",", ".", " ", ", ", "."])
    grouped = separator.join(digits[index : index + 3] for index in range(0, len(digits), 3))
    return (
        rng.choice(["", "$", "€", "£", "-", "+"])
        + grouped
        + rng.choice(["", "%", "°", "°C", " h", ":30", "€"])
    )


def _fuzz_corpus():
    import random

    rng = random.Random(FUZZ_SEED)
    return [_random_text(rng) for _ in range(FUZZ_CASES)]


def test_normalize_never_raises_on_what_a_model_may_answer():
    """Every view of both languages answers something for any input.

    The corpus is seeded, so a failure is reproducible from ``FUZZ_SEED`` alone.
    """
    for text in _fuzz_corpus():
        for language in ("fr", "en"):
            for view in VIEWS:
                assert isinstance(normalize(text, language, view), str)


def test_normalizing_what_was_already_normalized_changes_nothing():
    """The normalized view is idempotent on the same corpus.

    Two passes must agree, otherwise the score depends on how many times the text
    went through the normalizer.
    """
    for text in _fuzz_corpus():
        for language in ("fr", "en"):
            once = normalize(text, language, "normalized")
            assert normalize(once, language, "normalized") == once


def test_no_output_of_the_fuzz_corpus_keeps_a_stand_in_character():
    """The tokenizer's own stand-in characters never reach a reader.

    A stand-in leaking out is a real defect: it is invisible in a report and it
    compares unequal to the same text written plainly.
    """
    for text in _fuzz_corpus():
        for language in ("fr", "en"):
            for view in VIEWS:
                written = normalize(text, language, view)
                for stand_in in STAND_IN_CHARACTERS:
                    assert stand_in not in written, f"{text!r} -> {written!r}"


def test_a_number_a_model_writes_the_way_no_rule_expects_is_kept_as_it_is():
    """What the reader cannot read stays literal instead of stopping the run."""
    # These four raised before this was fixed: two crashes in the number reader
    # and two symbols the unit table did not know.
    for text, language in [
        ("$1,500,000", "en"),
        ("1,500,000%", "en"),
        ("a 90° angle", "en"),
        ("36°", "fr"),
        ("-5°", "en"),
        ("1,500.00", "en"),
    ]:
        for view in VIEWS:
            assert isinstance(normalize(text, language, view), str)
    assert normalize("$1,500,000", "en", "normalized") == (
        "one million five hundred thousand dollars"
    )
    assert normalize("a 90° angle", "en", "normalized") == "a ninety degrees angle"
    assert normalize("36°", "fr", "normalized") == "trente six degrés"


def test_the_number_reader_stops_at_the_invented_number():
    """A number no utterance can contain is read digit by digit, not as a cardinal.

    Reading a cardinal costs one step per group of nine digits, so an invented
    number of a few hundred digits would exhaust the stack. The length here is
    far past the limit the module documents (thirty digits) and short of the
    recursion limit, which is what makes the difference visible.
    """
    spoken = normalize("9" * 230, "en", "normalized")
    assert spoken.split()[:3] == ["nine", "nine", "nine"]
