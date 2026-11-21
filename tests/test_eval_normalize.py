"""The trap set: pairs the normalizer must join, and pairs it must keep apart."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402

from audiolab.eval.metrics import counts  # noqa: E402
from audiolab.eval.normalize import (  # noqa: E402
    NORMALIZER_VERSION,
    STAND_IN_CHARACTERS,
    V1,
    VIEWS,
    normalize,
)

TRAP_SET = Path(__file__).parent / "data" / "normalizer_traps.jsonl"

# Version 1 output, replayed text by text: the trap set says which pairs compare
# equal, this file says what the frozen rule set writes for each text.
V1_GOLDEN = Path(__file__).parent / "data" / "normalizer_v1_golden.jsonl"

# What each equal pair is meant to prove. Only the pairs whose intent is not
# obvious from the text are listed; the file itself stays free of explanations.
RULES = {
    "fr-elision-one-token": "an elision is one token, so dropping the apostrophe splits the word",
    "en-contraction": "version 1 leaves a contraction alone; version 2 expands it",
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


# The version 2 rules, case by case: ``(id, language, reference, version 2
# output, version 1 output)``. The reference is the text the rule changes and the
# version 2 output is the equivalent canonical spelling, so the same list pins
# both halves of proposed decision 0013: what version 2 does, and what the frozen
# version 1 still does. ``strict`` and ``normalized`` agree on every case, which
# is why one output column is enough.
V2_CASES = [
    ("rule1-pronoun-s", "en", "it's a test", "it is a test", "it's a test"),
    ("rule1-table-cannot", "en", "can't", "cannot", "can't"),
    ("rule1-table-will-not", "en", "won't", "will not", "won't"),
    ("rule1-table-let-us", "en", "let's go", "let us go", "let's go"),
    ("rule1-suffix-ll", "en", "it'll work", "it will work", "it'll work"),
    ("rule1-suffix-ve", "en", "you've seen it", "you have seen it", "you've seen it"),
    ("rule2-s-has-been", "en", "it's been fine", "it has been fine", "it's been fine"),
    ("rule2-s-has-got", "en", "he's got one", "he has got one", "he's got one"),
    (
        "rule2-s-has-gotten",
        "en",
        "that's gotten worse",
        "that has gotten worse",
        "that's gotten worse",
    ),
    ("rule2-s-is-after-pronoun", "en", "that's it", "that is it", "that's it"),
    ("rule1-possessive-kept", "en", "john's car", "john's car", "john's car"),
    ("rule1-d-never-expanded", "en", "he'd say", "he'd say", "he'd say"),
    # The elision letters match whole tokens only, so the plural possessive is
    # untouched and loses its apostrophe for the ordinary reason the strict view
    # strips outer punctuation.
    ("rule3-plural-possessive-kept", "en", "students' books", "students books", "students books"),
    ("rule3-elision-j", "fr", "j' ai faim", "j'ai faim", "j ai faim"),
    ("rule3-elision-qu", "fr", "qu' il a dit", "qu'il a dit", "qu il a dit"),
    ("rule3-elision-jusqu", "fr", "jusqu' ici", "jusqu'ici", "jusqu ici"),
    ("rule3-elision-l", "fr", "l' homme", "l'homme", "l homme"),
    ("rule3-closed-elision-kept", "fr", "l'homme", "l'homme", "l'homme"),
    ("rule3-missing-apostrophe-kept", "fr", "l homme", "l homme", "l homme"),
    (
        "rules1-and-3-mixed-language",
        "fr",
        "j' ai dit it's d' accord",
        "j'ai dit it is d'accord",
        "j ai dit it's d accord",
    ),
]

# The typographic apostrophe is mapped to the ASCII one before a rule runs, so
# the same cases hold with a curly apostrophe. ``(id, language, text, expected)``.
CURLY_CASES = [
    ("rule2-s-has-been-curly", "en", "it\u2019s been fine", "it has been fine"),
    ("rule3-elision-curly", "fr", "d\u2019 accord", "d'accord"),
]


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
    """Every pair under the version it pins; a pair that pins none uses today's."""
    version = pair.get("version", NORMALIZER_VERSION)
    reference = normalize(pair["ref"], pair["language"], pair["view"], version)
    hypothesis = normalize(pair["hyp"], pair["language"], pair["view"], version)
    assert (reference == hypothesis) is pair["expect_equal"], (
        f"{pair['id']}: {pair['ref']!r} -> {reference!r}, {pair['hyp']!r} -> {hypothesis!r}"
    )


@pytest.mark.parametrize("pair", PAIRS, ids=[pair["id"] for pair in PAIRS])
def test_trap_set_pair_under_version_1(pair: dict):
    """The frozen rule set still answers every pair the way the trap set says.

    Two of them are spelled out again by a version 2 rule (the contraction pair
    and the elision spacing), which is why ``version`` in the file can pin a
    pair; under version 1 nothing has moved, so the file's own answer holds for
    every pair.
    """
    reference = normalize(pair["ref"], pair["language"], pair["view"], V1)
    hypothesis = normalize(pair["hyp"], pair["language"], pair["view"], V1)
    assert (reference == hypothesis) is pair["expect_equal"], (
        f"{pair['id']}: {pair['ref']!r} -> {reference!r}, {pair['hyp']!r} -> {hypothesis!r}"
    )


def load_golden() -> list[dict]:
    """The frozen version 1 outputs; the header line carries none."""
    rows = [json.loads(line) for line in V1_GOLDEN.read_text(encoding="utf-8").splitlines()]
    return [row for row in rows if "text" in row]


GOLDEN = load_golden()


@pytest.mark.parametrize("row", GOLDEN, ids=[f"{number:04d}" for number in range(len(GOLDEN))])
def test_version_1_output_is_frozen(row: dict):
    """Every text of the golden fixture still normalizes to the recorded output."""
    assert normalize(row["text"], row["language"], row["view"], V1) == row["output"], row


def test_the_golden_fixture_covers_all_three_views_and_both_languages():
    assert {row["view"] for row in GOLDEN} == set(VIEWS)
    assert {row["language"] for row in GOLDEN} == {"fr", "en"}
    assert len(GOLDEN) >= 900


def test_rule_notes_describe_real_pairs():
    assert set(RULES) <= {pair["id"] for pair in PAIRS}


def test_version_is_recorded():
    assert NORMALIZER_VERSION == 2
    assert V1 == 1
    assert VIEWS == ("normalized", "strict", "formatted")


def test_an_unknown_normalizer_version_is_refused():
    with pytest.raises(ValueError):
        normalize("bonjour", "fr", "normalized", 3)


def test_unknown_view_or_language_is_refused():
    with pytest.raises(ValueError):
        normalize("bonjour", "fr", "leetspeak")
    with pytest.raises(ValueError):
        normalize("bonjour", "de")


@pytest.mark.parametrize("case", V2_CASES, ids=[case[0] for case in V2_CASES])
def test_version_2_rules_with_the_frozen_version_1_output(case):
    """Every rule of proposed decision 0013, in both views it changes.

    The reference and its version 2 output are the two spellings a rule makes
    equivalent, so they compare equal under version 2 and different under
    version 1 — otherwise the rule flatters one side instead of both.
    """
    _, language, reference, version_2, version_1 = case
    for view in ("strict", "normalized"):
        assert normalize(reference, language, view, NORMALIZER_VERSION) == version_2, (
            reference,
            view,
        )
        assert normalize(reference, language, view, V1) == version_1, (reference, view)
        assert normalize(reference, language, view, NORMALIZER_VERSION) == normalize(
            version_2, language, view, NORMALIZER_VERSION
        ), (reference, version_2, view)


@pytest.mark.parametrize("case", V2_CASES, ids=[case[0] for case in V2_CASES])
def test_neither_version_rewrites_its_own_output(case):
    """A rule set read twice gives the same words, so a score cannot depend on that."""
    _, language, _, version_2, version_1 = case
    for view in ("strict", "normalized"):
        assert normalize(version_2, language, view, NORMALIZER_VERSION) == version_2
        assert normalize(version_1, language, view, V1) == version_1


def flipping_cases() -> list:
    """The version 2 cases whose two spellings version 1 keeps apart."""
    selected = []
    for case in V2_CASES:
        _, language, _, version_2, version_1 = case
        for view in ("strict", "normalized"):
            if normalize(version_2, language, view, V1) != version_1:
                selected.append(pytest.param(case, view, id=f"{case[0]}-{view}"))
    return selected


@pytest.mark.parametrize("case, view", flipping_cases())
def test_a_rule_joins_the_two_spellings_in_both_directions(case, view):
    """A rule that fires on one side only would flatter the side it rewrites.

    The reference and its version 2 output are scored against each other with
    the same function that scores a run, in both directions: under version 2
    they are the same words, under version 1 they are not, whichever side is
    the reference.
    """
    _, language, reference, version_2, _ = case
    for first, second in ((reference, version_2), (version_2, reference)):
        assert counts(first, second, language, view, NORMALIZER_VERSION).errors == 0, (
            first,
            second,
            view,
        )
        assert counts(first, second, language, view, V1).errors > 0, (first, second, view)


@pytest.mark.parametrize("case", CURLY_CASES, ids=[case[0] for case in CURLY_CASES])
def test_typographic_apostrophes_take_the_same_rules(case):
    _, language, text, expected = case
    assert normalize(text, language, "normalized") == expected
    assert normalize(text, language, "normalized", V1) != expected


def test_formatted_is_the_same_in_both_versions():
    """The formatted view keeps the written text, so no version 2 rule reaches it."""
    for _, language, *texts in V2_CASES:
        for text in texts:
            assert normalize(text, language, "formatted", NORMALIZER_VERSION) == normalize(
                text, language, "formatted", V1
            )


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
