"""Text normalization for scoring (three views, one version).

Model output and reference texts never match character for character: one
writes "20 %", the other "vingt pour cent", one keeps the case, the other does
not. Every view is a pure function applied to both sides, so a difference in a
score always comes from a difference in text, never from a difference in
processing.

Three views (see `docs/stt/evaluation.md` for examples):

- ``strict``: case-folded, punctuation removed, apostrophes and hyphens kept
  inside words, so French elision ``l'homme`` stays one token.
- ``formatted``: case and punctuation kept, with punctuation split off words;
  meaningful only for references that have casing and punctuation.
- ``normalized``: ``strict`` plus number verbalization, abbreviation and
  filler-word handling: the headline number.

Changing a rule changes scores, so the version is bumped and recorded in every
summary (``NORMALIZER_VERSION``).
"""

from __future__ import annotations

import re
import unicodedata

NORMALIZER_VERSION = 1
VIEWS = ("normalized", "strict", "formatted")

# Typographic characters that mean the same thing as an ASCII one.
TYPOGRAPHIC = {
    "\u2018": "'",  # left single quotation mark
    "\u2019": "'",  # right single quotation mark (the usual French apostrophe)
    "\u02bc": "'",  # modifier letter apostrophe
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\u2012": "-",  # figure dash
    "\u2013": "-",  # en dash
    "\u2014": " ",  # em dash: a separator, not a word part
    "\u2212": "-",  # minus sign
    "\u00a0": " ",  # no-break space
    "\u202f": " ",  # narrow no-break space
    "\u2026": "...",
    "\u00ab": " ",
    "\u00bb": " ",
}

SENTENCE_PUNCTUATION = ".,?!;:"
KEPT_IN_WORD = "'-"
# Stand-ins for a comma or point between two digits, so tokenizing does not
# mistake a decimal mark for a sentence separator.
DECIMAL_COMMA = "\x01"
DECIMAL_POINT = "\x02"
CLOCK_COLON = "\x03"  # stands in for the colon of a time, as in 14:30
UNIT_CHARS = "%€$£°"

FILLERS = {"fr": {"euh", "heu", "hum", "hmm", "ben", "bah"}, "en": {"uh", "um", "erm", "hmm", "er"}}

# Only unambiguous strings: "sept" is the number seven as often as it is
# September, so month names are left alone. Dots are already gone by this point,
# which is why dotted abbreviations are matched without them.
ABBREVIATIONS = {
    "fr": {
        "m": "monsieur",
        "mm": "messieurs",
        "mme": "madame",
        "dr": "docteur",
        "st": "saint",
        "ste": "sainte",
        "km": "kilomètres",
        "kg": "kilogrammes",
        "bd": "boulevard",
        "h": "heures",
        "&": "et",
    },
    "en": {
        "mr": "mister",
        "mrs": "missus",
        "ms": "miss",
        "dr": "doctor",
        "st": "saint",
        "km": "kilometers",
        "kg": "kilograms",
        "hr": "hours",
        "vs": "versus",
        "&": "and",
    },
}

# Small on purpose: both spellings are correct, and a large list would hide
# real spelling errors instead of measuring them (decision 0002 wants a fixed,
# reviewable rule set).
SPELLING = {
    "colour": "color",
    "colours": "colors",
    "favourite": "favorite",
    "honour": "honor",
    "labour": "labor",
    "neighbour": "neighbor",
    "behaviour": "behavior",
    "centre": "center",
    "theatre": "theater",
    "metre": "meter",
    "litre": "liter",
    "organise": "organize",
    "organised": "organized",
    "realise": "realize",
    "realised": "realized",
    "recognise": "recognize",
    "recognised": "recognized",
    "analyse": "analyze",
    "travelling": "traveling",
    "cancelled": "canceled",
    "programme": "program",
    "grey": "gray",
    "plough": "plow",
    "defence": "defense",
    "licence": "license",
    "practise": "practice",
    "aluminium": "aluminum",
}

FR_UNITS = (
    "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze",
    "douze", "treize", "quatorze", "quinze", "seize", "dix-sept", "dix-huit", "dix-neuf",
)  # fmt: skip
FR_TENS = ("", "", "vingt", "trente", "quarante", "cinquante", "soixante")
EN_UNITS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
)  # fmt: skip
EN_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
EN_NUMBER_WORDS = (
    frozenset(EN_UNITS[:-1])
    | frozenset(EN_TENS)
    | {
        "hundred",
        "thousand",
        "million",
        "billion",
        "point",
    }
)

# Number-with-unit patterns. The unit is verbalized as a plain word so that
# "20 %" and "vingt pour cent" score equal.
FR_UNITS_WORDS = {"%": "pour cent", "€": "euros", "$": "dollars", "£": "livres", "°C": "degrés"}
EN_UNITS_WORDS = {"%": "percent", "€": "euros", "$": "dollars", "£": "pounds", "°C": "degrees"}
# Symbols carry no letters, so tokenizing keeps them only by looking them up here.
UNIT_SYMBOLS = frozenset(FR_UNITS_WORDS) | frozenset(EN_UNITS_WORDS)
# Symbols with no letters that still carry a word.
SYMBOL_WORDS = frozenset({"&"})


def normalize(text: str, language: str, view: str = "normalized") -> str:
    """Normalize ``text`` for ``language`` in one of the three views."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; known views: {', '.join(VIEWS)}")
    prepared = _prepare(text)
    if view == "formatted":
        return _punctuation_view(prepared)
    tokens = _strict_tokens(prepared)
    if view == "strict":
        return " ".join(tokens)
    return " ".join(_normalized_tokens(tokens, language))


def token_count(text: str, language: str, view: str = "normalized") -> int:
    """How many tokens the text has in a view (0 for an empty text)."""
    return len(normalize(text, language, view).split())


def _prepare(text: str) -> str:
    """One representation before tokenizing: NFC, ASCII punctuation, collapsed spaces."""
    folded = unicodedata.normalize("NFC", text)
    for typographic, plain in TYPOGRAPHIC.items():
        folded = folded.replace(typographic, plain)
    return re.sub(r"\s+", " ", folded).strip()


def _punctuation_view(text: str) -> str:
    """Keep the case, split sentence punctuation off words, keep word-internal marks."""
    spaced = re.sub(rf"([{re.escape(SENTENCE_PUNCTUATION)}])", r" \1 ", text)
    return re.sub(r"\s+", " ", spaced).strip()


def _strict_tokens(text: str) -> list[str]:
    """Case-folded tokens without punctuation; ``'`` and ``-`` stay inside words."""
    lowered = text.casefold()
    # A comma or point between two digits is a decimal mark, not punctuation:
    # hiding it keeps "3,5" one token instead of splitting it into 3 and 5.
    protected = re.sub(r"(?<=\d),(?=\d)", DECIMAL_COMMA, lowered)
    protected = re.sub(r"(?<=\d)\.(?=\d)", DECIMAL_POINT, protected)
    protected = re.sub(r"(?<=\d):(?=\d{2}\D|\d{2}$)", CLOCK_COLON, protected)
    tokens: list[str] = []
    for chunk in re.findall(rf"[^\s{re.escape(SENTENCE_PUNCTUATION)}]+", protected):
        word = _trim(chunk)
        word = _restore_marks(word)
        spelled = word.upper() in UNIT_SYMBOLS or word in SYMBOL_WORDS
        if word and (any(character.isalnum() for character in word) or spelled):
            tokens.append(word)
    return tokens


def _trim(word: str) -> str:
    """Remove outer punctuation, but keep the sign of a number: ``-3`` is not ``3``."""
    trimmed = word.strip("'")
    if not re.fullmatch(r"[+-]\d.*", trimmed):
        trimmed = trimmed.strip("-")
    return trimmed.strip("'")


def _restore_marks(word: str) -> str:
    """Put the protected marks back, canonicalizing a number on the way."""
    if CLOCK_COLON in word:
        return word.replace(CLOCK_COLON, ":")
    if DECIMAL_COMMA in word or DECIMAL_POINT in word:
        return _canonical_number(word)
    return word


def _canonical_number(word: str) -> str:
    """Drop thousands separators, write a decimal with a point.

    Both conventions then compare equal: ``2.000`` and ``2,000`` are the same
    number, and a model that writes ``3.5`` matches a reference that writes
    ``3,5``. A group of exactly three digits is a thousands group, anything
    else is a decimal.
    """
    restored = word.replace(DECIMAL_POINT, ".").replace(DECIMAL_COMMA, ",")
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", restored):
        return re.sub(r"[.,]", "", restored)
    if re.fullmatch(r"[+-]?\d+[.,]\d+", restored):
        return restored.replace(",", ".")
    return restored


def _normalized_tokens(tokens: list[str], language: str) -> list[str]:
    """Verbalize numbers and units, expand abbreviations, drop fillers."""
    if language not in ("fr", "en"):
        raise ValueError(f"no normalizer for language {language!r}")
    result: list[str] = []
    merged = _merge_number_groups(tokens)
    index = 0
    while index < len(merged):
        number = merged[index]
        symbol = merged[index + 1] if index + 1 < len(merged) else ""
        if symbol.upper() in UNIT_SYMBOLS and re.fullmatch(r"[+-]?\d[\d .,]*", number):
            result += _measure_words(number, symbol, language).split()
            index += 2
            continue
        result += _spoken_words(number, language)
        index += 1
    return _drop_number_connectors(result) if language == "en" else result


def _spoken_words(token: str, language: str) -> list[str]:
    """The spoken words of one token, without filler words."""
    return [
        spelled
        for word in _normalize_token(token, language)
        if word and word not in FILLERS[language]
        for spelled in word.replace("-", " ").split()
    ]


def _drop_number_connectors(words: list[str]) -> list[str]:
    """Drop the "and" of "one hundred and one", keeping the one in "salt and pepper"."""
    kept: list[str] = []
    for index, word in enumerate(words):
        number_and = (
            word == "and"
            and kept
            and kept[-1] in EN_NUMBER_WORDS
            and index + 1 < len(words)
            and words[index + 1] in EN_NUMBER_WORDS
        )
        if not number_and:
            kept.append(SPELLING.get(word, word))
    return kept


def _merge_number_groups(tokens: list[str]) -> list[str]:
    """Join digit groups separated by a space into one number.

    French writes thousands with a space (``1 234``). The rule is deliberately
    narrow: only a short leading group followed by exactly three digits merges,
    so "en 2021 45" stays two numbers.
    """
    merged: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if re.fullmatch(r"[+-]?\d{1,3}", token) and index + 1 < len(tokens):
            follower = tokens[index + 1]
            if re.fullmatch(r"\d{3}", follower):
                merged.append(token + " " + follower)
                index += 2
                continue
        merged.append(token)
        index += 1
    return merged


def _normalize_token(token: str, language: str) -> list[str]:
    """The words one token contributes, in order."""
    word = _trim(token)
    sign = _sign_words(word, language)
    body = word.lstrip("+-")
    grouped = re.fullmatch(r"(\d{1,3}) (\d{3})", body)
    if grouped:
        return (sign + _verbalize_number(int(body.replace(" ", "")), language)).split()
    if re.fullmatch(r"\d+", body):
        return (sign + _verbalize_number(int(body), language)).split()
    decimal = re.fullmatch(r"\d+[.,]\d+", body)
    if decimal:
        return (sign + _number_words(body, language)).split()
    if re.fullmatch(r"1(?:er|re|ère|e)", word):
        return [{"fr": "premier", "en": "first"}[language]]
    ordinal = re.fullmatch(r"(\d+)(?:e|er|re|ème|eme|th|st|nd|rd)", word)
    if ordinal:
        return [_ordinal(int(ordinal.group(1)), language)]
    clock = re.fullmatch(r"(\d{1,2})(?:h|:)(\d{2})?", word)
    if clock:
        return _clock_words(int(clock.group(1)), clock.group(2), language)
    attached = _attached_unit(word, language)
    if attached:
        return attached
    if word in ABBREVIATIONS[language]:
        return ABBREVIATIONS[language][word].split()
    if "-" in word:
        return [
            part for piece in word.split("-") if piece for part in _normalize_token(piece, language)
        ]
    return [word]


def _sign_words(word: str, language: str) -> str:
    """The spoken sign of a number: ``-3`` is "moins trois", ``+3`` "plus trois"."""
    if word.startswith("-"):
        return "moins " if language == "fr" else "minus "
    return "plus " if word.startswith("+") else ""


def _measure_words(number: str, symbol: str, language: str) -> str:
    """A quantity with its unit.

    Money reads the way it is said: "douze euros cinquante". Other units keep
    the decimal reading: "deux virgule cinq kilogrammes".
    """
    units = FR_UNITS_WORDS if language == "fr" else EN_UNITS_WORDS
    unit = units[symbol.upper()]
    money = re.fullmatch(r"([+-]?)(\d+)[.,](\d{2})", number)
    if money and symbol in "€$£":
        sign = _sign_words(number, language)
        whole = _verbalize_number(int(money.group(2)), language)
        cents = _verbalize_number(int(money.group(3)), language)
        return f"{sign}{whole} {unit} {cents}"
    return f"{_number_words(number, language)} {unit}"


def _without_spaces(word: str) -> str:
    return word.replace(" ", "")


def _number_words(text: str, language: str) -> str:
    """Read a number: ``3.5`` is "trois virgule cinq" in French, "three point five" in English."""
    cleaned = _without_spaces(text)
    sign = _sign_words(cleaned, language)
    body = cleaned.lstrip("+-")
    separator = "virgule" if language == "fr" else "point"
    if re.fullmatch(r"\d+[.,]\d+", body):
        whole, fraction = re.split(r"[.,]", body, maxsplit=1)
        whole_words = _verbalize_number(int(whole), language)
        fraction_words = _verbalize_number(int(fraction), language)
        return f"{sign}{whole_words} {separator} {fraction_words}"
    return sign + _verbalize_number(int(body), language)


def _verbalize_number(value: int, language: str) -> str:
    """The cardinal in words for the language of the text."""
    return _french_number(value) if language == "fr" else _english_number(value)


def _french_number(value: int) -> str:
    """Cardinal in words, using the hyphens only where the spelling needs them."""
    if value < 0:
        return "moins " + _french_number(-value)
    if value < 20:
        return FR_UNITS[value]
    if value < 70:
        tens, unit = divmod(value, 10)
        suffix = "-et-un" if unit == 1 else f"-{FR_UNITS[unit]}" if unit else ""
        return FR_TENS[tens] + suffix
    if value < 80:
        unit = value - 60
        return "soixante" + ("-et-onze" if unit == 11 else f"-{FR_UNITS[unit]}")
    if value < 100:
        unit = value - 80
        return "quatre-vingt" + ("s" if not unit else f"-{FR_UNITS[unit]}")
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        prefix = "cent" if hundreds == 1 else f"{FR_UNITS[hundreds]} cent"
        if rest == 0:
            return prefix + ("s" if hundreds > 1 else "")
        return f"{prefix} {_french_number(rest)}"
    for scale, singular, plural in (
        (10**9, "milliard", "milliards"),
        (10**6, "million", "millions"),
    ):
        if value >= scale:
            count, rest = divmod(value, scale)
            words = f"{_french_number(count)} {singular if count == 1 else plural}"
            return words if rest == 0 else f"{words} {_french_number(rest)}"
    thousands, rest = divmod(value, 1000)
    prefix = "mille" if thousands == 1 else f"{_french_number(thousands)} mille"
    return prefix if rest == 0 else f"{prefix} {_french_number(rest)}"


def _english_number(value: int) -> str:
    """Cardinal in words, hyphens replaced by spaces so both spellings match."""
    if value < 0:
        return "minus " + _english_number(-value)
    if value < 20:
        return EN_UNITS[value]
    if value < 100:
        tens, unit = divmod(value, 10)
        return EN_TENS[tens] + (f" {EN_UNITS[unit]}" if unit else "")
    if value < 1000:
        hundreds, rest = divmod(value, 100)
        prefix = f"{EN_UNITS[hundreds]} hundred"
        return prefix if rest == 0 else f"{prefix} {_english_number(rest)}"
    for scale, word in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if value >= scale:
            count, rest = divmod(value, scale)
            prefix = f"{_english_number(count)} {word}"
            return prefix if rest == 0 else f"{prefix} {_english_number(rest)}"
    return EN_UNITS[value]


def _ordinal(value: int, language: str) -> str:
    if language == "fr":
        words = _french_number(value)
        if words.endswith("cinq"):
            return words + "uième"
        if words.endswith("neuf"):
            return words + "ième"
        trimmed = words[:-1] if words.endswith("e") else words
        return trimmed + "ième"
    irregular = {
        1: "first",
        2: "second",
        3: "third",
        5: "fifth",
        8: "eighth",
        9: "ninth",
        12: "twelfth",
    }
    if value < 20:
        if value in irregular:
            return irregular[value]
        return _english_number(value) + "th"
    if value >= 100:
        scale = 1000 if value >= 1000 else 100
        count, rest = divmod(value, scale)
        prefix = f"{_english_number(count)} {'thousand' if scale == 1000 else 'hundred'}"
        return prefix + "th" if not rest else f"{prefix} {_ordinal(rest, language)}"
    tens, unit = divmod(value, 10)
    word = EN_TENS[tens]
    if not unit:
        return (word[:-1] + "ie" if word.endswith("y") else word) + "th"
    return word + " " + _ordinal(unit, language)


def _attached_unit(word: str, language: str) -> list[str] | None:
    """A unit written against the number: ``21%``, ``20°C``, ``$3.50``."""
    number = re.fullmatch(r"([$€£])([+-]?\d[\d.,]*)", word)
    if number:
        return _measure_words(number.group(2), number.group(1), language).split()
    attached = re.fullmatch(rf"([+-]?\d[\d.,]*)([{re.escape(UNIT_CHARS)}]|°c)", word)
    if attached:
        return _measure_words(attached.group(1), attached.group(2), language).split()
    return None


def _clock_words(hour: int, minutes: str | None, language: str) -> list[str]:
    """``14h30`` and ``14:30`` both read as "quatorze heures trente"."""
    whole = minutes is None or minutes == "00"
    if language == "fr":
        words = f"{_verbalize_number(hour, language)} heures"
    else:
        words = (
            f"{_verbalize_number(hour, language)} o'clock"
            if whole
            else _verbalize_number(hour, language)
        )
    if whole:
        return words.split()
    return f"{words} {_verbalize_number(int(minutes), language)}".split()
