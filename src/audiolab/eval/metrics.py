"""What a panel's numbers mean, computed from scored items.

Every number here is a pure function of the items, so scoring the same records
twice gives the same summary. Two kinds of panel: speech panels measure how
wrong the transcript is, non-speech panels measure whether the model invents
text where there is nothing to hear.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from audiolab.eval.normalize import normalize

# A reference of this many words or more can be judged truncated.
TRUNCATION_MIN_WORDS = 5
# A hypothesis this much shorter than the reference looks cut off.
TRUNCATION_SHARE = 0.8
# Tail thresholds: "how bad is the worst tenth" and "how many are unusable".
TAIL_QUANTILES = (0.90, 0.95)
TAIL_SHARES = (0.25, 0.50)
# A hypothesis this long on non-speech is more than a stray token.
HALLUCINATION_LONG_WORDS = 5
LOOP_NGRAM_MAX = 4
LOOP_REPEATS = 3
# How many repeated hypotheses a report shows; they are public non-speech clips.
TOP_HYPOTHESES = 10


@dataclass(frozen=True)
class Item:
    """One scored unit: a reference, what the model said, and how it went."""

    id: str
    panel: str
    group: str
    category: str
    language: str
    reference: str
    hypothesis: str = ""
    error: str = ""
    latency_s: float = 0.0
    duration_s: float = 0.0
    cold: bool = False
    text_has_casing_punct: bool = False
    attempts: int = 1  # requests sent for this item, retries included

    @property
    def failed(self) -> bool:
        return bool(self.error)


@dataclass(frozen=True)
class Counts:
    """S/D/I/C counts of one hypothesis against its reference."""

    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.reference_words if self.reference_words else 0.0


def tokens(text: str, language: str, view: str) -> list[str]:
    """The tokens a text contributes to one view."""
    return normalize(text, language_for(language), view).split()


def view_for(language: str) -> str:
    """The view a text is scored in: the normalized one, for every item.

    The non-speech battery has no language of its own, so its words are
    normalized with the English rules: the question it asks is only whether the
    model said anything, and one view for all items means a word count in a
    report always means the same thing.
    """
    return "normalized"


def language_for(language: str) -> str:
    """The rules a language is normalized with; "none" falls back to English."""
    return language if language in ("fr", "en") else "en"


def counts(reference: str, hypothesis: str, language: str, view: str) -> Counts:
    """S/D/I/C between two texts in one view."""
    return edit_counts(tokens(reference, language, view), tokens(hypothesis, language, view))


def edit_counts(reference: list[str], hypothesis: list[str]) -> Counts:
    """Align two token lists and count the edits.

    The alignment is a plain Levenshtein table; where several alignments cost
    the same, substitutions win over deletions and insertions, so the S/D/I
    split is stable. The total is not affected by that choice.
    """
    rows, columns = len(reference), len(hypothesis)
    table = [[0] * (columns + 1) for _ in range(rows + 1)]
    for row in range(rows + 1):
        table[row][0] = row
    for column in range(columns + 1):
        table[0][column] = column
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            same = reference[row - 1] == hypothesis[column - 1]
            table[row][column] = min(
                table[row - 1][column - 1] + (0 if same else 1),
                table[row - 1][column] + 1,
                table[row][column - 1] + 1,
            )
    substitutions = deletions = insertions = 0
    row, column = rows, columns
    while row or column:
        same = row and column and reference[row - 1] == hypothesis[column - 1]
        if same and table[row][column] == table[row - 1][column - 1]:
            row, column = row - 1, column - 1
        elif row and column and table[row][column] == table[row - 1][column - 1] + 1:
            substitutions, row, column = substitutions + 1, row - 1, column - 1
        elif row and table[row][column] == table[row - 1][column] + 1:
            deletions, row = deletions + 1, row - 1
        else:
            insertions, column = insertions + 1, column - 1
    return Counts(substitutions, deletions, insertions, rows)


@dataclass(frozen=True)
class UtteranceScore:
    """One item scored in every view it supports."""

    item: Item
    counts: dict[str, Counts]

    def words(self, text: str) -> list[str]:
        """Tokenize the reference or the hypothesis for their language."""
        return tokens(text, self.item.language, view_for(self.item.language))

    @property
    def hypothesis_words(self) -> list[str]:
        return self.words(self.item.hypothesis)

    @property
    def reference_words(self) -> list[str]:
        return self.words(self.item.reference)


def totals_by_group(scored: list[UtteranceScore], value) -> dict[str, tuple[float, float]]:
    """Per group sums of ``value(score)``, which the bootstrap resamples."""
    totals: dict[str, list[float]] = {}
    for score in scored:
        numerator, denominator = value(score)
        running = totals.setdefault(score.item.group, [0.0, 0.0])
        running[0] += numerator
        running[1] += denominator
    return {group: (pair[0], pair[1]) for group, pair in totals.items()}


def wer_value(view: str):
    """Per item (errors, reference words) for one view."""

    def value(score: UtteranceScore) -> tuple[float, float]:
        counts_ = score.counts.get(view)
        return (float(counts_.errors), float(counts_.reference_words)) if counts_ else (0.0, 0.0)

    return value


def hallucination_value(score: UtteranceScore) -> tuple[float, float]:
    """Per item (invented?, 1) for a non-speech panel."""
    return (1.0 if score.hypothesis_words else 0.0, 1.0)


def score_items(items: list[Item]) -> list[UtteranceScore]:
    """Score every item in the views it supports; failed requests are skipped."""
    scored: list[UtteranceScore] = []
    for item in items:
        if item.failed:
            continue
        scored.append(UtteranceScore(item, _counts_for(item)))
    return scored


def _counts_for(item: Item) -> dict[str, Counts]:
    """The views one item can be scored in: formatted needs a formatted reference."""
    views = ["normalized", "strict"] if item.language in ("fr", "en") else ["normalized"]
    if item.text_has_casing_punct:
        views.append("formatted")
    return {view: counts(item.reference, item.hypothesis, item.language, view) for view in views}


def speech_summary(items: list[Item]) -> dict:
    """Every speech number for one panel."""
    scored = score_items(items)
    failed = [item for item in items if item.failed]
    summary: dict = {
        "kind": "speech",
        "items": len(items),
        "scored_items": len(scored),
        "groups": len({item.group for item in items}),
        "failed_requests": len(failed),
        "wer": {view: _view_wer(scored, view) for view in ("normalized", "strict", "formatted")},
        "sentence_error_rate": _share(scored, _is_wrong_sentence),
        "tail": _tail(scored),
        "loop_rate": _share(scored, _has_loop_utterance),
        "truncation_rate": _share(scored, _is_truncated),
        "empty_output_rate": _share(scored, lambda score: not score.hypothesis_words),
        "latency": _latency(items),
    }
    return summary


def nonspeech_summary(items: list[Item]) -> dict:
    """Hallucination numbers for one non-speech panel."""
    scored = score_items(items)
    failed = [item for item in items if item.failed]
    word_counts = [len(score.hypothesis_words) for score in scored]
    hypotheses = Counter(
        " ".join(score.hypothesis_words) for score in scored if score.hypothesis_words
    )
    return {
        "kind": "nonspeech",
        "items": len(items),
        "scored_items": len(scored),
        "failed_requests": len(failed),
        "hallucination_rate": _share(scored, lambda score: bool(score.hypothesis_words)),
        "words": {
            "mean": _mean(word_counts),
            "max": max(word_counts) if word_counts else 0,
            "share_at_least_5": _ratio(
                sum(1 for count in word_counts if count >= HALLUCINATION_LONG_WORDS), len(scored)
            ),
        },
        "top_hypotheses": [
            {"text": text, "count": count}
            for text, count in sorted(hypotheses.items(), key=lambda pair: (-pair[1], pair[0]))[
                :TOP_HYPOTHESES
            ]
        ],
        "by_category": _by_category(scored),
        "latency": _latency(items),
    }


def _view_wer(scored: list[UtteranceScore], view: str) -> dict | None:
    """Corpus WER for one view, or None where the references cannot support it."""
    supported = [score for score in scored if view in score.counts]
    if not supported:
        return None
    errors = sum(score.counts[view].errors for score in supported)
    words = sum(score.counts[view].reference_words for score in supported)
    return {
        "errors": errors,
        "reference_words": words,
        # No reference word means no error rate: a model that answers where there
        # is nothing to answer has not scored 0 %, it has scored nothing.
        "value": errors / words if words else None,
        "utterances": len(supported),
    }


def _share(scored: list[UtteranceScore], flag) -> dict:
    """The share of utterances a predicate holds for."""
    return _ratio(float(sum(1 for score in scored if flag(score))), len(scored))


def _is_wrong_sentence(score: UtteranceScore) -> bool:
    return score.counts["normalized"].errors > 0


def _has_loop_utterance(score: UtteranceScore) -> bool:
    return _has_loop(score.hypothesis_words, score.reference_words)


def _ratio(numerator: float, denominator: int) -> dict:
    return {
        "value": numerator / denominator if denominator else 0.0,
        "count": int(numerator),
        "total": denominator,
    }


def _tail(scored: list[UtteranceScore]) -> dict:
    """How bad the worst utterances are, in quantiles and in shares."""
    wers = [score.counts["normalized"].wer for score in scored]
    tail = {
        f"p{int(quantile * 100)}": float(np.percentile(wers, quantile * 100)) if wers else 0.0
        for quantile in TAIL_QUANTILES
    }
    for threshold in TAIL_SHARES:
        tail[f"share_over_{int(threshold * 100)}"] = _ratio(
            float(sum(1 for wer in wers if wer > threshold)), len(wers)
        )
    return tail


def _has_loop(hypothesis: list[str], reference: list[str]) -> bool:
    """True when the hypothesis repeats an n-gram three times and the reference does not."""
    for size in range(1, LOOP_NGRAM_MAX + 1):
        for start in range(len(hypothesis) - size * LOOP_REPEATS + 1):
            gram = hypothesis[start : start + size]
            if all(
                hypothesis[start + step * size : start + (step + 1) * size] == gram
                for step in range(LOOP_REPEATS)
            ) and not _repeats(reference, gram):
                return True
    return False


def _repeats(tokens_: list[str], gram: list[str]) -> bool:
    """True when the reference repeats ``gram`` the same way the hypothesis does."""
    size = len(gram)
    for start in range(len(tokens_) - size * LOOP_REPEATS + 1):
        if all(
            tokens_[start + step * size : start + (step + 1) * size] == gram
            for step in range(LOOP_REPEATS)
        ):
            return True
    return False


def _is_truncated(score: UtteranceScore) -> bool:
    """A long reference answered with much too few words."""
    reference = score.reference_words
    hypothesis = score.hypothesis_words
    if len(reference) < TRUNCATION_MIN_WORDS:
        return False
    return len(hypothesis) < TRUNCATION_SHARE * len(reference)


def _by_category(scored: list[UtteranceScore]) -> dict[str, dict]:
    """Hallucination numbers per category, so silence can be told from a cough."""
    categories: dict[str, list[UtteranceScore]] = {}
    for score in scored:
        categories.setdefault(score.item.category or "unknown", []).append(score)
    result = {}
    for name in sorted(categories):
        group = categories[name]
        invented = sum(1 for score in group if score.hypothesis_words)
        result[name] = {
            "items": len(group),
            "hallucinated": invented,
            "hallucination_rate": _ratio(invented, len(group)),
            "mean_words": _mean([len(score.hypothesis_words) for score in group]),
        }
    return result


def _latency(items: list[Item]) -> dict:
    """Warm latency and a real-time factor that includes the network."""
    warm = [item for item in items if not item.cold and not item.failed]
    latencies = [item.latency_s for item in warm]
    ratios = [item.latency_s / item.duration_s for item in warm if item.duration_s > 0]
    return {
        "warm_items": len(warm),
        "cold_items": sum(1 for item in items if item.cold),
        # A retried item costs more requests than the panel has items.
        "requests": sum(item.attempts for item in items),
        "median_s": _percentile(latencies, 50),
        "p95_s": _percentile(latencies, 95),
        "median_rtf": _percentile(ratios, 50),
        "mean_rtf": _mean(ratios),
        "label": "API RTF (includes network)",
    }


def _percentile(values: list[float], percent: float) -> float:
    return float(np.percentile(values, percent)) if values else 0.0


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0
