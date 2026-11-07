"""Metrics: the numbers a panel reports, on hand-built items."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402

from audiolab.eval import metrics  # noqa: E402
from audiolab.eval.metrics import Item  # noqa: E402


def speech_item(
    identifier: str, reference: str, hypothesis: str, group: str = "s1", **extra
) -> Item:
    values = {
        "id": identifier,
        "panel": "fake-full",
        "group": group,
        "category": "",
        "language": "fr",
        "reference": reference,
        "hypothesis": hypothesis,
        "duration_s": 10.0,
        "latency_s": 1.0,
        "text_has_casing_punct": True,
    }
    values.update(extra)
    return Item(**values)


def battery_item(identifier: str, hypothesis: str, category: str = "silence") -> Item:
    return Item(
        id=identifier,
        panel="nonspeech-full",
        group=identifier,
        category=category,
        language="none",
        reference="",
        hypothesis=hypothesis,
        duration_s=5.0,
        latency_s=1.0,
    )


def test_an_empty_reference_puts_its_insertions_in_the_tail():
    """A hypothesis over an empty reference is an error, not a perfect utterance.

    The sentence error rate already counts it, so the per-utterance WER has to
    as well, or the worst tenth of a panel would hide a model talking over a
    reference that normalized away (a filler word, punctuation only). Panel WER
    keeps its own denominator: the invented words are errors counted over the
    words that were really there.
    """
    talking, silent, clean = metrics.score_items(
        [
            speech_item("s1", "euh", "bonjour tout le monde"),
            speech_item("s2", "euh", ""),
            speech_item("s3", "bonjour le monde", "bonjour le monde"),
        ]
    )
    summary = metrics.speech_summary([talking.item, silent.item, clean.item])
    assert talking.counts["normalized"].reference_words == 0
    assert talking.counts["normalized"].insertions == 4
    assert talking.counts["normalized"].wer == 4.0
    assert silent.counts["normalized"].wer == 0.0
    assert summary["sentence_error_rate"]["value"] == pytest.approx(1 / 3)
    assert summary["tail"]["share_over_50"]["value"] == pytest.approx(1 / 3)
    # Four invented words over the three reference words of the panel.
    assert summary["wer"]["normalized"]["value"] == pytest.approx(4 / 3)
    assert summary["wer"]["normalized"]["reference_words"] == 3


def test_a_panel_with_no_reference_words_at_all_has_no_wer():
    """With nothing to divide by, the rate is `n/a` and the errors still count."""
    talking = metrics.score_items([speech_item("s1", "euh", "bonjour")])[0]
    summary = metrics.speech_summary([talking.item])
    assert summary["wer"]["normalized"]["value"] is None
    assert summary["wer"]["normalized"]["errors"] == 1


def test_utterance_wer_uses_the_normalizer():
    item = speech_item("a", "Il a 21 ans.", "il a vingt et un ans", text_has_casing_punct=True)
    assert metrics.counts(item.reference, item.hypothesis, "fr", "normalized").errors == 0
    assert metrics.counts(item.reference, item.hypothesis, "fr", "strict").errors > 0


def test_formatted_view_is_only_computed_when_the_reference_has_it():
    with_format = speech_item("a", "Bonjour, monde.", "bonjour monde")
    without = speech_item("b", "bonjour monde", "bonjour monde", text_has_casing_punct=False)
    scored = {score.item.id: score.counts for score in metrics.score_items([with_format, without])}
    assert set(scored["a"]) == {"normalized", "strict", "formatted"}
    assert set(scored["b"]) == {"normalized", "strict"}


def test_failed_requests_are_not_scored():
    items = [
        speech_item("a", "un deux", "un deux"),
        speech_item("b", "trois", "", error="HTTP 500"),
    ]
    summary = metrics.speech_summary(items)
    assert summary["items"] == 2 and summary["scored_items"] == 1
    assert summary["failed_requests"] == 1
    assert summary["wer"]["normalized"]["reference_words"] == 2


def test_speech_summary_counts_what_the_report_shows():
    reference = "un deux trois quatre cinq"
    items = [
        speech_item("a", reference, reference),
        speech_item("b", reference, "un deux trois quatre", group="s2"),
        speech_item("c", reference, "", group="s3"),
    ]
    summary = metrics.speech_summary(items)
    assert summary["wer"]["normalized"]["errors"] == 6
    assert summary["wer"]["normalized"]["value"] == 6 / 15
    assert summary["sentence_error_rate"]["value"] == 2 / 3
    assert summary["empty_output_rate"]["value"] == 1 / 3
    assert summary["truncation_rate"]["value"] == 1 / 3  # only the empty one is cut short
    assert summary["tail"]["share_over_25"]["value"] == 1 / 3
    assert summary["tail"]["p95"] >= summary["tail"]["p90"]


def test_short_references_are_never_called_truncated():
    items = [speech_item("a", "un deux trois quatre", "un")]
    summary = metrics.speech_summary(items)
    assert summary["truncation_rate"]["value"] == 0.0
    assert summary["empty_output_rate"]["value"] == 0.0  # "un" is short, not empty


def test_a_repeated_phrase_is_a_loop_only_when_the_reference_does_not_repeat_it():
    looping = speech_item("a", "le chat", "le chat le chat le chat")
    repeating = speech_item("b", "le chat le chat le chat", "le chat le chat le chat")
    summary = metrics.speech_summary([looping, repeating])
    assert summary["loop_rate"]["count"] == 1


def test_a_single_repeated_word_is_a_loop():
    assert metrics._has_loop(["bonjour", "bonjour", "bonjour"], ["salut"])


def test_two_repetitions_are_not_a_loop():
    assert not metrics._has_loop(["bonjour", "bonjour"], ["salut"])


def test_latency_excludes_cold_calls_and_reports_a_network_inclusive_rtf():
    items = [
        speech_item("a", "un deux", "un deux", latency_s=9.0, cold=True, duration_s=10.0),
        speech_item("b", "un deux", "un deux", latency_s=1.0, duration_s=10.0, group="s2"),
        speech_item("c", "un deux", "un deux", latency_s=2.0, duration_s=10.0, group="s3"),
    ]
    latency = metrics.speech_summary(items)["latency"]
    assert latency["cold_items"] == 1 and latency["warm_items"] == 2
    assert latency["median_s"] == 1.5
    assert latency["mean_rtf"] == pytest.approx(0.15)
    assert latency["label"] == "API RTF (includes network)"
    assert latency["requests"] == 3


def test_latency_counts_the_requests_a_retried_item_cost():
    """A panel of two clips answered in five requests says so."""
    items = [
        speech_item("a", "un deux", "un deux", latency_s=1.0, attempts=3),
        speech_item("b", "un deux", "un deux", latency_s=1.0, attempts=2, group="s2"),
    ]
    latency = metrics.speech_summary(items)["latency"]
    assert (latency["warm_items"], latency["requests"]) == (2, 5)


def test_nonspeech_summary_counts_hallucinations_by_category():
    items = [
        battery_item("a", "", "silence"),
        battery_item("b", "thank you for watching this video", "silence"),
        battery_item("c", "oh", "noise"),
    ]
    summary = metrics.nonspeech_summary(items)
    assert summary["hallucination_rate"]["value"] == 2 / 3
    assert summary["words"]["mean"] == 7 / 3
    assert summary["words"]["max"] == 6
    assert summary["words"]["share_at_least_5"]["value"] == 1 / 3
    assert summary["top_hypotheses"] == [
        {"text": "oh", "count": 1},
        {"text": "thank you for watching this video", "count": 1},
    ]
    assert summary["by_category"]["silence"]["hallucinated"] == 1
    assert summary["by_category"]["silence"]["hallucination_rate"]["value"] == 0.5
    assert summary["by_category"]["noise"]["mean_words"] == 1.0


def test_the_battery_counts_words_in_the_same_view_as_speech():
    """One view for every item, so a word count means the same thing everywhere.

    "21 %" is three words ("twenty one percent") on the battery too; counting it
    as the single token the strict view produces would understate how much a
    model invented on a clip where it should have said nothing.
    """
    summary = metrics.nonspeech_summary([battery_item("a", "21 %")])
    assert summary["words"]["mean"] == 3
    assert summary["words"]["share_at_least_5"]["value"] == 0.0


def test_a_panel_with_no_reference_words_has_no_wer():
    """A rate over zero reference words is not 0 %, it is undefined."""
    items = [
        speech_item("one", "", "bonjour"),
        speech_item("two", "", ""),
    ]
    summary = metrics.speech_summary(items)
    assert summary["wer"]["normalized"]["value"] is None
    assert summary["wer"]["normalized"]["errors"] == 1
    assert summary["wer"]["normalized"]["reference_words"] == 0
    assert summary["wer"]["normalized"]["utterances"] == 2


def test_top_hypotheses_are_ordered_by_count_then_text():
    items = [battery_item(letter, "bruit blanc") for letter in "abc"] + [battery_item("d", "ah")]
    top = metrics.nonspeech_summary(items)["top_hypotheses"]
    assert top[0] == {"text": "bruit blanc", "count": 3}
    assert top[1] == {"text": "ah", "count": 1}


def test_totals_by_group_add_up_per_group():
    items = [
        speech_item("a", "un deux", "un", group="s1"),
        speech_item("b", "un deux", "un deux", group="s1"),
    ]
    scored = metrics.score_items(items)
    totals = metrics.totals_by_group(scored, metrics.wer_value("normalized"))
    assert totals == {"s1": (1.0, 4.0)}


def test_a_hypothesis_that_cannot_be_scored_is_recorded_and_the_rest_still_score():
    """One unreadable answer must not cost the panel its numbers.

    Scoring runs on model output, so the reader has to be total. This pins the
    second half of that: if a reader ever does raise again, the run records a
    scoring error for that utterance, counts it, and keeps the other utterances.
    """
    items = [
        speech_item("s1", "un deux trois", "un deux trois"),
        speech_item("s2", "un deux trois", "un deux trois"),
        speech_item("s3", "un deux trois", "un deux trois"),
    ]
    original = metrics._counts_for

    def counts_or_fails(item):
        if item.id == "s2":
            raise ValueError("int() could not read the number in this hypothesis")
        return original(item)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(metrics, "_counts_for", counts_or_fails)
    try:
        scored = metrics.score_items(items)
        assert [score.error for score in scored] == ["", scored[1].error, ""]
        assert "int() could not read the number" in scored[1].error
        assert [score.item.id for score in metrics.scorable(scored)] == ["s1", "s3"]
        summary = metrics.speech_summary(items)
    finally:
        monkeypatch.undo()
    assert summary["items"] == 3
    assert summary["scored_items"] == 2
    assert summary["scoring_errors"] == {
        "count": 1,
        "items": [{"id": "s2", "error": scored[1].error}],
    }
    # The numbers cover the two utterances that scored, not three.
    assert summary["wer"]["normalized"]["utterances"] == 2
    assert summary["wer"]["normalized"]["reference_words"] == 6
    assert summary["sentence_error_rate"]["total"] == 2


def test_a_scoring_error_leaves_no_group_of_zeros_behind():
    """A group whose only utterance failed to score carries no numbers at all."""
    items = [
        speech_item("a", "un deux", "un deux", group="s1"),
        speech_item("b", "un deux", "un deux", group="s2"),
    ]
    original = metrics._counts_for

    def counts_or_fails(item):
        if item.group == "s2":
            raise KeyError("'°'")
        return original(item)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(metrics, "_counts_for", counts_or_fails)
    try:
        totals = metrics.totals_by_group(
            metrics.score_items(items), metrics.wer_value("normalized")
        )
    finally:
        monkeypatch.undo()
    assert totals == {"s1": (0.0, 2.0)}
