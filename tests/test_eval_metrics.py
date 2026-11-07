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


def test_edit_counts_are_the_usual_three_kinds():
    assert metrics.edit_counts(["a", "b", "c"], ["a", "b", "c"]) == metrics.Counts(0, 0, 0, 3)
    assert metrics.edit_counts(["a", "b", "c"], ["a", "x", "c"]).substitutions == 1
    assert metrics.edit_counts(["a", "b", "c"], ["a", "c"]).deletions == 1
    assert metrics.edit_counts(["a", "c"], ["a", "b", "c"]).insertions == 1
    counts = metrics.edit_counts(["a", "b", "c"], ["a", "x", "c", "d"])
    assert counts.errors == 2 and counts.reference_words == 3
    assert counts.wer == 2 / 3


def test_wer_of_an_empty_reference_is_zero():
    assert metrics.edit_counts([], ["invented"]).wer == 0.0


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
