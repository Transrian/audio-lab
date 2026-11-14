"""Label-audit cache and sample: the key holds everything that can change the result."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from audiolab.data.label_audit import (
    SAMPLE_SEED,
    AuditCacheError,
    answer_path,
    read_answer,
    read_sample,
    sample_path,
    write_answer,
    write_sample,
)

AUDIT = Path(__file__).resolve().parent.parent / "experiments/label-audit/audit.py"
spec = importlib.util.spec_from_file_location("label_audit_audit", AUDIT)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)

KEY = dict(model="stt-v-qwen3-asr-1.7B", language="en", request={}, audio_sha256="a" * 64)


def write(tmp_path: Path, **overrides) -> Path:
    path = tmp_path / "answer.json"
    write_answer(path, text="the quick brown fox", **{**KEY, **overrides})
    return path


def test_round_trip(tmp_path):
    path = write(tmp_path)
    assert read_answer(path, **KEY) == "the quick brown fox"


def test_missing_file_is_no_answer(tmp_path):
    assert read_answer(tmp_path / "none.json", **KEY) is None


def test_torn_file_is_no_answer(tmp_path):
    path = write(tmp_path)
    path.write_text(path.read_text()[:20], encoding="utf-8")
    assert read_answer(path, **KEY) is None


@pytest.mark.parametrize(
    "field, value",
    [
        ("model", "stt-v-whisper-large-v3"),
        ("language", "fr"),
        ("request", {"response_format": "verbose_json"}),
        ("audio_sha256", "b" * 64),
    ],
)
def test_a_changed_key_is_asked_again(tmp_path, field, value):
    """A model, a language, a request or an audio that differs is not served the old answer."""
    path = write(tmp_path)
    assert read_answer(path, **{**KEY, field: value}) is None


def test_a_text_field_that_is_not_text_is_no_answer(tmp_path):
    path = tmp_path / "answer.json"
    row = {**KEY, "text": 42}
    path.write_text(json.dumps(row), encoding="utf-8")
    assert read_answer(path, **KEY) is None


def test_answer_path_is_by_model_and_clip_id(tmp_path):
    path = answer_path(tmp_path, "stt-v-qwen3-asr-1.7B", "nasa-en-train:one-000001")
    expected = tmp_path / "label-audit" / "stt-v-qwen3-asr-1.7B" / "nasa-en-train:one-000001.json"
    assert path == expected


IDS = [f"nasa-en-train:talk-{index:05d}" for index in range(100)]


def test_draw_is_fixed_and_matches_an_independent_draw(tmp_path):
    path = sample_path(tmp_path, "nasa-en-train")
    draw = write_sample(path, dataset="nasa-en-train", count=30, seed=SAMPLE_SEED, manifest_ids=IDS)
    assert draw == random.Random(SAMPLE_SEED).sample(sorted(IDS), 30)
    # The second read takes the file back, and it holds the same draw.
    assert (
        read_sample(path, dataset="nasa-en-train", count=30, seed=SAMPLE_SEED, manifest_ids=IDS)
        == draw
    )


def test_draw_rejected_when_the_dataset_changed(tmp_path):
    path = sample_path(tmp_path, "nasa-en-train")
    write_sample(path, dataset="nasa-en-train", count=30, seed=SAMPLE_SEED, manifest_ids=IDS)
    other_ids = IDS[:-1] + ["nasa-en-train:talk-00123"]
    assert (
        read_sample(
            path, dataset="nasa-en-train", count=30, seed=SAMPLE_SEED, manifest_ids=other_ids
        )
        is None
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"count": 31},
        {"seed": SAMPLE_SEED + 1},
        {"dataset": "other-en-train"},
    ],
)
def test_draw_rejected_when_the_caller_asks_something_else(tmp_path, overrides):
    path = sample_path(tmp_path, "nasa-en-train")
    write_sample(path, dataset="nasa-en-train", count=30, seed=SAMPLE_SEED, manifest_ids=IDS)
    call = {
        "dataset": "nasa-en-train",
        "count": 30,
        "seed": SAMPLE_SEED,
        "manifest_ids": IDS,
    }
    call.update(overrides)
    assert read_sample(path, **call) is None


def test_draw_longer_than_the_dataset_is_refused(tmp_path):
    with pytest.raises(AuditCacheError, match="asks for 101 clips"):
        write_sample(
            sample_path(tmp_path, "nasa-en-train"),
            dataset="nasa-en-train",
            count=101,
            seed=SAMPLE_SEED,
            manifest_ids=IDS,
        )


def test_torn_sample_file_is_no_sample(tmp_path):
    path = sample_path(tmp_path, "nasa-en-train")
    write_sample(path, dataset="nasa-en-train", count=30, seed=SAMPLE_SEED, manifest_ids=IDS)
    path.write_text(path.read_text()[:20], encoding="utf-8")
    assert (
        read_sample(path, dataset="nasa-en-train", count=30, seed=SAMPLE_SEED, manifest_ids=IDS)
        is None
    )


# --- the scoring: the agreement rule on synthetic rows ----------------------


def row(human, whisper, qwen, base_wer=None) -> dict:
    return {
        "id": f"nasa-en-train:talk-{len(human):05d}",
        "human": human,
        "whisper": whisper,
        "qwen": qwen,
        "base_wer": base_wer,
        "language": "en",
    }


def test_verdicts_apply_the_pipeline_rule_and_score():
    rows = [
        # exact: agreed, kept, no error
        row("one two three four", "one two three four", "one two three four", 0.05),
        # one missing word: 1/10 disagreement, kept, 1 error over 10 words
        row(
            "one two three four five six seven eight nine ten",
            "one two three four five six seven eight nine ten",
            "one two three four five six seven eight nine",
            0.15,
        ),
        # four different words: 1.0 disagreement, rejected, 4 substitutions
        row("one two three four", "one two three four", "five six seven eight", 0.4),
    ]
    scored = audit.verdicts(rows)
    assert [row_["kept"] for row_ in scored] == [True, True, False]
    assert scored[0]["errors"] == 0
    assert scored[1]["errors"] == 1 and scored[1]["per_clip_wer"] == pytest.approx(0.1)
    assert scored[2]["errors"] == 4 and scored[2]["per_clip_wer"] == pytest.approx(1.0)


def test_summary_splits_kept_and_rejected_and_buckets():
    rows = [
        row("one two three four", "one two three four", "one two three four", 0.05),
        row(
            "one two three four five six seven eight nine ten",
            "one two three four five six seven eight nine ten",
            "one two three four five six seven eight nine",
            0.15,
        ),
        row("one two three four", "one two three four", "five six seven eight", 0.4),
    ]
    summary = audit.summarize(rows)
    assert summary["clips"] == 3
    assert summary["kept"]["count"] == 2
    assert summary["kept"]["wer"] == pytest.approx(1 / 14)
    assert summary["rejected"]["count"] == 1
    assert summary["rejected"]["share"] == pytest.approx(1 / 3)
    assert summary["rejected"]["wer"] == pytest.approx(1.0)
    buckets = {b["bucket"]: b for b in summary["base_wer_buckets"]}
    assert buckets["base_wer < 0.1"]["clips"] == 1
    assert buckets["base_wer < 0.1"]["kept_count"] == 1
    assert buckets["base_wer < 0.1"]["kept_wer"] == pytest.approx(0.0)
    assert buckets["0.1 <= base_wer < 0.2"]["kept_wer"] == pytest.approx(0.1)
    assert buckets["base_wer >= 0.3"]["kept_count"] == 0
    assert buckets["base_wer >= 0.3"]["rejected_wer"] == pytest.approx(1.0)


def test_summary_without_base_wer_has_no_buckets():
    summary = audit.summarize([row("one two", "one two", "one two")])
    assert "base_wer_buckets" not in summary


# --- the example categories --------------------------------------------------


def diffs(human, qwen) -> list:
    # Whisper hears what Qwen hears, so the rule keeps the clip and only the
    # label is what is examined.
    scored = audit.verdicts([row(human, qwen, qwen)])
    assert scored[0]["kept"], "the test rows must be kept clips"
    return audit.clip_diffs(scored[0])


def test_category_filler_insertions():
    assert audit.category(diffs("we are ready", "we are yeah ready")) == "filler insertions"


def test_category_numbers():
    assert audit.category(diffs("we launched in 2010", "we launched in 2012")) == "numbers"


def test_category_name_spellings():
    assert audit.category(diffs("we heard Curry again", "we heard Kerry again")) == "name spellings"


def test_category_other():
    assert audit.category(diffs("the data was lost", "the data is lost")) == "other"


def test_sentence_start_capital_is_not_a_name():
    # Both capitalized, but at the start of the sentence: not a name spelling.
    assert audit.category(diffs("Curry came to the lab", "Kerry came to the lab")) == "other"


def test_raw_under_carries_a_number_onto_every_word_it_reads_as():
    tokens = audit.tokens("the year 2024 was good", "en", "normalized")
    mapped = audit.raw_under(tokens, "the year 2024 was good", "en")
    assert [word for word, _ in mapped] == [
        "the",
        "year",
        "2024",
        "2024",
        "2024",
        "was",
        "good",
    ]


def test_raw_under_is_none_when_the_words_do_not_recompose_the_tokens():
    # The normalizer drops the "and" of "one hundred and one" in context, per
    # word it keeps it, so the raw words cannot be put under the tokens.
    tokens = audit.tokens("one hundred and one", "en", "normalized")
    assert len(tokens) == 3
    assert audit.raw_under(tokens, "one hundred and one", "en") is None
