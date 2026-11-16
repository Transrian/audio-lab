"""Label-audit cache and sample: the key holds everything that can change the result."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
from helpers import make_clip, write_tone

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
from audiolab.data.store import write_manifest
from audiolab.data.talks import LABEL_MODEL, SEGMENT_MODEL, Served, model_config

AUDIT = Path(__file__).resolve().parent.parent / "experiments/label-audit/audit.py"
spec = importlib.util.spec_from_file_location("label_audit_audit", AUDIT)
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)

KEY = dict(
    model="stt-v-qwen3-asr-1.7B",
    language="en",
    request={},
    send_language=True,
    audio_sha256="a" * 64,
)
# The row the cache held before the send_language field existed: the default
# caller's key, the same fields, no flag.
OLD_ROW = {
    "model": "stt-v-qwen3-asr-1.7B",
    "language": "en",
    "request": {},
    "audio_sha256": "a" * 64,
    "text": "the quick brown fox",
}


def write_old_row(path: Path, *, model: str, language: str, audio_sha256: str, text: str) -> None:
    """One row as the cache wrote it before the send_language field existed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "model": model,
        "language": language,
        "request": {},
        "audio_sha256": audio_sha256,
        "text": text,
    }
    path.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


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
        ("send_language", False),
        ("audio_sha256", "b" * 64),
    ],
)
def test_a_changed_key_is_asked_again(tmp_path, field, value):
    """A model, a language, a request, the language flag or an audio that differs
    is not served the old answer."""
    path = write(tmp_path)
    assert read_answer(path, **{**KEY, field: value}) is None


def test_the_default_key_writes_byte_identical_to_the_old_row(tmp_path):
    """The default (an empty request, the language sent) writes exactly the
    fields and bytes the cache held before the send_language field existed."""
    path = write(tmp_path)
    old = json.dumps(OLD_ROW, ensure_ascii=False, sort_keys=True) + "\n"
    assert path.read_bytes() == old.encode("utf-8")


def test_an_answer_cached_by_the_old_code_is_reused_by_the_default(tmp_path):
    """A row written before the send_language field existed is read back by the
    default key: the caches on disk stay valid, and nothing is asked again. A
    row that does carry the field, with the default's value, is read too."""
    path = tmp_path / "answer.json"
    old_bytes = json.dumps(OLD_ROW, ensure_ascii=False, sort_keys=True) + "\n"
    path.write_text(old_bytes, encoding="utf-8")
    assert read_answer(path, **KEY) == "the quick brown fox"
    assert read_answer(path, **{**KEY, "send_language": False}) is None
    assert read_answer(path, **{**KEY, "request": {"temperature": 0}}) is None
    row = {**OLD_ROW, "send_language": True}
    path.write_text(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    assert read_answer(path, **KEY) == "the quick brown fox"


def test_not_sending_the_language_is_asked_again(tmp_path):
    """An answer asked with the language is not read by a caller that does not
    send it, and the reverse: the flag is in the key."""
    path = tmp_path / "answer.json"
    write_answer(path, text="asked with the language", **KEY)
    no_language = dict(KEY, send_language=False)
    assert read_answer(path, **no_language) is None
    write_answer(path, text="asked without the language", **no_language)
    assert read_answer(path, **KEY) is None
    assert read_answer(path, **no_language) == "asked without the language"


def test_two_effective_requests_are_asked_again_of_each_other(tmp_path):
    """An empty request and a non-empty one are different questions on the same
    clip: neither answer is read by the other's key."""
    path = tmp_path / "answer.json"
    write_answer(path, text="the default answer", **KEY)
    effective = dict(KEY, request={"temperature": 0})
    assert read_answer(path, **effective) is None
    write_answer(path, text="the effective answer", **effective)
    assert read_answer(path, **KEY) is None
    assert read_answer(path, **effective) == "the effective answer"


def test_a_non_default_row_carries_its_request_and_the_flag(tmp_path):
    """Off the default the row holds the effective request and the flag
    explicitly, whatever it is, and reads it back."""
    path = tmp_path / "answer.json"
    effective = dict(KEY, request={"temperature": 0})
    write_answer(path, text="the effective answer", **effective)
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["request"] == {"temperature": 0}
    assert row["send_language"] is True
    assert read_answer(path, **effective) == "the effective answer"
    no_language = dict(KEY, send_language=False)
    write_answer(path, text="asked without the language", **no_language)
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["request"] == {}
    assert row["send_language"] is False
    assert read_answer(path, **no_language) == "asked without the language"


def test_a_request_that_sends_differently_is_asked_again(tmp_path):
    """The request is compared as the client sends it, field by field:
    ``{t: 0}`` sends ``"0"`` and ``{t: false}`` sends ``"False"`` — different
    requests, so the second is asked again, though Python calls the dicts equal."""
    path = tmp_path / "answer.json"
    zero = dict(KEY, request={"t": 0})
    write_answer(path, text="asked with zero", **zero)
    assert read_answer(path, **zero) == "asked with zero"
    assert read_answer(path, **dict(KEY, request={"t": False})) is None


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


# --- the transcribe pass: the key the cache is read and written with ---------


def _small_source(root: Path) -> list:
    """A prepared ``nasa-en-train`` of two clips, with the real tones behind them."""
    dataset_dir = root / "nasa-en-train"
    clips = []
    for index in range(2):
        clip_id = f"nasa-en-train:talk-{index:05d}"
        seconds = 1.0 + index * 0.1
        write_tone(dataset_dir / f"audio/{clip_id.replace(':', '_')}.wav", seconds=seconds)
        clip = make_clip(
            clip_id,
            language="en",
            group=f"nasa-en-train:talk-{index:05d}",
            duration_s=seconds,
        )
        clips.append(clip)
    write_manifest(dataset_dir, clips)
    return clips


def _run_transcribe(
    tmp_path: Path, monkeypatch, cache: Path, *, changed: dict | None = None
) -> list:
    """audit.transcribe over the two-clip source, the server faked; what was asked.

    ``changed`` holds a field of the model config to move off the committed one
    (``request``, ``send_language``); nothing moves it when it is None.
    """
    root = tmp_path / "data"
    _small_source(root)
    monkeypatch.setitem(
        audit.AUDITS, "nasa-en-train", {"count": 2, "language": "en", "human": "a test label"}
    )
    monkeypatch.setattr(audit, "cache_dir", lambda: cache)
    if changed is not None:
        base = {model: model_config(model) for model in (SEGMENT_MODEL, LABEL_MODEL)}
        monkeypatch.setattr(
            audit, "model_config", lambda model: dataclasses.replace(base[model], **changed)
        )
    asked: list[bytes] = []

    def fake_transcribe(wav: bytes) -> dict:
        asked.append(wav)
        return {"text": "a fresh transcript"}

    class FakeClient:
        def close(self) -> None:
            pass

    monkeypatch.setattr(audit, "client", lambda model: FakeClient())
    monkeypatch.setattr(
        audit, "served", lambda client_, language: Served(transcribe=fake_transcribe, concurrency=2)
    )
    audit.transcribe("nasa-en-train", root, verbose=False)
    return asked


def test_transcribe_reads_the_answers_cached_by_the_old_code(tmp_path, monkeypatch):
    """A cache the old code wrote — the default key, no send_language field —
    is read by the committed config: nothing is asked."""
    cache = tmp_path / "cache"
    for clip in _small_source(tmp_path / "data"):
        for model in (SEGMENT_MODEL, LABEL_MODEL):
            write_old_row(
                answer_path(cache, model, clip.id),
                model=model,
                language="en",
                audio_sha256=clip.sha256,
                text=f"the old answer of {clip.id}",
            )
    assert _run_transcribe(tmp_path, monkeypatch, cache) == []


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"request": {"temperature": 0}}, id="request"),
        pytest.param({"send_language": False}, id="send_language"),
    ],
)
def test_transcribe_asks_again_when_the_model_config_changes_the_key(
    tmp_path, monkeypatch, changed
):
    """An answer cached under the committed config is not read when the config
    now asks another way: the effective request or the language flag changed,
    so every clip of every model is asked again."""
    cache = tmp_path / "cache"
    for clip in _small_source(tmp_path / "data"):
        for model in (SEGMENT_MODEL, LABEL_MODEL):
            write_answer(
                answer_path(cache, model, clip.id),
                model=model,
                language="en",
                audio_sha256=clip.sha256,
                request={},
                send_language=True,
                text=f"the old answer of {clip.id}",
            )
    asked = _run_transcribe(tmp_path, monkeypatch, cache, changed=changed)
    assert len(asked) == 4  # two models, two clips: nothing from the old key was read
    # The fresh answers were written under the changed key, not the default's:
    # the default read (the relabel fetcher's key) does not get them back.
    base = {model: model_config(model) for model in (SEGMENT_MODEL, LABEL_MODEL)}
    for clip in _small_source(tmp_path / "data"):
        for model in (SEGMENT_MODEL, LABEL_MODEL):
            path = answer_path(cache, model, clip.id)
            row = json.loads(path.read_text(encoding="utf-8"))
            assert row["request"] == changed.get("request", base[model].request)
            assert row.get("send_language", True) == changed.get(
                "send_language", base[model].send_language
            )
            assert (
                read_answer(
                    path,
                    model=model,
                    language="en",
                    audio_sha256=clip.sha256,
                    request={},
                    send_language=True,
                )
                is None
            )
    # And the score side reads the fresh answers back under the changed key:
    # a load that read the default key would find nothing and refuse.
    rows = audit.load_answers("nasa-en-train", audit.AUDITS["nasa-en-train"], tmp_path / "data")
    assert [(row["whisper"], row["qwen"]) for row in rows] == [
        ("a fresh transcript", "a fresh transcript"),
        ("a fresh transcript", "a fresh transcript"),
    ]


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


def test_align_ops_consumes_every_token_exactly_once():
    """The edit path is a valid alignment: each side is consumed, in order, once."""
    cases = [
        (["we"], ["we", "are", "ready"]),  # insertions at the end: the row boundary
        (["we", "are", "ready"], ["we"]),  # deletions at the end: the column boundary
        ([], ["a", "b"]),
        (["a", "b"], []),
        (["a"], ["b"]),
        (["the", "quick"], ["quick", "brown", "fox"]),
        (["a", "b", "c", "d"], ["a", "x", "c", "y", "d"]),
    ]
    for reference, hypothesis in cases:
        ops = audit.align_ops(reference, hypothesis)
        consumed_reference = [token for _, token, _ in ops if token]
        consumed_hypothesis = [token for _, _, token in ops if token]
        assert consumed_reference == reference
        assert consumed_hypothesis == hypothesis
