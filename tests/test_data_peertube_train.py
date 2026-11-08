"""Labelling ``peertube-fr-train`` without a speech server.

The three speech models are replaced by fakes, so these tests check which clips
are labelled, which labels are kept, what a manifest line carries, how talks are
batched, and that the models are asked one after the other.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import write_tone  # noqa: E402
from test_data_peertube import INSTANCE, video  # noqa: E402

from audiolab.core.audio import SAMPLE_RATE  # noqa: E402
from audiolab.data import peertube, peertube_train  # noqa: E402
from audiolab.data.build import ManifestBuilder  # noqa: E402
from audiolab.data.download import DownloadError  # noqa: E402
from audiolab.data.peertube_clips import Answers, Segment  # noqa: E402
from audiolab.data.peertube_train import Talk  # noqa: E402
from audiolab.data.store import read_manifest  # noqa: E402
from audiolab.data.train import Take  # noqa: E402
from audiolab.eval.client import TranscriptionError  # noqa: E402

WHISPER, QWEN, PARAKEET = (
    peertube_train.SEGMENT_MODEL,
    peertube_train.LABEL_MODEL,
    peertube_train.BASE_MODEL,
)


def segment(text: str, start: float, end: float) -> Segment:
    return Segment(start_s=start, end_s=end, text=text)


def whisper_row(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": f" {text}"}


def make_take(tmp_path, cap: int = 12_000, cap_per_group: int = 200) -> Take:
    return Take(
        name="peertube-fr-train",
        builder=ManifestBuilder(tmp_path / "dataset", verbose=False),
        language="fr",
        source_split="listing",
        license="CC-BY",
        text_has_casing_punct=True,
        group_kind="recording",
        cap=cap,
        cap_per_group=cap_per_group,
    )


def answering_by_length(texts: dict[float, str]):
    """A fake clip model that answers by clip length, counting its calls.

    A length it has no text for is a failed request.
    """
    calls = []

    def transcribe(wav: bytes) -> dict:
        seconds = round((len(wav) - 44) / 2 / SAMPLE_RATE, 2)
        calls.append(seconds)
        if seconds not in texts:
            raise TranscriptionError("HTTP 500")
        return {"text": texts[seconds]}

    return transcribe, calls


def make_talk(tmp_path, segments: list[Segment], seconds: float, **video_fields) -> Talk:
    path = write_tone(tmp_path / "talk.wav", seconds=seconds)
    return Talk(video(**video_fields), path, segments)


def test_agreement_ignores_casing_punctuation_and_how_numbers_are_written():
    assert peertube_train.disagreement("bonjour à tous", "Bonjour à tous !") == 0.0
    assert peertube_train.disagreement("en 2024", "En deux mille vingt-quatre.") == 0.0
    one_in_seven = peertube_train.disagreement(
        "on installe le paquet avec npm install", "On installe le paquet avec NPM installe."
    )
    assert one_in_seven == pytest.approx(1 / 7)
    assert one_in_seven <= peertube_train.MAX_DISAGREEMENT
    assert peertube_train.disagreement("un deux trois quatre", "un deux") == 0.5


def test_only_clips_the_two_models_agree_on_are_kept(tmp_path):
    talk = make_talk(
        tmp_path,
        [
            segment("Bonjour à tous.", 0.0, 6.0),
            segment("On parle de Kubernetes.", 6.0, 13.0),
            segment("Oui.", 13.5, 13.55),
            segment("Une phrase beaucoup trop longue.", 15.0, 35.0),
        ],
        seconds=40.0,
    )
    # With their padding, the clips last 6.0 s, 7.2 s, 0.45 s and 20.2 s.
    qwen, calls = answering_by_length({6.0: "Bonjour à tous.", 7.2: "Il pleut sur Toulouse."})
    tally = Counter()

    agreed = peertube_train.agreeing(talk, Answers(tmp_path / "q.jsonl"), qwen, 200, tally)

    assert [(clip.start_s, label) for clip, label in agreed] == [(0.0, "Bonjour à tous.")]
    assert calls == [6.0, 7.2], "a clip outside 0.5-19 s is not labelled"
    assert tally == Counter(
        {
            "clips labelled": 2,
            "clips where Whisper and Qwen disagree": 1,
            "clips outside 0.5-19 s": 2,
        }
    )
    assert "(50.0%) rejected" in peertube_train.report(tally)


def test_a_failed_label_request_is_counted_and_asked_again_next_time(tmp_path):
    talk = make_talk(
        tmp_path,
        [segment("Bonjour à tous.", 0.0, 6.0), segment("On parle de Kubernetes.", 7.0, 14.0)],
        seconds=20.0,
    )
    qwen, calls = answering_by_length({6.2: "Bonjour à tous."})
    tally = Counter()

    agreed = peertube_train.agreeing(talk, Answers(tmp_path / "q.jsonl"), qwen, 200, tally)
    peertube_train.agreeing(talk, Answers(tmp_path / "q.jsonl"), qwen, 200, Counter())

    assert len(agreed) == 1 and tally["clips Qwen failed on"] == 1
    assert calls == [6.2, 7.4, 7.4], "the answer is cached, the failure is not"


def test_a_talk_stops_being_labelled_at_its_share_of_the_cap(tmp_path):
    talk = make_talk(
        tmp_path, [segment(f"Phrase {i}.", 6.0 * i, 6.0 * i + 5.0) for i in range(5)], seconds=30.0
    )
    qwen, calls = answering_by_length({5.2: "Phrase 0.", 5.4: "Phrase 1."})

    agreed = peertube_train.agreeing(talk, Answers(tmp_path / "q.jsonl"), qwen, 2, Counter())

    assert len(agreed) == 2 and len(calls) == 2


def test_a_kept_clip_carries_its_attribution_and_the_base_models_error(tmp_path):
    talk = make_talk(tmp_path, [], seconds=20.0, uuid="v1", licence_id=7)
    talk.agreed = [
        (segment("Bonjour à tous.", 0.0, 6.0), "Bonjour à tous."),
        (segment("On parle de Kubernetes.", 6.0, 10.0), "On parle de Kubernetes."),
        (segment("Merci.", 10.0, 13.0), "Merci."),
    ]
    parakeet, _ = answering_by_length({6.0: "bonjour à tous", 4.0: "On parle de cube air net."})
    take = make_take(tmp_path)
    tally = Counter()

    kept = peertube_train.offer_talk(talk, Answers(tmp_path / "p.jsonl"), parakeet, take, tally)
    take.builder.write()

    assert kept == 3
    first, second, third = read_manifest(tmp_path / "dataset")
    assert first.id == "peertube-fr-train:v1-0000000"
    assert first.text == "Bonjour à tous." and first.text_has_casing_punct
    assert first.group == "peertube-fr-train:v1" and first.group_kind == "recording"
    assert first.license == "CC0-1.0", "the license is the video's own"
    assert first.source_url == f"{INSTANCE}/videos/watch/v1"
    assert first.source_title == "Talk v1" and first.source_author == "Capitole du Libre"
    assert first.base_wer == 0.0, "casing and punctuation are not errors"
    assert second.base_wer == 0.75, "the base model is scored, never asked to agree"
    assert third.base_wer is None, "a failed request leaves the error unknown, not the clip out"
    assert "gets 1 of the 3 kept clips wrong (33.3%)" in peertube_train.report(tally)
    assert tally["kept clips Parakeet failed on (no base_wer)"] == 1


def test_a_talk_that_cannot_be_downloaded_or_heard_is_counted_and_skipped(tmp_path, monkeypatch):
    def refuse(http, video, destination):
        raise DownloadError("HTTP 404")

    monkeypatch.setattr(peertube, "fetch_audio", refuse)
    assert peertube_train._audio_file(None, tmp_path, video()) is None

    def failing(wav: bytes) -> dict:
        raise TranscriptionError("HTTP 500")

    tally = Counter()
    missing = peertube_train.heard_talk(video("a"), None, tmp_path, failing, tally)
    deaf = peertube_train.heard_talk(
        video("b"), write_tone(tmp_path / "b.wav", seconds=2.0), tmp_path, failing, tally
    )

    assert missing.segments == deaf.segments == []
    assert tally == Counter({"talks that could not be downloaded": 1, "talks Whisper failed on": 1})
    assert (
        peertube_train.agreeing(missing, Answers(tmp_path / "q.jsonl"), failing, 200, tally) == []
    )


def test_the_first_batch_counts_on_the_expected_yield_and_later_ones_on_the_seen_one(tmp_path):
    take = make_take(tmp_path)
    assert peertube_train.batch_size(take, talks_read=0) == 86, "12 000 clips at 140 a talk"

    take.kept = 11_000
    assert peertube_train.batch_size(take, talks_read=88) == 11, "1 000 at 125 a talk, x 1.3"

    take.kept = 0
    assert peertube_train.batch_size(take, talks_read=5) == 15_600, "no yield counts as one"


class FakeClient:
    """Stands in for a TranscriptionClient of one served model, logging each request."""

    def __init__(self, model: str, answer, log: list[str]):
        self.model, self.answer, self.log = model, answer, log

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def transcribe(self, wav: bytes, language: str):
        assert language == "fr"
        self.log.append(self.model)
        return type("Transcription", (), {"answer": self.answer(wav)})()


def serve_fakes(monkeypatch, videos, answers: dict, log: list[str]) -> None:
    def fetch_audio(http, video, destination):
        log.append(f"download {video.uuid}")
        return write_tone(destination, seconds=12.0)

    monkeypatch.setattr(peertube, "read_listing", lambda: ("2026-11-08", videos))
    monkeypatch.setattr(peertube, "fetch_audio", fetch_audio)
    monkeypatch.setattr(
        peertube_train,
        "_client",
        lambda model, request=None: FakeClient(model, answers[model], log),
    )


def test_each_batch_runs_the_three_passes_one_model_after_the_other(tmp_path, monkeypatch, capsys):
    rows = [whisper_row(0.0, 5.0, "Bonjour."), whisper_row(6.0, 11.0, "Salut.")]
    answers = {
        WHISPER: lambda wav: {"segments": rows, "text": ""},
        QWEN: lambda wav: {"text": "Bonjour."},
        PARAKEET: lambda wav: {"text": "Bonjour."},
    }
    log: list[str] = []
    serve_fakes(monkeypatch, [video(uuid) for uuid in "abcd"], answers, log)
    # Each talk gives one clip; the first batch counts on two.
    monkeypatch.setattr(peertube_train, "TRAIN_CLIP_CAP", 3)
    monkeypatch.setattr(peertube_train, "EXPECTED_CLIPS_PER_TALK", 2)

    count = peertube_train.fetch_peertube_fr_train(
        "peertube-fr-train", tmp_path / "cache", tmp_path / "dataset", verbose=True
    )

    assert count == 3
    requests = [entry for entry in log if not entry.startswith("download")]
    first_batch = [WHISPER] * 2 + [QWEN] * 4 + [PARAKEET] * 2
    # The second batch reads two talks, at the one clip a talk seen so far, x 1.3;
    # the last clip fills the cap, and the fourth talk is not labelled.
    second_batch = [WHISPER] * 2 + [QWEN] + [PARAKEET]
    assert requests == first_batch + second_batch, (
        "each pass sends every request of its batch before the next pass starts"
    )
    clips = read_manifest(tmp_path / "dataset")
    assert [(clip.text, clip.base_wer) for clip in clips] == [("Bonjour.", 0.0)] * 3
    cache = tmp_path / "cache" / "peertube-fr-train"
    for model, talks in ((f"{WHISPER}.300s", 4), (f"{QWEN}.clips", 3), (f"{PARAKEET}.clips", 3)):
        assert len(list(cache.glob(f"*.{model}.jsonl"))) == talks, "one cache file per talk"
    assert "2 talks (" in capsys.readouterr().out


def test_a_talk_is_heard_while_the_next_one_downloads(tmp_path, monkeypatch):
    answers = {
        model: lambda wav: {"segments": [], "text": ""} for model in (WHISPER, QWEN, PARAKEET)
    }
    log: list[str] = []
    serve_fakes(monkeypatch, [video(uuid) for uuid in "abc"], answers, log)

    peertube_train.fetch_peertube_fr_train(
        "peertube-fr-train", tmp_path / "cache", tmp_path / "dataset", verbose=False
    )

    downloads = [position for position, entry in enumerate(log) if entry.startswith("download")]
    assert log.index(WHISPER) < downloads[2], (
        "only the next talk downloads before Whisper starts, not the whole batch"
    )
