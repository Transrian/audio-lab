"""Labelling ``peertube-fr-train`` without a speech server.

The two speech models are replaced by fakes, so these tests check how a talk is
cut into segments, which segments are labelled, which labels are kept, and what
a manifest line carries.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import write_tone  # noqa: E402
from test_data_peertube import INSTANCE, video  # noqa: E402

from audiolab.core.audio import SAMPLE_RATE, Audio  # noqa: E402
from audiolab.data import peertube, peertube_train  # noqa: E402
from audiolab.data.build import ManifestBuilder  # noqa: E402
from audiolab.data.peertube_train import ClipTexts, Segment, Talk  # noqa: E402
from audiolab.data.store import read_manifest  # noqa: E402
from audiolab.data.train import Take  # noqa: E402


def silence(seconds: float) -> Audio:
    return Audio(
        samples=np.zeros(round(seconds * SAMPLE_RATE), np.float32), sample_rate=SAMPLE_RATE
    )


def whisper_row(start: float, end: float, text: str, **fields) -> dict:
    return {"start": start, "end": end, "text": f" {text}", **fields}


def test_a_talk_is_sent_in_chunks_that_restart_after_the_last_whole_segment():
    sent = []

    def whisper(wav: bytes) -> dict:
        sent.append((len(wav) - 44) / 2 / SAMPLE_RATE)
        if len(sent) == 1:
            rows = [whisper_row(0.0, 120.0, "un"), whisper_row(120.0, 290.0, "deux")]
            return {"segments": rows + [whisper_row(290.0, 300.0, "coupé au mi-")]}
        return {"segments": [whisper_row(0.0, 5.0, "trois"), whisper_row(5.0, 100.0, "quatre")]}

    segments = peertube_train.segment_audio(silence(390.0), whisper)

    assert sent == [300.0, 100.0], "the second chunk starts at 290 s, where 'deux' ended"
    assert [(s.start_s, s.end_s, s.text) for s in segments] == [
        (0.0, 120.0, "un"),
        (120.0, 290.0, "deux"),
        (290.0, 295.0, "trois"),
        (295.0, 390.0, "quatre"),
    ]


def test_a_chunk_without_a_whole_segment_moves_on_by_a_full_chunk():
    answers = iter([{"segments": [whisper_row(0.0, 300.0, "musique")]}, {"segments": []}])
    sent = []

    def whisper(wav: bytes) -> dict:
        sent.append(wav)
        return next(answers)

    assert peertube_train.segment_audio(silence(400.0), whisper) == []
    assert len(sent) == 2


def segment(text: str = "bonjour à tous", start: float = 0.0, end: float = 2.0, **fields):
    return Segment(start_s=start, end_s=end, text=text, **fields)


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (segment(), ""),
        (segment(""), "without words"),
        (segment("Euh..."), "without words"),
        (segment("merci " * 20), "in a repetition loop"),
        (segment(no_speech_prob=0.9, avg_logprob=-1.5), "not speech"),
        (segment(no_speech_prob=0.9, avg_logprob=-0.2), ""),
    ],
)
def test_whisper_doubts_are_found_before_a_segment_is_labelled(candidate, reason):
    assert peertube_train.doubt(candidate) == reason


def spans(clips: list[Segment]) -> list[tuple[float, float, str]]:
    return [(clip.start_s, clip.end_s, clip.text) for clip in clips]


def test_phrases_are_joined_up_to_a_full_stop_once_the_clip_lasts_five_seconds():
    phrases = [
        segment("Alors, F-Droid,", 0.0, 1.5),
        segment("c'est un magasin d'applications.", 1.5, 6.0),
        segment("Oui.", 6.2, 6.5),
        segment("Mais pas comme le Play Store ?", 6.5, 12.0),
    ]
    assert spans(peertube_train.candidates(phrases, Counter())) == [
        (0.0, 6.0, "Alors, F-Droid, c'est un magasin d'applications."),
        (6.2, 12.0, "Oui. Mais pas comme le Play Store ?"),
    ]


def test_a_long_pause_a_doubtful_segment_or_the_length_limit_end_a_sentence():
    left_out = Counter()
    phrases = [
        segment("avant la pause", 0.0, 2.0),
        segment("après la pause", 3.5, 5.0),
        segment("merci " * 20, 5.0, 8.0),
        segment("et on reprend", 8.0, 9.0),
        segment("une phrase sans fin", 9.0, 28.0),
    ]
    assert spans(peertube_train.candidates(phrases, left_out)) == [
        (0.0, 2.0, "avant la pause"),
        (3.5, 5.0, "après la pause"),
        (8.0, 9.0, "et on reprend"),
        (9.0, 28.0, "une phrase sans fin"),
    ]
    assert left_out == Counter({"segments in a repetition loop": 1})


def test_agreement_ignores_casing_punctuation_and_how_numbers_are_written():
    assert peertube_train.disagreement("bonjour à tous", "Bonjour à tous !") == 0.0
    assert peertube_train.disagreement("en 2024", "En deux mille vingt-quatre.") == 0.0
    one_in_seven = peertube_train.disagreement(
        "on installe le paquet avec npm install", "On installe le paquet avec NPM installe."
    )
    assert one_in_seven == pytest.approx(1 / 7)
    assert one_in_seven <= peertube_train.MAX_DISAGREEMENT
    assert peertube_train.disagreement("un deux trois quatre", "un deux") == 0.5


def make_take(tmp_path, cap_per_group: int = 200) -> Take:
    return Take(
        name="peertube-fr-train",
        builder=ManifestBuilder(tmp_path / "dataset", verbose=False),
        language="fr",
        source_split="listing",
        license="CC-BY",
        text_has_casing_punct=True,
        group_kind="recording",
        cap_per_group=cap_per_group,
    )


def answering_by_length(texts: dict[float, str]):
    """A fake clip model that answers by clip length, counting its calls."""
    calls = []

    def transcribe(wav: bytes) -> dict:
        seconds = round((len(wav) - 44) / 2 / SAMPLE_RATE, 2)
        calls.append(seconds)
        return {"text": texts[seconds]}

    return transcribe, calls


def make_talk(tmp_path, segments: list[Segment], seconds: float, **video_fields) -> Talk:
    path = write_tone(tmp_path / "talk.wav", seconds=seconds)
    return Talk(video(**video_fields), path, segments)


def test_only_clips_the_two_models_agree_on_are_kept(tmp_path):
    talk = make_talk(
        tmp_path,
        [
            segment("Bonjour à tous.", 0.0, 6.0),
            segment("On parle de Kubernetes.", 6.0, 13.0),
            segment("Oui.", 13.5, 13.8),
            segment("Une phrase beaucoup trop longue.", 15.0, 35.0),
        ],
        seconds=40.0,
    )
    qwen, calls = answering_by_length({6.0: "Bonjour à tous.", 7.0: "Il pleut sur Toulouse."})
    tally = Counter()

    agreed = peertube_train.agreeing(talk, ClipTexts(tmp_path / "q.jsonl"), qwen, 200, tally)

    assert [(clip.start_s, label) for clip, label in agreed] == [(0.0, "Bonjour à tous.")]
    assert calls == [6.0, 7.0], "a clip outside 0.5-19 s is not labelled"
    assert tally == Counter(
        {
            "clips labelled": 2,
            "clips where Whisper and Qwen disagree": 1,
            "clips outside 0.5-19 s": 2,
        }
    )
    assert "(50.0%) rejected" in peertube_train.report(tally)


def test_a_talk_stops_being_labelled_at_its_share_of_the_cap(tmp_path):
    talk = make_talk(
        tmp_path, [segment("Bonjour.", 6.0 * i, 6.0 * i + 5.0) for i in range(5)], seconds=30.0
    )
    qwen, calls = answering_by_length({5.0: "Bonjour."})

    agreed = peertube_train.agreeing(talk, ClipTexts(tmp_path / "q.jsonl"), qwen, 2, Counter())

    assert len(agreed) == 2 and len(calls) == 2


def test_a_kept_clip_carries_its_attribution_and_the_base_models_error(tmp_path):
    talk = make_talk(tmp_path, [], seconds=20.0, uuid="v1", licence_id=7)
    talk.agreed = [
        (segment("Bonjour à tous.", 0.0, 6.0), "Bonjour à tous."),
        (segment("On parle de Kubernetes.", 6.0, 10.0), "On parle de Kubernetes."),
    ]
    parakeet, _ = answering_by_length({6.0: "bonjour à tous", 4.0: "On parle de cube air net."})
    take = make_take(tmp_path)
    tally = Counter()

    kept = peertube_train.offer_talk(talk, ClipTexts(tmp_path / "p.jsonl"), parakeet, take, tally)
    take.builder.write()

    assert kept == 2
    first, second = read_manifest(tmp_path / "dataset")
    assert first.id == "peertube-fr-train:v1-0000000"
    assert first.text == "Bonjour à tous." and first.text_has_casing_punct
    assert first.group == "peertube-fr-train:v1" and first.group_kind == "recording"
    assert first.license == "CC0-1.0", "the license is the video's own"
    assert first.source_url == f"{INSTANCE}/videos/watch/v1"
    assert first.source_title == "Talk v1" and first.source_author == "Capitole du Libre"
    assert first.base_wer == 0.0, "casing and punctuation are not errors"
    assert second.base_wer == 0.75, "the base model is scored, never asked to agree"
    assert "gets 1 of the 2 kept clips wrong (50.0%)" in peertube_train.report(tally)


def test_clip_texts_are_cached_by_the_stretch_of_audio_they_were_heard_in(tmp_path):
    path = tmp_path / "texts.jsonl"
    audio = silence(5.0)
    qwen, calls = answering_by_length({2.0: "Bonjour.", 3.0: "Bonjour."})
    ClipTexts(path).text(segment(start=0.0, end=2.0), audio, qwen)
    ClipTexts(path).text(segment(start=0.0, end=2.0), audio, qwen)
    assert calls == [2.0], "a resumed preparation reads the text it already has"

    ClipTexts(path).text(segment(start=0.0, end=3.0), audio, qwen)
    assert calls == [2.0, 3.0], "a clip cut differently is transcribed again"


def test_segments_are_cached_under_the_model_and_the_chunk_length(tmp_path):
    asked = []

    def whisper(wav: bytes) -> dict:
        asked.append(wav)
        return {"segments": [whisper_row(0.0, 2.0, "bonjour")]}

    audio_path = write_tone(tmp_path / "talk.wav", seconds=3.0)
    path = tmp_path / peertube_train.segments_file(video())
    first = peertube_train.cached_segments(audio_path, whisper, path)
    again = peertube_train.cached_segments(audio_path, whisper, path)

    assert first == again == [segment("bonjour", 0.0, 2.0)] and len(asked) == 1
    assert "stt-c-whisper-large-v3" in path.name and "300s" in path.name


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


def test_fetch_sends_each_model_all_its_requests_before_the_next_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(peertube, "read_listing", lambda: ("2026-11-08", [video("a"), video("b")]))
    monkeypatch.setattr(
        peertube,
        "fetch_audio",
        lambda http, video, destination: write_tone(destination, seconds=12.0),
    )
    rows = [whisper_row(0.0, 5.0, "Bonjour."), whisper_row(6.0, 11.0, "Salut.")]
    answers = {
        peertube_train.SEGMENT_MODEL: lambda wav: {"segments": rows},
        peertube_train.LABEL_MODEL: lambda wav: {"text": "Bonjour."},
        peertube_train.BASE_MODEL: lambda wav: {"text": "Bonjour."},
    }
    log: list[str] = []
    monkeypatch.setattr(
        peertube_train,
        "_client",
        lambda model, request=None: FakeClient(model, answers[model], log),
    )

    count = peertube_train.fetch_peertube_fr_train(
        "peertube-fr-train", tmp_path / "cache", tmp_path / "dataset", verbose=True
    )

    assert count == 2, "one clip of each talk agrees"
    assert "2 talks (" in capsys.readouterr().out
    assert (
        log
        == ["stt-c-whisper-large-v3"] * 2
        + ["stt-a-qwen3-asr-1.7B"] * 4
        + ["stt-c-parakeet-tdt-0.6B-v3"] * 2
    ), "the models never alternate"
    clips = read_manifest(tmp_path / "dataset")
    assert [(clip.text, clip.base_wer) for clip in clips] == [("Bonjour.", 0.0)] * 2
    cache = tmp_path / "cache" / "peertube-fr-train"
    assert (cache / "a.stt-a-qwen3-asr-1.7B.jsonl").is_file()
    assert (cache / "b.stt-c-parakeet-tdt-0.6B-v3.jsonl").is_file()
