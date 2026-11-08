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
from audiolab.data.peertube_train import Labels, Segment  # noqa: E402
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


def qwen_answering(labels: dict[float, str]):
    """A fake label model that answers by clip length, counting its calls."""
    calls = []

    def transcribe(wav: bytes) -> dict:
        seconds = round((len(wav) - 44) / 2 / SAMPLE_RATE, 2)
        calls.append(seconds)
        return {"text": labels[seconds]}

    return transcribe, calls


def test_only_labels_that_agree_become_clips_with_the_video_attribution(tmp_path):
    segments = [
        segment("Bonjour à tous.", 0.0, 6.0),
        segment("On parle de Kubernetes.", 6.0, 13.0),
        segment("Oui.", 13.5, 13.8),
        segment("Une phrase beaucoup trop longue.", 15.0, 35.0),
    ]
    qwen, calls = qwen_answering({6.0: "Bonjour à tous.", 7.0: "Il pleut sur Toulouse."})
    take = make_take(tmp_path)
    left_out = Counter()
    labels = Labels(tmp_path / "labels.jsonl")

    kept = peertube_train.offer_video(
        video("v1", 7), silence(40.0), segments, labels, qwen, take, left_out
    )
    take.builder.write()

    assert kept == 1 and calls == [6.0, 7.0], "a clip outside 0.5-19 s is not labelled"
    assert left_out == Counter(
        {
            "clips labelled": 2,
            "clips where the two models disagree": 1,
            "clips outside 0.5-19 s": 2,
        }
    )
    (clip,) = read_manifest(tmp_path / "dataset")
    assert clip.id == "peertube-fr-train:v1-0000000"
    assert clip.text == "Bonjour à tous." and clip.text_has_casing_punct
    assert clip.group == "peertube-fr-train:v1" and clip.group_kind == "recording"
    assert clip.license == "CC0-1.0", "the license is the video's own"
    assert clip.source_url == f"{INSTANCE}/videos/watch/v1"
    assert clip.source_title == "Talk v1" and clip.source_author == "Capitole du Libre"
    assert "(50.0%) rejected" in peertube_train.report(left_out)


def test_a_video_stops_being_labelled_at_its_share_of_the_cap(tmp_path):
    segments = [segment("Bonjour.", 6.0 * i, 6.0 * i + 5.0) for i in range(5)]
    qwen, calls = qwen_answering({5.0: "Bonjour."})
    take = make_take(tmp_path, cap_per_group=2)
    labels = Labels(tmp_path / "labels.jsonl")

    kept = peertube_train.offer_video(
        video(), silence(30.0), segments, labels, qwen, take, Counter()
    )

    assert kept == 2 and len(calls) == 2


def test_labels_are_cached_by_the_stretch_of_audio_they_were_heard_in(tmp_path):
    path = tmp_path / "labels.jsonl"
    qwen, calls = qwen_answering({2.0: "Bonjour à tous.", 3.0: "Bonjour à tous."})
    first = [segment("bonjour à tous", 0.0, 2.0)]
    peertube_train.offer_video(
        video(), silence(5.0), first, Labels(path), qwen, make_take(tmp_path), Counter()
    )
    peertube_train.offer_video(
        video(), silence(5.0), first, Labels(path), qwen, make_take(tmp_path), Counter()
    )
    assert calls == [2.0], "a resumed preparation reads the label it already has"

    moved = [segment("bonjour à tous", 0.0, 3.0)]
    peertube_train.offer_video(
        video(), silence(5.0), moved, Labels(path), qwen, make_take(tmp_path), Counter()
    )
    assert calls == [2.0, 3.0], "a segment cut differently is labelled again"


def test_segments_are_cached_under_the_model_and_the_chunk_length(tmp_path):
    asked = []

    def whisper(wav: bytes) -> dict:
        asked.append(wav)
        return {"segments": [whisper_row(0.0, 2.0, "bonjour")]}

    path = tmp_path / peertube_train.segments_file(video())
    first = peertube_train.cached_segments(silence(3.0), whisper, path)
    again = peertube_train.cached_segments(silence(3.0), whisper, path)

    assert first == again == [segment("bonjour", 0.0, 2.0)] and len(asked) == 1
    assert "stt-c-whisper-large-v3" in path.name and "300s" in path.name


class FakeClient:
    """Stands in for a TranscriptionClient of one served model."""

    def __init__(self, answer):
        self.answer = answer

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def transcribe(self, wav: bytes, language: str):
        assert language == "fr"
        return type("Transcription", (), {"answer": self.answer(wav)})()


def test_fetch_prepares_the_listed_talks_into_a_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(peertube, "read_listing", lambda: ("2026-11-08", [video("a")]))
    monkeypatch.setattr(
        peertube,
        "fetch_audio",
        lambda http, video, destination: write_tone(destination, seconds=6.0),
    )
    whisper = FakeClient(
        lambda wav: {
            "segments": [whisper_row(0.0, 5.0, "Bonjour."), whisper_row(5.5, 6.0, "Salut.")]
        }
    )
    qwen = FakeClient(lambda wav: {"text": "Bonjour."})
    monkeypatch.setattr(
        peertube_train,
        "_client",
        lambda model, request=None: whisper if model == peertube_train.SEGMENT_MODEL else qwen,
    )

    count = peertube_train.fetch_peertube_fr_train(
        "peertube-fr-train", tmp_path / "cache", tmp_path / "dataset", verbose=False
    )

    assert count == 1
    assert [clip.text for clip in read_manifest(tmp_path / "dataset")] == ["Bonjour."]
    assert (tmp_path / "cache" / "peertube-fr-train" / "a.stt-a-qwen3-asr-1.7B.jsonl").is_file()
