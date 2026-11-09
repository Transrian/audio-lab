"""A talk source labelled and downloaded without a speech server or a network.

The three speech models are replaced by fakes, so these tests check which clips
are labelled, which labels are kept, what a manifest line carries, how talks are
batched, and that the models are asked one after the other. The catalogue is a
pinned listing written by the test itself: everything a source contributes is
that listing, the language, and where the audio of a talk comes from.
"""

from __future__ import annotations

import json
import sys
import threading
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import write_tone  # noqa: E402

from audiolab.core.audio import SAMPLE_RATE  # noqa: E402
from audiolab.data import talk_clips, talk_listing, talk_sources, talks  # noqa: E402
from audiolab.data.build import ManifestBuilder  # noqa: E402
from audiolab.data.download import DownloadError  # noqa: E402
from audiolab.data.store import read_manifest  # noqa: E402
from audiolab.data.talk_clips import Answers, Segment  # noqa: E402
from audiolab.data.talk_listing import Talk, TalkSource, write_listing  # noqa: E402
from audiolab.data.talks import TalkInProgress  # noqa: E402
from audiolab.data.train import Take  # noqa: E402
from audiolab.eval.client import TranscriptionError  # noqa: E402

WHISPER, QWEN, PARAKEET = (
    talks.SEGMENT_MODEL,
    talks.LABEL_MODEL,
    talks.BASE_MODEL,
)
INSTANCE = "https://tube.example"


def talk(talk_id: str = "v1", *, license: str = "CC-BY", channel: str = "Talks", **fields):
    values = {
        "id": talk_id,
        "url": f"{INSTANCE}/videos/watch/{talk_id}",
        "title": f"Talk {talk_id}",
        "duration_s": 600,
        "license": license,
        "author": "Capitole du Libre",
        "channel": channel,
        "instance": INSTANCE,
    }
    return Talk(**(values | fields))


def make_source(tmp_path, listed: list[Talk], **fields) -> TalkSource:
    """A talk source whose pinned listing holds ``listed``, written on 2026-11-08."""
    source = TalkSource(
        name=fields.pop("name", "test-fr-train"),
        language=fields.pop("language", "fr"),
        allowed_licenses=fields.pop("allowed_licenses", frozenset({"CC-BY", "CC0-1.0"})),
        listing_path=tmp_path / f"{fields.get('name', 'test')}-listing.json",
        listing_filter={"language": "fr"},
        lister=lambda http: listed,
        audio_url=fields.pop("audio_url", lambda http, one: one.url),
        **fields,
    )
    write_listing(source, "2026-11-08", listed)
    return source


def segment(text: str, start: float, end: float) -> Segment:
    return Segment(start_s=start, end_s=end, text=text)


def whisper_row(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": f" {text}"}


def make_take(tmp_path, language: str = "fr", cap: int = 12_000, cap_per_group: int = 200):
    return Take(
        name="test-fr-train",
        builder=ManifestBuilder(tmp_path / "dataset", verbose=False),
        language=language,
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


def make_pass(tmp_path, segments: list[Segment], seconds: float, **talk_fields) -> TalkInProgress:
    path = write_tone(tmp_path / "talk.wav", seconds=seconds)
    return TalkInProgress(talk(**talk_fields), path, segments)


def test_a_pinned_listing_keeps_its_date_its_filter_and_its_talks(tmp_path):
    source = make_source(tmp_path, [talk("a"), talk("b", license="CC0-1.0")])

    listed_on, listed = talk_listing.read_listing(source)

    assert listed_on == "2026-11-08"
    assert listed == [talk("a"), talk("b", license="CC0-1.0")]
    assert source.listing_filter == {"language": "fr"}, "the filter is recorded for a reader"


def test_a_listing_is_one_line_per_talk_with_what_they_share_written_once(tmp_path):
    listed = [talk("a"), talk("b", license="CC0-1.0", title="L'été", duration_s=900)]
    source = make_source(tmp_path, listed)
    written = source.listing_path.read_text(encoding="utf-8")
    listing = json.loads(written)

    assert listing["shared"] == {
        "author": "Capitole du Libre",
        "channel": "Talks",
        "instance": INSTANCE,
    }
    assert listing["fields"] == ["id", "url", "title", "duration_s", "license"]
    assert listing["talks"][1] == ["b", f"{INSTANCE}/videos/watch/b", "L'été", 900, "CC0-1.0"]
    rows = [line for line in written.splitlines() if line.startswith("  [")]
    assert len(rows) == len(listed), "a line per talk"
    assert talk_listing.read_listing(source)[1] == listed

    defaults = make_source(tmp_path, [talk("a", instance=""), talk("b", instance="")])
    assert "instance" not in json.loads(defaults.listing_path.read_text())["shared"], (
        "a default every talk shares is not written"
    )
    assert talk_listing.read_listing(defaults)[1][0].instance == ""


def test_a_listing_holding_another_licence_is_refused(tmp_path):
    source = make_source(tmp_path, [talk("a", license="CC-BY-SA-4.0")])

    with pytest.raises(DownloadError, match="another licence"):
        talk_listing.read_listing(source)


def test_the_reading_order_is_seeded_and_spreads_over_the_groups(tmp_path):
    listed = [talk(f"v{index:02d}", channel=f"c{index % 4}") for index in range(20)]
    order = talk_listing.reading_order(listed, seed=1, spread_by=lambda one: one.channel)

    spread = lambda one: one.channel  # noqa: E731 - the group of this source
    assert order == talk_listing.reading_order(list(reversed(listed)), seed=1, spread_by=spread)
    assert order != talk_listing.reading_order(listed, seed=2, spread_by=lambda one: one.channel)
    assert sorted(one.id for one in order) == sorted(one.id for one in listed)
    assert {one.channel for one in order[:4]} == {"c0", "c1", "c2", "c3"}, "one per group first"
    assert order[4] not in order[:4] and len({one.channel for one in order[:8]}) == 4
    assert talk_listing.reading_order(listed, seed=1) == talk_listing.reading_order(
        list(reversed(listed)), seed=1
    ), "without a spread it is one shuffle"


def test_only_the_first_talks_of_the_reading_order_are_prepared(tmp_path, monkeypatch):
    listed = [talk(f"v{index}", duration_s=1800) for index in range(10)]
    source = make_source(tmp_path, listed)
    order = [one.id for one in talk_listing.reading_order(listed, seed=1)]
    fetched: list[str] = []

    def fetch_audio(http, one, destination, source):
        fetched.append(one.id)
        if one.id == order[0]:
            raise DownloadError("HTTP 404")
        write_tone(destination, seconds=1.0)
        return destination

    monkeypatch.setattr(talks, "fetch_audio", fetch_audio)
    prepared = talks.prepare_audio(source, tmp_path / "cache", hours=1.5, verbose=False)

    assert fetched == order[:3], "1.5 h of 30-minute talks: three of them"
    assert (prepared.talks, prepared.failed, round(prepared.hours, 1)) == (2, 1, 1.0), (
        "the hours are the hours of the talks that arrived, not of the selection"
    )

    fetched.clear()
    monkeypatch.setattr(talks, "fetch_audio", lambda http, one, destination, source: destination)
    talks.prepare_audio(source, tmp_path / "cache", hours=1.5, verbose=False)
    assert fetched == [], "a second preparation finds the hours already prepared"


def test_fetch_audio_keeps_the_audio_and_deletes_the_video(tmp_path, monkeypatch):
    source_file = tmp_path / "remote.mp4"
    source_file.write_bytes(b"pixels and sound")
    source = make_source(tmp_path, [talk()], audio_url=lambda http, one: str(source_file))
    commands = []

    def fake_ffmpeg(command, **options):
        commands.append(command)
        assert Path(command[command.index("-i") + 1]).is_file()
        write_tone(Path(command[-1]), seconds=2.0)
        return type("Finished", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(talks.subprocess, "run", fake_ffmpeg)
    audio = talks.fetch_audio(None, talk(), tmp_path / "cache" / "v1.wav", source)

    assert audio.is_file() and audio.name == "v1.wav"
    assert sorted(one.name for one in audio.parent.iterdir()) == ["v1.wav"]
    assert ["-ac", "1", "-ar", "16000"] == commands[0][8:12]
    talks.fetch_audio(None, talk(), audio, source)
    assert len(commands) == 1, "an extracted talk is not downloaded again"


def test_fetch_audio_reports_ffmpeg_failing_and_leaves_nothing_behind(tmp_path, monkeypatch):
    source_file = tmp_path / "remote.mp4"
    source_file.write_bytes(b"pixels and sound")
    source = make_source(tmp_path, [talk()], audio_url=lambda http, one: str(source_file))
    failed = type("Finished", (), {"returncode": 1, "stderr": "no audio stream"})()
    monkeypatch.setattr(talks.subprocess, "run", lambda command, **options: failed)

    with pytest.raises(DownloadError, match="no audio stream"):
        talks.fetch_audio(None, talk(), tmp_path / "cache" / "v1.wav", source)
    assert list((tmp_path / "cache").iterdir()) == []


def test_agreement_ignores_casing_punctuation_and_how_numbers_are_written():
    assert talks.disagreement("bonjour à tous", "Bonjour à tous !", "fr") == 0.0
    assert talks.disagreement("en 2024", "En deux mille vingt-quatre.", "fr") == 0.0
    one_in_seven = talks.disagreement(
        "on installe le paquet avec npm install", "On installe le paquet avec NPM installe.", "fr"
    )
    assert one_in_seven == pytest.approx(1 / 7)
    assert one_in_seven <= talks.MAX_DISAGREEMENT
    assert talks.disagreement("un deux trois quatre", "un deux", "fr") == 0.5
    assert (
        talks.disagreement("the package is in the archive", "The package is in the archive.", "en")
        == 0.0
    )


def test_only_clips_the_two_models_agree_on_are_kept(tmp_path):
    one = make_pass(
        tmp_path,
        [
            segment("Bonjour à tous.", 0.0, 6.0),
            segment("On parle de Kubernetes.", 6.0, 13.0),
            segment("Oui.", 13.5, 13.55),
            segment("Une phrase beaucoup trop longue.", 15.0, 36.0),
        ],
        seconds=40.0,
    )
    # With their padding, the clips last 6.0 s, 7.2 s, 0.45 s and 21.0 s.
    qwen, calls = answering_by_length({6.0: "Bonjour à tous.", 7.2: "Il pleut sur Toulouse."})
    tally = Counter()

    agreed = talks.agreeing(one, Answers(tmp_path / "q.jsonl"), qwen, "fr", 200, tally)

    assert [(clip.start_s, label) for clip, label in agreed] == [(0.0, "Bonjour à tous.")]
    assert sorted(calls) == [6.0, 7.2], "a clip outside 0.5-20 s is not labelled"
    assert tally == Counter(
        {
            "clips labelled": 2,
            "clips where Whisper and Qwen disagree": 1,
            "clips outside 0.5-20 s": 2,
        }
    )
    assert "(50.0%) rejected" in talks.report(tally)


def test_a_clip_of_exactly_0_5_or_20_s_fits_whatever_the_float_subtraction_leaves():
    assert 32.02 - 12.02 > 20.0 and 0.57 - 0.07 < 0.5, "the float traps this guards against"
    assert talks.fits(segment("Une longue phrase.", 12.02, 32.02))
    assert talks.fits(segment("Oui.", 0.07, 0.57))
    assert not talks.fits(segment("Une longue phrase.", 12.02, 32.03))
    assert not talks.fits(segment("Oui.", 0.07, 0.56))


def test_a_failed_label_request_is_counted_and_asked_again_next_time(tmp_path):
    one = make_pass(
        tmp_path,
        [segment("Bonjour à tous.", 0.0, 6.0), segment("On parle de Kubernetes.", 7.0, 14.0)],
        seconds=20.0,
    )
    qwen, calls = answering_by_length({6.2: "Bonjour à tous."})
    tally = Counter()

    agreed = talks.agreeing(one, Answers(tmp_path / "q.jsonl"), qwen, "fr", 200, tally)
    talks.agreeing(one, Answers(tmp_path / "q.jsonl"), qwen, "fr", 200, Counter())

    assert len(agreed) == 1 and tally["clips Qwen failed on"] == 1
    assert sorted(calls) == [6.2, 7.4, 7.4], "the answer is cached, the failure is not"


def test_a_talk_stops_being_labelled_at_its_share_of_the_cap(tmp_path):
    one = make_pass(
        tmp_path,
        [segment(f"Phrase {i}.", 6.0 * i, 6.0 * i + 5.0) for i in range(5)],
        seconds=30.0,
    )
    qwen, calls = answering_by_length({5.2: "Phrase 0.", 5.4: "Phrase 1."})

    agreed = talks.agreeing(one, Answers(tmp_path / "q.jsonl"), qwen, "fr", 2, Counter())

    assert len(agreed) == 2 and len(calls) == 2


def test_a_talk_short_of_its_share_sends_only_the_clips_it_still_wants(tmp_path):
    one = make_pass(
        tmp_path,
        [segment(f"Phrase {i}.", 1.0 + 6.0 * i, 6.0 + 6.0 * i + i / 10) for i in range(6)],
        seconds=40.0,
    )
    # With their padding, the clips last 5.4 s, 5.5 s, ... 5.9 s; Qwen mishears the second.
    texts = {round(5.4 + i / 10, 2): f"Phrase {i}." for i in range(6)}
    texts[5.5] = "Rien."
    qwen, calls = answering_by_length(texts)

    agreed = talks.agreeing(one, Answers(tmp_path / "q.jsonl"), qwen, "fr", 3, Counter())

    assert [label for _, label in agreed] == ["Phrase 0.", "Phrase 2.", "Phrase 3."]
    assert sorted(calls[:3]) == [5.4, 5.5, 5.6] and calls[3:] == [5.7], (
        "three clips at once, then the one clip still wanted"
    )


def test_a_kept_clip_carries_its_attribution_and_the_base_models_error(tmp_path):
    one = make_pass(tmp_path, [], seconds=20.0, id="v1", license="CC0-1.0")
    one.agreed = [
        (segment("Bonjour à tous.", 0.0, 6.0), "Bonjour à tous."),
        (segment("On parle de Kubernetes.", 6.0, 10.0), "On parle de Kubernetes."),
        (segment("Merci.", 10.0, 13.0), "Merci."),
    ]
    parakeet, _ = answering_by_length({6.0: "bonjour à tous", 4.0: "On parle de cube air net."})
    take = make_take(tmp_path)
    tally = Counter()

    kept = talks.offer_talk(one, Answers(tmp_path / "p.jsonl"), parakeet, take, "fr", tally)
    take.builder.write()

    assert kept == 3
    first, second, third = read_manifest(tmp_path / "dataset")
    assert first.id == "test-fr-train:v1-0000000"
    assert first.text == "Bonjour à tous." and first.text_has_casing_punct
    assert first.group == "test-fr-train:v1" and first.group_kind == "recording"
    assert first.license == "CC0-1.0", "the license is the talk's own"
    assert first.source_url == f"{INSTANCE}/videos/watch/v1"
    assert first.source_title == "Talk v1" and first.source_author == "Capitole du Libre"
    assert first.base_wer == 0.0, "casing and punctuation are not errors"
    assert second.base_wer == 0.75, "the base model is scored, never asked to agree"
    assert third.base_wer is None, "a failed request leaves the error unknown, not the clip out"
    assert "gets 1 of the 3 kept clips wrong (33.3%)" in talks.report(tally)
    assert tally["kept clips Parakeet failed on (no base_wer)"] == 1


def test_a_talk_that_cannot_be_downloaded_or_heard_is_counted_and_skipped(tmp_path, monkeypatch):
    source = make_source(tmp_path, [talk()])

    def refuse(http, one, destination, source):
        raise DownloadError("HTTP 404")

    monkeypatch.setattr(talks, "fetch_audio", refuse)
    assert talks.audio_file(None, tmp_path, source, talk()) is None

    def failing(wav: bytes) -> dict:
        raise TranscriptionError("HTTP 500")

    tally = Counter()
    missing = talks.heard_talk(talk("a"), None, tmp_path, failing, tally)
    deaf = talks.heard_talk(
        talk("b"), write_tone(tmp_path / "b.wav", seconds=2.0), tmp_path, failing, tally
    )

    assert missing.segments == deaf.segments == []
    assert tally == Counter({"talks that could not be downloaded": 1, "talks Whisper failed on": 1})
    assert talks.agreeing(missing, Answers(tmp_path / "q.jsonl"), failing, "fr", 200, tally) == []


def test_the_first_batch_counts_on_the_expected_yield_and_later_ones_on_the_seen_one(tmp_path):
    take = make_take(tmp_path)
    assert talks.batch_size(take, talks_read=0) == 86, "12 000 clips at 140 a talk"

    take.kept = 11_000
    assert talks.batch_size(take, talks_read=88) == 11, "1 000 at 125 a talk, x 1.3"

    take.kept = 0
    assert talks.batch_size(take, talks_read=5) == 15_600, "no yield counts as one"


class FakeClient:
    """Stands in for a TranscriptionClient of one served model, logging each request."""

    def __init__(self, model: str, answer, log: list[str], language: str = "fr"):
        self.model, self.answer, self.log, self.language = model, answer, log, language

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def transcribe(self, wav: bytes, language: str):
        assert language == self.language, "a model is asked in the language of its source"
        self.log.append(self.model)
        return type("Transcription", (), {"answer": self.answer(wav)})()


def serve_fakes(monkeypatch, listed, answers: dict, log: list[str], language: str = "fr") -> None:
    def fetch_audio(http, one, destination, source):
        log.append(f"download {one.id}")
        return write_tone(destination, seconds=12.0)

    monkeypatch.setattr(talks, "fetch_audio", fetch_audio)
    monkeypatch.setattr(
        talks,
        "client",
        lambda model, request=None: FakeClient(model, answers[model], log, language),
    )


def test_each_batch_runs_the_three_passes_one_model_after_the_other(tmp_path, monkeypatch, capsys):
    rows = [whisper_row(0.0, 5.0, "Bonjour."), whisper_row(6.0, 11.0, "Salut.")]
    answers = {
        WHISPER: lambda wav: {"segments": rows, "text": ""},
        QWEN: lambda wav: {"text": "Bonjour."},
        PARAKEET: lambda wav: {"text": "Bonjour."},
    }
    log: list[str] = []
    source = make_source(tmp_path, [talk(uuid) for uuid in "abcd"])
    serve_fakes(monkeypatch, None, answers, log)
    # Each talk gives one clip; the first batch counts on two.
    monkeypatch.setattr(talks, "TRAIN_CLIP_CAP", 3)
    monkeypatch.setattr(talks, "EXPECTED_CLIPS_PER_TALK", 2)

    count = talks.fetch_talks(
        source, tmp_path / "cache", tmp_path / "dataset", license="CC-BY", verbose=True
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
    cache = tmp_path / "cache" / "test-fr-train"
    for model, count_of_talks in (
        (f"{WHISPER}.300s", 4),
        (f"{QWEN}.clips", 3),
        (f"{PARAKEET}.clips", 3),
    ):
        assert len(list(cache.glob(f"*.{model}.jsonl"))) == count_of_talks, (
            "one cache file per talk"
        )
    assert "2 talks (" in capsys.readouterr().out


def test_an_english_source_is_labelled_in_english_and_says_so_in_its_clips(tmp_path, monkeypatch):
    rows = [whisper_row(0.0, 5.0, "Hello everyone."), whisper_row(6.0, 11.0, "Bye.")]
    answers = {
        WHISPER: lambda wav: {"segments": rows, "text": ""},
        QWEN: lambda wav: {"text": "Hello everyone."},
        PARAKEET: lambda wav: {"text": "Hello everyone."},
    }
    log: list[str] = []
    source = make_source(
        tmp_path, [talk(uuid) for uuid in "abc"], name="test-en-train", language="en"
    )
    serve_fakes(monkeypatch, None, answers, log, language="en")
    monkeypatch.setattr(talks, "TRAIN_CLIP_CAP", 1)
    monkeypatch.setattr(talks, "EXPECTED_CLIPS_PER_TALK", 1)

    talks.fetch_talks(
        source, tmp_path / "cache", tmp_path / "dataset", license="CC-BY", verbose=False
    )

    clip = read_manifest(tmp_path / "dataset")[0]
    assert clip.language == "en"
    assert log.count(WHISPER) == log.count(QWEN) == log.count(PARAKEET) == 1
    assert not list((tmp_path / "dataset").glob("*.fr*"))


def test_the_names_peertube_fr_train_is_cached_and_labelled_under_do_not_change(
    tmp_path, monkeypatch
):
    """The caches of a preparation already made are found again only under these names.

    A change to the listing, its reading order, a cache file name or the clip id
    would make the next preparation of ``peertube-fr-train`` ask the models again
    for talks already labelled, or label other talks than the ones it started on.
    """
    source = talk_sources.SOURCES["peertube-fr-train"]
    listed_on, listed = talk_listing.read_listing(source)
    order = talk_listing.reading_order(listed, talks.SEED, source.spread_by)
    assert (listed_on, len(listed)) == ("2026-11-08", 519)
    assert [one.id for one in order[:3]] == [
        "6a2bdc8c-970e-465f-bdc7-fe01c74de834",
        "8950c782-ae34-4556-9693-f7acdd7f4161",
        "8f69da0b-2d13-4f28-85df-6a174cc1a891",
    ]

    uuid = order[0].id
    answers = {
        WHISPER: lambda wav: {"segments": [whisper_row(0.5, 5.0, "Bonjour.")], "text": ""},
        QWEN: lambda wav: {"text": "Bonjour."},
        PARAKEET: lambda wav: {"text": "Bonjour."},
    }
    serve_fakes(monkeypatch, None, answers, [])
    monkeypatch.setattr(talks, "TRAIN_CLIP_CAP", 1)
    pinned = make_source(tmp_path, [talk(uuid)], name="peertube-fr-train")
    talks.fetch_talks(
        pinned, tmp_path / "cache", tmp_path / "dataset", license="CC-BY", verbose=False
    )

    cached = sorted(path.name for path in (tmp_path / "cache" / "peertube-fr-train").iterdir())
    assert cached == [
        f"{uuid}.stt-c-parakeet-tdt-0.6B-v3.clips.jsonl",
        f"{uuid}.stt-v-qwen3-asr-1.7B.clips.jsonl",
        f"{uuid}.stt-v-whisper-large-v3.300s.jsonl",
        f"{uuid}.wav",
    ]
    # The clip's start in hundredths of a second, on seven digits: Whisper said
    # 0.5 s, and the cut starts a little before what was heard.
    assert [clip.id for clip in read_manifest(tmp_path / "dataset")] == [
        f"peertube-fr-train:{uuid}-0000030"
    ]


def test_a_talk_is_heard_while_the_next_one_downloads(tmp_path, monkeypatch):
    whisper_asked = threading.Event()

    def empty_answer(wav: bytes) -> dict:
        whisper_asked.set()
        return {"segments": [], "text": ""}

    answers = {WHISPER: empty_answer, QWEN: empty_answer, PARAKEET: empty_answer}
    log: list[str] = []
    source = make_source(tmp_path, [talk(uuid) for uuid in "abc"])
    serve_fakes(monkeypatch, None, answers, log)
    fetch_audio = talks.fetch_audio
    waited: list[bool] = []

    def third_after_whisper(http, one, destination, source):
        if one.id == "c":
            waited.append(whisper_asked.wait(timeout=10))
        return fetch_audio(http, one, destination, source)

    monkeypatch.setattr(talks, "fetch_audio", third_after_whisper)
    talks.fetch_talks(
        source, tmp_path / "cache", tmp_path / "dataset", license="CC-BY", verbose=False
    )

    assert waited == [True], "only the next talk downloads before Whisper starts, not the batch"


def test_a_pass_sends_requests_in_parallel_and_reads_the_answers_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(talks, "PARALLEL_REQUESTS", 3)
    # Three requests must be in flight together for any of them to be answered.
    together = threading.Barrier(3, timeout=10)
    one = make_pass(
        tmp_path,
        [segment(f"Phrase {i}.", 1.0 + 6.0 * i, 6.0 + 6.0 * i + i / 10) for i in range(6)],
        seconds=40.0,
    )

    # With their padding, the clips last 5.4 s, 5.5 s, ... 5.9 s; Qwen mishears the fourth.
    texts = {round(5.4 + i / 10, 2): f"Phrase {i}." for i in range(6)}
    texts[5.7] = "Rien."
    by_length, _ = answering_by_length(texts)

    def qwen(wav: bytes) -> dict:
        together.wait()
        return by_length(wav)

    agreed = talks.agreeing(one, Answers(tmp_path / "q.jsonl"), qwen, "fr", 200, Counter())

    assert [label for _, label in agreed] == [f"Phrase {i}." for i in (0, 1, 2, 4, 5)]
    assert len(Answers(tmp_path / "q.jsonl").answers) == 6, "every answer is cached, whole"


def test_the_whisper_pass_hears_talks_at_once_and_returns_them_in_batch_order(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(talks, "PARALLEL_REQUESTS", 2)
    # Both talks must be in flight together for either of them to be answered.
    together = threading.Barrier(2, timeout=10)
    short_one_failed = threading.Event()

    def whisper(wav: bytes) -> dict:
        together.wait()
        if (len(wav) - 44) / 2 / SAMPLE_RATE < 10.0:
            short_one_failed.set()
            raise TranscriptionError("HTTP 500")
        assert short_one_failed.wait(timeout=10), "the first talk is answered last"
        return {"segments": [whisper_row(0.0, 5.0, "Bonjour.")], "text": ""}

    batch = [
        (talk("a"), None),
        (talk("b"), write_tone(tmp_path / "b.wav", seconds=12.0)),
        (talk("c"), write_tone(tmp_path / "c.wav", seconds=8.0)),
    ]
    tally = Counter({"talks that could not be downloaded": 2})

    heard = talks.heard_talks(batch, tmp_path, whisper, tally)

    assert [one.talk.id for one in heard] == ["a", "b", "c"]
    assert heard[1].segments == [segment("Bonjour.", 0.0, 5.0)], "a failing talk spoils no other"
    assert heard[2].segments == [] and heard[2].audio_path is not None
    assert tally == Counter({"talks that could not be downloaded": 3, "talks Whisper failed on": 1})


def test_a_stopped_whisper_pass_stops_the_talks_in_flight_at_their_next_chunk(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(talk_clips, "CHUNK_SECONDS", 2.0)  # a talk of 10 s is five chunks
    in_parallel = talks.in_parallel
    stopping: list[threading.Event] = []

    def watched(ask, items, stop):
        stopping.append(stop)
        return in_parallel(ask, items, stop)

    monkeypatch.setattr(talks, "in_parallel", watched)
    whisper_asked = threading.Event()
    calls = []

    def whisper(wav: bytes) -> dict:
        calls.append(wav)
        whisper_asked.set()
        assert stopping[0].wait(timeout=10), "the first chunk is answered once the run stops"
        return {"segments": [], "text": ""}

    def batch():
        yield talk("a"), write_tone(tmp_path / "a.wav", seconds=10.0)
        whisper_asked.wait(timeout=10)
        raise KeyboardInterrupt  # Ctrl-C while the talk is heard

    with pytest.raises(KeyboardInterrupt):
        talks.heard_talks(batch(), tmp_path, whisper, Counter())

    assert len(calls) == 1, "the talk in flight stops at its next chunk"
    assert len(Answers(tmp_path / f"a.{WHISPER}.300s.jsonl").answers) == 1, (
        "the answer already written stays cached"
    )


def test_answers_written_from_several_threads_are_each_cached_whole(tmp_path):
    answers = Answers(tmp_path / "answers.jsonl")
    keys = [f"{index}.00-{index + 1}.00" for index in range(200)]
    text = "mot " * 2000

    talks.in_parallel(lambda key: answers.answer(key, lambda: {"text": text}), keys)

    assert Answers(tmp_path / "answers.jsonl").answers == {key: {"text": text} for key in keys}
    lines = (tmp_path / "answers.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 200, "no line is torn or written over another"
