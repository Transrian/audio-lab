"""The human-caption set of the education ministry, prepared without a network.

The captions scan, the WebVTT parsing, the cut of the cues into clips, and the
WER gate against Whisper are checked against fakes: a listing the test writes,
a cached scan the test feeds, and a Whisper that answers by the length of the
clip it is sent. The only live behaviour is left to the scan itself, which one
test checks against a fake of the API's JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import make_clip, write_tone  # noqa: E402

from audiolab.core.audio import SAMPLE_RATE  # noqa: E402
from audiolab.data import peertube_edu_captions as captions  # noqa: E402
from audiolab.data import talks  # noqa: E402
from audiolab.data.download import DownloadError  # noqa: E402
from audiolab.data.store import read_manifest  # noqa: E402
from audiolab.data.talk_clips import Segment  # noqa: E402
from audiolab.data.talk_listing import Talk, TalkSource, write_listing  # noqa: E402
from audiolab.eval.client import ModelConfig, TranscriptionError  # noqa: E402

INSTANCE_A = "https://tube-cycle-2.apps.education.fr"
INSTANCE_B = "https://tube-maternelle.apps.education.fr"


def talk(
    talk_id: str,
    *,
    license: str = "CC-BY",
    instance: str = INSTANCE_A,
    title: str | None = None,
    duration_s: int = 2000,
    **fields,
) -> Talk:
    values = {
        "id": talk_id,
        "url": f"{instance}/videos/watch/{talk_id}",
        "title": title or f"Talk {talk_id}",
        "duration_s": duration_s,
        "license": license,
        "author": "un-equipe",
        "channel": "Canal du test",
        "instance": instance,
    }
    return Talk(**(values | fields))


def make_source(tmp_path, listed: list[Talk], **fields) -> TalkSource:
    """A talk source whose pinned listing holds ``listed``, written on 2026-11-08."""
    source = TalkSource(
        name=fields.pop("name", captions.NAME),
        language=fields.pop("language", "fr"),
        allowed_licenses=fields.pop("allowed_licenses", frozenset({"CC-BY", "CC0-1.0"})),
        listing_path=tmp_path / f"{fields.get('name', 'captions')}-listing.json",
        listing_filter={"language": "fr"},
        lister=lambda http: listed,
        audio_url=lambda http, one: one.url,
        spread_by=lambda one: one.instance,
        **fields,
    )
    write_listing(source, "2026-11-08", listed)
    return source


def cue(start: float, end: float, text: str) -> Segment:
    return Segment(start_s=start, end_s=end, text=text)


class FakeClient:
    """Stands in for a TranscriptionClient of Whisper, logging each request."""

    def __init__(self, answer, log: list[str]):
        self.answer, self.log = answer, log
        self.config = ModelConfig(name=captions.SEGMENT_MODEL, model=captions.SEGMENT_MODEL)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def warm_up(self):
        self.log.append("warm-up whisper")

    def transcribe(self, wav: bytes, language: str):
        assert language == "fr", "the model is asked in the language of the set"
        self.log.append("whisper")
        return type("Transcription", (), {"answer": self.answer(wav)})()


def whisper_by_length(texts: dict[float, str]):
    """A fake Whisper that answers by the length of the clip, counting its calls."""
    calls: list[float] = []

    def answer(wav: bytes) -> dict:
        seconds = round((len(wav) - 44) / 2 / SAMPLE_RATE, 2)
        calls.append(seconds)
        if seconds not in texts:
            raise TranscriptionError("HTTP 500")
        return {"text": texts[seconds]}

    return answer, calls


def test_parse_cues_reads_both_timestamp_forms_joins_lines_and_drops_tags(tmp_path):
    vtt = """WEBVTT

NOTE
a note is not a cue, and neither is its next line

1
00:01.000 --> 00:02.500
Bonjour <i>tout le</i> monde.

2
00:02,500 --> 00:03.000
second line of the cue

00:12:04.000 --> 00:12:06.250
an hour, a minute, a second and a millisecond
"""
    cues = captions.parse_cues(vtt)

    assert [(c.start_s, c.end_s, c.text) for c in cues] == [
        (1.0, 2.5, "Bonjour tout le monde."),
        (2.5, 3.0, "second line of the cue"),
        (724.0, 726.25, "an hour, a minute, a second and a millisecond"),
    ]


def test_parse_cues_decodes_the_cue_entities_only_when_asked():
    # The ministry's files write their spaces as &nbsp;: the default reading keeps
    # the text as written, the decoded reading gives back the character named.
    vtt = "WEBVTT\n\n00:00.000 --> 00:01.000\n5&nbsp;x 2 &amp; 3&lt;4.\n"

    assert captions.parse_cues(vtt)[0].text == "5&nbsp;x 2 &amp; 3&lt;4."
    assert captions.parse_cues(vtt, unescape=True)[0].text == "5 x 2 & 3<4."

    # Decoding happens before the markup is dropped, so an entity and a tag in
    # the same cue both give back what they name or mark.
    vtt = "WEBVTT\n\n00:00.000 --> 00:01.000\nmot&nbsp;du <b>test</b>\n"

    assert captions.parse_cues(vtt, unescape=True)[0].text == "mot du test"


def test_clips_merge_cues_at_sentence_ends_and_gaps():
    cues = [
        cue(0.0, 2.0, "Bonjour."),
        cue(2.0, 4.5, "tout le monde."),
        cue(8.0, 10.0, "On commence"),
        cue(10.0, 12.0, "le cours."),
        cue(30.0, 31.5, "Ok."),
    ]
    clips, left_out = captions.clips_from_cues(cues)

    assert [(c.start_s, c.end_s, c.text) for c in clips] == [
        (0.0, 4.5, "Bonjour. tout le monde."),
        (8.0, 12.0, "On commence le cours."),
    ]
    assert dict(left_out) == {"a run of cues shorter than 3 s": 1}


def test_a_run_without_a_sentence_end_closes_at_the_ceiling():
    cues = [
        cue(0.0, 4.0, "un mot"),
        cue(4.0, 8.0, "un autre"),
        cue(8.0, 12.0, "encore un"),
        cue(12.0, 16.0, "et encore"),
    ]
    clips, left_out = captions.clips_from_cues(cues)

    # The first three cues reach 12 s without a full stop; the fourth would push
    # the run to 16, so the clip ends where it can and the fourth cue starts its
    # own run, which reaches the floor on its own.
    assert [(c.start_s, c.end_s, c.text) for c in clips] == [
        (0.0, 12.0, "un mot un autre encore un"),
        (12.0, 16.0, "et encore"),
    ]
    assert not left_out


def test_a_single_cue_longer_than_the_ceiling_is_left_out():
    clips, left_out = captions.clips_from_cues([cue(0.0, 20.0, "une très longue phrase")])

    assert clips == []
    assert dict(left_out) == {"a cue longer than 15 s": 1}


def test_only_a_human_french_caption_is_kept():
    human = {
        "language": {"id": "fr", "label": "French"},
        "automaticallyGenerated": False,
        "captionPath": "/vtt/a-fr.vtt",
    }
    machine = dict(human, automaticallyGenerated=True)
    english = dict(human, language={"id": "en", "label": "English"})
    unmarked = {"language": {"id": "fr", "label": "French"}, "captionPath": "/vtt/b-fr.vtt"}

    assert captions.human_french_caption([machine]) is None
    assert captions.human_french_caption([english]) is None
    assert captions.human_french_caption([unmarked]) is None
    assert captions.human_french_caption([machine, human]) == human


def test_the_scan_caches_each_caption_list_and_never_asks_twice(tmp_path, monkeypatch):
    asked = []

    def get_json(http, url):
        asked.append(url)
        return {"data": [{"language": {"id": "fr"}, "automaticallyGenerated": True}]}

    monkeypatch.setattr(captions.peertube, "get_json", get_json)
    talks_list = [talk("t-one"), talk("t-two", instance=INSTANCE_B)]
    cache = tmp_path / "cache"

    found = captions.scan_captions(http=None, talks=talks_list, cache=cache, verbose=False)
    # The instances are read in parallel, so the order of the requests is not a
    # contract; the contract is that every video is asked exactly once.
    assert sorted(asked) == [
        f"{INSTANCE_A}/api/v1/videos/t-one/captions",
        f"{INSTANCE_B}/api/v1/videos/t-two/captions",
    ]
    cached = json.loads((cache / "captions" / "t-one.json").read_text(encoding="utf-8"))
    assert found["t-one"] == cached["data"]
    assert (cache / "captions" / "t-two.json").is_file()

    asked.clear()
    found = captions.scan_captions(http=None, talks=talks_list, cache=cache, verbose=False)
    assert not asked, "a cached list is never asked again"
    assert found["t-one"] == cached["data"]


def test_the_vtt_download_retries_a_refused_request(tmp_path, monkeypatch):
    class Response:
        def __init__(self, status: int, content: bytes = b"", headers: dict | None = None):
            self.status_code, self.content, self.headers = status, content, headers or {}

    class Http:
        def __init__(self):
            self.calls = 0

        def get(self, url: str) -> Response:
            self.calls += 1
            if self.calls == 1:
                return Response(429, headers={"retry-after": "0"})
            return Response(200, content=b"WEBVTT\n\n00:00.000 --> 00:01.000\nSalut.\n")

    monkeypatch.setattr(captions.peertube, "PAUSE_S", 0.0)
    http = Http()
    destination = tmp_path / "vtt" / "t-one.fr.vtt"

    captions._get_vtt(http, f"{INSTANCE_A}/vtt/t-one-fr.vtt", destination)

    assert http.calls == 2
    assert "Salut." in destination.read_text(encoding="utf-8")

    def refuse(url: str) -> Response:
        return Response(404)

    with pytest.raises(DownloadError, match="HTTP 404"):
        refusing = type("Http", (), {"get": staticmethod(refuse)})()
        captions._get_vtt(refusing, f"{INSTANCE_A}/x", tmp_path / "y.vtt")


def test_the_fetcher_stops_when_too_few_talks_hold_human_captions(tmp_path, monkeypatch, capsys):
    source = make_source(tmp_path, [talk(f"t-{i}") for i in range(30)])
    monkeypatch.setattr(captions, "SOURCE", source)
    # Only ten of the thirty hold a human French caption: the set cannot measure.
    monkeypatch.setattr(
        captions,
        "scan_captions",
        lambda http, talks_list, cache, verbose=False: {
            talk.id: [{"language": {"id": "fr"}, "automaticallyGenerated": False}] if i < 10 else []
            for i, talk in enumerate(talks_list)
        },
    )

    with pytest.raises(DownloadError, match="only 10 talks"):
        captions.fetch_peertube_edu_captions_fr(
            captions.NAME, tmp_path / "cache", tmp_path / "dataset", verbose=True
        )

    assert not (tmp_path / "dataset").exists()
    out = capsys.readouterr().out
    assert "10 talks with human French captions" in out


def test_the_fetcher_skips_the_train_talks_and_gates_the_clips_with_whisper(
    tmp_path, monkeypatch, capsys
):
    # Talk "t-two" is in the prepared training set: its clips may not measure.
    train_dir = tmp_path / "peertube-edu-fr-train"
    train_dir.mkdir()
    (train_dir / "manifest.jsonl").write_text(
        make_clip(
            "peertube-edu-fr-train:t-two", language="fr", group="peertube-edu-fr-train:t-two"
        ).manifest_line()
        + "\n",
        encoding="utf-8",
    )
    listed = [
        talk("t-one", title="Cours de Python pour tous"),
        talk("t-two", title="L'évaluation en classe"),
    ]
    source = make_source(tmp_path, listed)
    monkeypatch.setattr(captions, "SOURCE", source)
    monkeypatch.setattr(captions, "MIN_TALKS", 2)
    captions_of = {
        "t-one": {"language": {"id": "fr"}, "automaticallyGenerated": False},
        "t-two": {"language": {"id": "fr"}, "automaticallyGenerated": False},
    }
    monkeypatch.setattr(
        captions,
        "scan_captions",
        lambda http, talks_list, cache, verbose=False: {
            t.id: [captions_of[t.id]] for t in talks_list
        },
    )
    cues_of = {
        "t-one": [
            cue(0.0, 3.0, "Bonjour."),
            cue(3.0, 7.5, "tout le monde."),
            cue(7.5, 11.5, "On commence le cours."),
            cue(18.0, 23.0, "Et maintenant, un exemple."),
            cue(23.0, 25.0, "Regardez."),
            cue(25.0, 26.5, "le code."),
        ],
        "t-two": [cue(0.0, 3.0, "Bonjour.")],
    }
    monkeypatch.setattr(
        captions,
        "fetch_captions",
        lambda http, one, caption, cache, unescape=False: (
            cache / "vtt" / f"{one.id}.fr.vtt",
            cues_of[one.id],
        ),
    )

    def audio_file(http, cache, source_, one):
        return write_tone(cache / f"{one.id}.wav", seconds=60.0)

    monkeypatch.setattr(captions, "audio_file", audio_file)

    log: list[str] = []
    heard = whisper_by_length(
        {
            7.5: "bonjour tout le monde",
            4.0: "on commence le cours",
            7.0: "le code est tout faux",  # the caption and the hearing do not meet
        }
    )
    monkeypatch.setattr(captions, "client", lambda model, request=None: FakeClient(heard[0], log))
    monkeypatch.delenv("AUDIOLAB_CONCURRENCY", raising=False)

    count = captions.fetch_peertube_edu_captions_fr(
        captions.NAME, tmp_path / "cache", tmp_path / "dataset", verbose=True
    )

    assert count == 2, "three candidate clips, one the gate leaves out"
    clips = read_manifest(tmp_path / "dataset")
    assert [(c.text, c.category, c.license) for c in clips] == [
        ("Bonjour. tout le monde.", "tech", "CC-BY"),
        ("On commence le cours.", "tech", "CC-BY"),
    ]
    assert all(c.group == f"{captions.NAME}:t-one" for c in clips)
    assert all(c.base_wer is None for c in clips), "no model wrote the label: no base_wer"
    # Whisper was asked once per candidate clip of the one kept talk, never for
    # the talk the training set holds.
    assert log == ["warm-up whisper", "whisper", "whisper", "whisper"]
    wers_path = tmp_path / "cache" / captions.NAME / "alignment-wers.json"
    wers = json.loads(wers_path.read_text(encoding="utf-8"))
    assert [round(wer, 2) for _, wer in wers["wers"]] == [0.0, 0.0, 1.0]
    out = capsys.readouterr().out
    assert "1 of the 2 usable talks are in peertube-edu-fr-train and are left out" in out
    assert "deciles" in out and "2 of 3 clips at or under 0.35" in out
    assert "cues left out: a run of cues shorter than 3 s" in out


def test_a_preparation_of_the_talk_source_skips_the_talks_of_an_evaluation_set(
    tmp_path, monkeypatch, capsys
):
    # The shared talk pipeline: a source that feeds an evaluation set leaves
    # its talks out of every preparation, before any download.
    from dataclasses import replace

    eval_dir = tmp_path / "fake-eval"
    eval_dir.mkdir()
    (eval_dir / "manifest.jsonl").write_text(
        make_clip("fake-eval:b", language="fr", group="fake-eval:b").manifest_line() + "\n",
        encoding="utf-8",
    )
    source = replace(
        make_source(tmp_path, [talk("a"), talk("b")], name="test-fr-train"),
        evaluation_datasets=("fake-eval",),
    )
    serve_fakes(monkeypatch)
    monkeypatch.setattr(talks, "TRAIN_CLIP_CAP", 100)
    asked_downloads: list[str] = []

    def fetch_audio(http, one, destination, source_):
        asked_downloads.append(one.id)
        return write_tone(destination, seconds=12.0)

    monkeypatch.setattr(talks, "fetch_audio", fetch_audio)

    count = talks.fetch_talks(
        source, tmp_path / "cache", tmp_path / "dataset", license="CC-BY", verbose=True
    )

    assert count >= 1
    clips = read_manifest(tmp_path / "dataset")
    assert clips and all(c.group == "test-fr-train:a" for c in clips)
    assert asked_downloads == ["a"], "the reserved talk is not even downloaded"
    assert "1 talks of this listing are left out: 1 evaluation talk" in capsys.readouterr().out


def serve_fakes(monkeypatch) -> None:
    """The three labelling models of the talk pipeline, as fakes, one at a time."""
    monkeypatch.delenv("AUDIOLAB_CONCURRENCY", raising=False)
    rows = [
        {"start": 0.0, "end": 5.0, "text": " Bonjour."},
        {"start": 6.0, "end": 11.0, "text": " Salut."},
    ]
    answers = {
        talks.SEGMENT_MODEL: lambda wav: {"segments": rows, "text": ""},
        talks.LABEL_MODEL: lambda wav: {"text": "Bonjour."},
        talks.BASE_MODEL: lambda wav: {"text": "Bonjour."},
    }
    log: list[str] = []
    monkeypatch.setattr(
        talks, "client", lambda model, request=None: FakeClient(answers[model], log)
    )


def test_the_decoded_set_is_registered_and_kept_out_of_the_talks():
    from audiolab.data.datasets import DATASETS
    from audiolab.data.fetch import FETCHERS
    from audiolab.data.talk_sources import SOURCES

    # The decoded set is a dataset of its own, sharing the source set's cache.
    info = DATASETS["peertube-edu-captions-fr-v2"]
    assert info.cache_name == captions.NAME
    assert FETCHERS["peertube-edu-captions-fr-v2"] is captions.fetch_peertube_edu_captions_fr_v2
    # Every talk source that leaves the source set's talks out leaves these out
    # as well: a clip of a talk may not measure a model trained on it.
    for name in ("peertube-edu-fr-train", "peertube-tech-fr-train"):
        assert captions.NAME in SOURCES[name].evaluation_datasets
        assert "peertube-edu-captions-fr-v2" in SOURCES[name].evaluation_datasets


def test_the_decoded_preparation_writes_its_own_gate_and_keeps_the_source_gate(
    tmp_path, monkeypatch
):
    # The decoded preparation reads the source's cache: the caption files and
    # the Whisper answers. Its gate goes into its own file, beside the
    # source's, which it never touches.
    train_dir = tmp_path / "peertube-edu-fr-train"
    train_dir.mkdir()
    (train_dir / "manifest.jsonl").write_text(
        make_clip(
            "peertube-edu-fr-train:t-two", language="fr", group="peertube-edu-fr-train:t-two"
        ).manifest_line()
        + "\n",
        encoding="utf-8",
    )
    source = make_source(tmp_path, [talk("t-one"), talk("t-two")])
    monkeypatch.setattr(captions, "SOURCE", source)
    monkeypatch.setattr(captions, "MIN_TALKS", 2)
    captions_of = {
        "t-one": {"language": {"id": "fr"}, "automaticallyGenerated": False},
        "t-two": {"language": {"id": "fr"}, "automaticallyGenerated": False},
    }
    monkeypatch.setattr(
        captions,
        "scan_captions",
        lambda http, talks_list, cache, verbose=False: {
            t.id: [captions_of[t.id]] for t in talks_list
        },
    )
    # The cached caption files hold the entities as the instance served them;
    # the preparation decodes them, it does not rewrite the files.
    vtt_dir = tmp_path / "cache" / captions.NAME / "vtt"
    vtt_dir.mkdir(parents=True)
    (vtt_dir / "t-one.fr.vtt").write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:05.000\n"
        "5&nbsp;x 2.\n\n"
        "00:00:07.000 --> 00:00:10.000\n"
        "Le code est tout faux.\n",
        encoding="utf-8",
    )
    (vtt_dir / "t-two.fr.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:03.000\nBonjour.\n", encoding="utf-8"
    )

    def audio_file(http, cache, source_, one):
        return write_tone(cache / f"{one.id}.wav", seconds=60.0)

    monkeypatch.setattr(captions, "audio_file", audio_file)

    heard = whisper_by_length({5.0: "5 x 2.", 3.0: "bonjour"})
    monkeypatch.setattr(captions, "client", lambda model, request=None: FakeClient(heard[0], []))
    monkeypatch.delenv("AUDIOLAB_CONCURRENCY", raising=False)

    # The source set's gate, as an earlier run wrote it: bytes the decoded
    # preparation must not change.
    source_gate = tmp_path / "cache" / captions.NAME / "alignment-wers.json"
    source_gate.write_text('{"threshold": 0.35, "wers": [["old", 0.0]]}\n', encoding="utf-8")

    count = captions.fetch_peertube_edu_captions_fr_v2(
        "peertube-edu-captions-fr-v2", tmp_path / "cache", tmp_path / "dataset", verbose=False
    )

    assert count == 1, "the decoded clip passes; the unheard one does not"
    clips = read_manifest(tmp_path / "dataset")
    assert [c.text for c in clips] == ["5 x 2."], "the label is the decoded text"
    # The source's gate is byte for byte the earlier run's; the decoded
    # preparation's gate is its own file with its own clips.
    assert (
        source_gate.read_text(encoding="utf-8") == '{"threshold": 0.35, "wers": [["old", 0.0]]}\n'
    )
    gate = json.loads(
        (tmp_path / "cache" / captions.NAME / "alignment-wers-v2.json").read_text(encoding="utf-8")
    )
    assert gate["threshold"] == captions.MAX_CLIP_WER
    assert gate["wers"] == [["t-one-0000000", 0.0], ["t-one-0000700", 1.0]]


def test_the_dry_run_counts_the_clips_a_decoded_preparation_would_ask(tmp_path, monkeypatch):
    # The dry run reads the source's cache only: the caption scan, the caption
    # files, the Whisper answers. Its count is the number of clips a decoded
    # preparation would have to ask.
    data_root = tmp_path / "data"
    train_dir = data_root / "peertube-edu-fr-train"
    train_dir.mkdir(parents=True)
    (train_dir / "manifest.jsonl").write_text(
        make_clip(
            "peertube-edu-fr-train:t-two", language="fr", group="peertube-edu-fr-train:t-two"
        ).manifest_line()
        + "\n",
        encoding="utf-8",
    )
    source = make_source(tmp_path, [talk("t-one"), talk("t-two"), talk("t-three")])
    monkeypatch.setattr(captions, "SOURCE", source)
    cache_dir = tmp_path / "cache"
    scan_dir = cache_dir / captions.NAME / "captions"
    scan_dir.mkdir(parents=True)
    human = {"language": {"id": "fr"}, "automaticallyGenerated": False}
    for talk_id in ("t-one", "t-two", "t-three"):
        (scan_dir / f"{talk_id}.json").write_text(json.dumps({"data": [human]}), encoding="utf-8")
    vtt_dir = cache_dir / captions.NAME / "vtt"
    vtt_dir.mkdir()
    (vtt_dir / "t-one.fr.vtt").write_text(
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:05.000\n"
        "5&nbsp;x 2.\n\n"
        "00:00:07.000 --> 00:00:10.000\n"
        "Le code est tout faux.\n",
        encoding="utf-8",
    )
    (vtt_dir / "t-three.fr.vtt").write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nBonjour.\n", encoding="utf-8"
    )
    answers_of = {
        "t-one": [{"key": "0.00-5.00", "answer": {"text": "5 x 2."}}],
        "t-three": [{"key": "0.00-5.00", "answer": {"text": "bonjour"}}],
    }
    for talk_id, rows in answers_of.items():
        (cache_dir / captions.NAME / f"{talk_id}.{captions.SEGMENT_MODEL}.clips.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    result = captions.dry_run_unescaped(cache_dir, data_root)

    # The train set's talk is out; of the other two, one of the three decoded
    # clips has no cached answer and is the one a preparation would ask.
    assert result["talks"] == 2
    assert result["segments"] == 3
    assert result["missing"] == 1
    assert result["per_talk"]["t-one"] == {"segments": 2, "missing": 1}
    assert result["per_talk"]["t-three"] == {"segments": 1, "missing": 0}


def test_the_dry_run_refuses_a_cache_it_cannot_read(tmp_path, monkeypatch):
    source = make_source(tmp_path, [talk("t-one"), talk("t-two")])
    monkeypatch.setattr(captions, "SOURCE", source)
    scan_dir = tmp_path / "cache" / captions.NAME / "captions"
    scan_dir.mkdir(parents=True)
    (scan_dir / "t-one.json").write_text(
        json.dumps({"data": [{"language": {"id": "fr"}, "automaticallyGenerated": False}]}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="not cached"):
        captions.dry_run_unescaped(tmp_path / "cache", tmp_path / "data")
