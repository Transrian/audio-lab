import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # the tests import shared helpers from conftest

from helpers import make_clip, write_tone  # noqa: E402

from audiolab.core.audio import pcm_sha256  # noqa: E402
from audiolab.data.store import (  # noqa: E402
    clip_from_dict,
    manifest_id_sha256,
    read_manifest,
    verify_clips,
    write_manifest,
)


def test_manifest_round_trip(tmp_path):
    clips = [make_clip("d:002"), make_clip("d:001")]
    write_manifest(tmp_path, clips)
    assert read_manifest(tmp_path) == sorted(clips, key=lambda clip: clip.id)


def test_manifest_keeps_accents_and_quotes(tmp_path):
    clip = make_clip("d:001", text="Il a dit « déjà vu » — l'homme, 20 %")
    write_manifest(tmp_path, [clip])
    assert read_manifest(tmp_path)[0].text == clip.text


def test_manifest_lines_are_one_json_object_per_line(tmp_path):
    write_manifest(tmp_path, [make_clip("d:001")])
    lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == "d:001"


def test_read_manifest_refuses_an_unprepared_dataset(tmp_path):
    with pytest.raises(FileNotFoundError, match="not prepared"):
        read_manifest(tmp_path)


def test_clip_from_dict_refuses_unknown_and_missing_fields():
    with pytest.raises(ValueError, match="unknown fields: \\['whoops'\\]"):
        clip_from_dict({"whoops": 1})
    with pytest.raises(ValueError, match="missing fields"):
        clip_from_dict({"id": "d:001"})


def test_a_line_without_the_optional_fields_reads_with_them_empty():
    line = json.loads(make_clip("d:001").manifest_line())
    assert not {"source_url", "source_title", "source_author", "base_wer"} & set(line), (
        "a source without attribution writes the lines it always wrote"
    )
    del line["category"]
    clip = clip_from_dict(line)
    assert (clip.category, clip.source_url, clip.source_title, clip.source_author) == ("",) * 4
    assert clip.base_wer is None


def test_per_recording_fields_are_written_when_set():
    clip = replace(make_clip("d:001"), source_url="https://example.org/v", base_wer=0.0)
    line = json.loads(clip.manifest_line())
    assert line["source_url"] == "https://example.org/v" and line["base_wer"] == 0.0
    assert clip_from_dict(line) == clip


def test_verify_clips_reports_missing_and_altered_audio(tmp_path):
    tone = write_tone(tmp_path / "audio" / "001.wav")
    good = make_clip("d:001", audio="audio/001.wav", sha256=pcm_sha256(tone))
    missing = make_clip("d:002", audio="audio/002.wav")
    write_manifest(tmp_path, [good, missing])
    assert verify_clips(tmp_path, [good, missing]) == ["d:002: missing audio audio/002.wav"]

    write_manifest(tmp_path, [make_clip("d:001", audio="audio/001.wav", sha256="1" * 64)])
    assert "does not match its sha256" in verify_clips(tmp_path, read_manifest(tmp_path))[0]


def test_manifest_id_sha256_ignores_line_order():
    clips = [make_clip("d:001"), make_clip("d:002")]
    assert manifest_id_sha256(clips) == manifest_id_sha256(list(reversed(clips)))
    assert manifest_id_sha256(clips) != manifest_id_sha256([clips[0]])
