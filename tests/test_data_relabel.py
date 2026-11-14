"""The relabelled set ``nasa-en-train-qwen``: same clips and audio, the model's labels.

Everything runs on the CPU: the STT client is faked, and its answer is derived
from the length of the tone written for each clip — one length per clip — so a
test can say which clip got which label. The audio is real short tones, so the
link, the hash and the duration the fetcher writes are the real things.
"""

from __future__ import annotations

import dataclasses
import io
import os
import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from helpers import make_clip, write_tone

from audiolab.core.audio import pcm_sha256
from audiolab.data import relabel
from audiolab.data.download import DownloadError
from audiolab.data.label_audit import answer_path, write_answer
from audiolab.data.relabel import EMPTY_ANSWER_LIMIT, fetch_nasa_en_train_qwen
from audiolab.data.store import read_manifest, write_manifest
from audiolab.data.talks import LABEL_MODEL, Served
from audiolab.train.data import select_clips

NAME = "nasa-en-train-qwen"


def suffix(clip_id: str) -> str:
    return clip_id.split(":", 1)[1]


def _duration(index: int) -> float:
    """One length per clip, so the answer of a clip is its index."""
    return 1.0 + index * 0.1


def make_source(root: Path, talks: int, per_talk: int, *, base_wer: float | None = None) -> list:
    """A prepared ``nasa-en-train`` the fetcher can read, with a tone per clip.

    The tone of the clip numbered ``index`` (in id order) lasts
    ``1 + index / 10`` seconds, so the fake server reads the clip back from its
    audio alone.
    """
    source_dir = root / "nasa-en-train"
    clips = []
    for talk in range(talks):
        for index in range(per_talk):
            clip_id = f"nasa-en-train:t{talk}-{index:03d}"
            # As a real preparation writes it: the length read from the audio, to the millisecond.
            seconds = round(_duration(talk * per_talk + index), 3)
            wav_path = source_dir / f"audio/{clip_id.replace(':', '_')}.wav"
            write_tone(wav_path, seconds=seconds)
            # A manifest holds the hash of the audio it points to, as a real one does.
            clip = make_clip(
                clip_id,
                language="en",
                group=f"nasa-en-train:t{talk}",
                group_kind="recording",
                source_split="listing",
                license="public domain (US government work)",
                duration_s=seconds,
                sha256=pcm_sha256(wav_path),
                **({"base_wer": base_wer} if base_wer is not None else {}),
            )
            clips.append(clip)
    write_manifest(source_dir, clips)
    return clips


@pytest.fixture
def fake_qwen(monkeypatch):
    """The Qwen client of the module, faked; what it was asked for, in ``called``."""

    class Fake:
        def __init__(self) -> None:
            self.called: list[int] = []
            self.texts = lambda index: f"the answer is {index}"

        def transcribe(self, wav: bytes) -> dict:
            samples, rate = sf.read(io.BytesIO(wav), dtype="float32")
            index = round((len(samples) / rate - 1.0) * 10)
            self.called.append(index)
            return {"text": self.texts(index)}

    fake = Fake()

    class FakeClient:
        """A client the fetcher only opens and closes: no network behind it."""

        def close(self) -> None:
            pass

    def as_client(model: str) -> object:
        if model != LABEL_MODEL:
            pytest.fail(f"the relabelled set asks {model}, not {LABEL_MODEL}")
        return FakeClient()

    monkeypatch.setattr(relabel, "client", as_client)
    monkeypatch.setattr(
        relabel,
        "served",
        lambda client_, language: Served(transcribe=fake.transcribe, concurrency=2),
    )
    return fake


def fetch(tmp_path, fake_qwen, texts, **source_kwargs):
    """One preparation from a source of the test's shape; the count and the data root."""
    root = tmp_path / "data"
    make_source(root, **source_kwargs)
    fake_qwen.texts = texts
    return fetch_nasa_en_train_qwen(NAME, tmp_path / "cache", root / NAME), root


# --- the draw the test compares ---------------------------------------------------


def test_select_clips_picks_the_same_suffixes_for_the_relabelled_dataset():
    """Equal usable counts => the same clips are drawn, the prefixes being constant."""
    suffixes = [f"talk-{i:03d}" for i in range(40)]
    durations = [5.0, 12.0, 19.9, 0.1, 8.0]  # a cycle; one of five is shorter than 0.3 s
    source = [
        make_clip(f"nasa-en-train:{s}", language="en", duration_s=durations[i % len(durations)])
        for i, s in enumerate(suffixes)
    ]
    derived = [
        make_clip(f"{NAME}:{s}", language="en", duration_s=durations[i % len(durations)])
        for i, s in enumerate(suffixes)
    ]
    random.Random(1).shuffle(derived)  # the order the manifest happens to be in
    for seed in (20261107, 7, 42):
        for count in (10, 25):
            drawn_source = select_clips(source, count=count, seed=seed, max_seconds=20.0)
            drawn_derived = select_clips(derived, count=count, seed=seed, max_seconds=20.0)
            assert [suffix(c.id) for c in drawn_source.clips] == [
                suffix(c.id) for c in drawn_derived.clips
            ]
            assert drawn_source.available == drawn_derived.available
            assert drawn_source.outside_duration == drawn_derived.outside_duration


# --- the fetcher -------------------------------------------------------------------


def test_fetch_relabels_the_source_with_the_qwen_answers(tmp_path, fake_qwen):
    """Every clip of the source, renamed, with the Qwen answer as its text."""
    root = tmp_path / "data"
    make_source(root, 2, 6)
    fake_qwen.texts = lambda index: f"the answer is {index}"
    # One clip's answer already in the label audit's cache: it is not asked again.
    cached = read_manifest(root / "nasa-en-train")[0]
    write_answer(
        answer_path(tmp_path / "cache", LABEL_MODEL, cached.id),
        model=LABEL_MODEL,
        language="en",
        audio_sha256=cached.sha256,
        request={},
        text="the answer is 0",
    )
    count = fetch_nasa_en_train_qwen(NAME, tmp_path / "cache", root / NAME)
    assert count == 12
    assert sorted(fake_qwen.called) == sorted(set(range(12)) - {0})

    source_clips = read_manifest(root / "nasa-en-train")
    derived = read_manifest(root / NAME)
    assert [c.id for c in derived] == [f"{NAME}:{suffix(c.id)}" for c in source_clips]
    for number, (source, clip) in enumerate(zip(source_clips, derived, strict=True)):
        assert clip.text == f"the answer is {number}"
        assert clip.group == f"{NAME}:{suffix(source.group)}"
        assert clip.sha256 == source.sha256
        assert clip.duration_s == source.duration_s
        assert clip.license == source.license
        # The audio is the source's own file, not a copy of it.
        assert (
            os.stat(root / "nasa-en-train" / source.audio).st_ino
            == os.stat(root / NAME / clip.audio).st_ino
        )


def test_fetch_keeps_the_source_fields(tmp_path, fake_qwen):
    """The group, the split and the base WER travel with the clip."""
    count, root = fetch(
        tmp_path,
        fake_qwen,
        lambda index: f"the answer is {index}",
        talks=1,
        per_talk=2,
        base_wer=0.25,
    )
    assert count == 2
    for clip in read_manifest(root / NAME):
        assert clip.base_wer == 0.25
        assert clip.group == f"{NAME}:t0"
        assert clip.source_split == "listing"
        assert clip.license == "public domain (US government work)"


def test_fetch_asks_nothing_when_the_answers_are_cached(tmp_path, fake_qwen):
    """A second preparation of the same source asks the server for nothing."""
    root = tmp_path / "data"
    make_source(root, 2, 4)
    cache = tmp_path / "cache"
    first = fetch_nasa_en_train_qwen(NAME, cache, root / NAME)
    assert first == 8
    fake_qwen.texts = lambda index: pytest.fail("the cached answers must not be asked again")
    (
        root / NAME / "manifest.jsonl"
    ).unlink()  # a killed writer leaves the answers, not the manifest
    second = fetch_nasa_en_train_qwen(NAME, cache, root / NAME)
    assert second == 8
    assert len(read_manifest(root / NAME)) == 8


def test_fetch_refuses_more_than_one_percent_empty(tmp_path, fake_qwen):
    """An empty share above the limit means the server did not answer the clips."""
    root = tmp_path / "data"
    make_source(root, 2, 30)  # 60 clips: two empties is 3.3 %, over the limit
    fake_qwen.texts = lambda index: "" if index in (0, 1) else f"the answer is {index}"
    with pytest.raises(DownloadError, match="are empty"):
        fetch_nasa_en_train_qwen(NAME, tmp_path / "cache", root / NAME)
    assert not (root / NAME / "manifest.jsonl").is_file()


def test_fetch_allows_at_most_one_percent_empty(tmp_path, fake_qwen):
    """Below the limit the empties are kept as empty texts, and counted, not refused."""
    root = tmp_path / "data"
    make_source(root, 101, 1)  # 101 clips: one empty is 0.99 %, below the limit
    fake_qwen.texts = lambda index: "" if index == 0 else f"the answer is {index}"
    count = fetch_nasa_en_train_qwen(NAME, tmp_path / "cache", root / NAME)
    assert count == 101
    derived = read_manifest(root / NAME)
    empty = [c for c in derived if c.text == ""]
    assert len(empty) == 1
    assert empty[0].id == f"{NAME}:t0-000"
    assert not empty[0].text_has_casing_punct
    assert all(c.text for c in derived if c is not empty[0])
    assert 1 / 101 < EMPTY_ANSWER_LIMIT


def test_fetch_copies_when_the_link_fails(tmp_path, fake_qwen, monkeypatch):
    """Different filesystems: the copy is the fallback, counted the same way."""

    def no_link(*args, **kwargs):
        raise OSError("cross-device link")

    monkeypatch.setattr(os, "link", no_link)
    root = tmp_path / "data"
    make_source(root, 1, 3)
    fake_qwen.texts = lambda index: f"the answer is {index}"
    count = fetch_nasa_en_train_qwen(NAME, tmp_path / "cache", root / NAME)
    assert count == 3
    for source, clip in zip(
        read_manifest(root / "nasa-en-train"), read_manifest(root / NAME), strict=True
    ):
        source_wav = (root / "nasa-en-train" / source.audio).read_bytes()
        dest_wav = root / NAME / clip.audio
        assert dest_wav.read_bytes() == source_wav
        assert os.stat(dest_wav).st_nlink == 1  # a copy, not a link


def test_fetch_never_writes_through_the_link(tmp_path, fake_qwen):
    """A write to a linked file lands on the source's inode, on the same file.

    The fetcher therefore only links and copies the source's bytes verbatim;
    the source's files come back exactly as they were, checked on samples near
    full scale, where a 16-bit to float32 round trip would move a sample by
    one — the silent change a re-encoding of the linked file would make.
    """
    root = tmp_path / "data"
    make_source(root, 1, 2)
    clip_dir = root / "nasa-en-train"
    clips = read_manifest(clip_dir)
    wav_path = clip_dir / clips[1].audio
    samples = np.array([32767, -32768, 30000, -30000, 16384, -16385], dtype="<i2")
    sf.write(str(wav_path), samples, 16000, subtype="PCM_16")
    before = wav_path.read_bytes()
    clips[1] = dataclasses.replace(clips[1], sha256=pcm_sha256(wav_path))
    write_manifest(clip_dir, clips)

    fake_qwen.texts = lambda index: f"the answer is {index}"
    fetch_nasa_en_train_qwen(NAME, tmp_path / "cache", root / NAME)
    assert wav_path.read_bytes() == before
    derived = read_manifest(root / NAME)
    assert os.stat(root / NAME / derived[1].audio).st_ino == os.stat(wav_path).st_ino

    # A second preparation keeps the links; it writes nothing through them either.
    (root / NAME / "manifest.jsonl").unlink()
    fetch_nasa_en_train_qwen(NAME, tmp_path / "cache", root / NAME)
    assert wav_path.read_bytes() == before


def test_fetch_fails_cleanly_when_the_source_is_missing(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    with pytest.raises(DownloadError, match="prepare nasa-en-train first"):
        fetch_nasa_en_train_qwen(NAME, tmp_path / "cache", root / NAME)
