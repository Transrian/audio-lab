"""A preparation that is stopped keeps its clips, and the next one takes them back."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from audiolab.core.audio import SAMPLE_RATE, Audio  # noqa: E402
from audiolab.core.stop import Stopped, request_stop  # noqa: E402
from audiolab.data import build  # noqa: E402
from audiolab.data.build import PARTIAL_NAME, ManifestBuilder  # noqa: E402
from audiolab.data.store import MANIFEST_NAME, read_manifest  # noqa: E402
from audiolab.data.train import Take, write_clips  # noqa: E402

FIELDS = {
    "language": "fr",
    "text": "Bonjour.",
    "text_has_casing_punct": True,
    "group": "example:speaker",
    "group_kind": "speaker",
    "source_split": "test",
    "license": "CC-BY-4.0",
}


def tone(seconds: float = 1.0, level: float = 0.25) -> Audio:
    return Audio(np.full(int(seconds * SAMPLE_RATE), level, dtype=np.float32), SAMPLE_RATE)


def never_read() -> Audio:
    raise AssertionError("a clip already prepared was read again")


def prepare(directory: Path, ids: list[str], *, stop_after: int | None = None) -> ManifestBuilder:
    """Add one clip per id, asking for a stop once ``stop_after`` were added."""
    builder = ManifestBuilder(directory, verbose=False)
    for index, clip_id in enumerate(ids, start=1):
        if index == stop_after:
            request_stop()
        builder.add(clip_id, tone, **FIELDS)
    builder.write()
    return builder


def test_a_stopped_preparation_keeps_its_clips_and_writes_no_manifest(tmp_path):
    with pytest.raises(Stopped, match="preparation of example stopped"):
        prepare(tmp_path / "example", ["example:1", "example:2", "example:3"], stop_after=2)
    directory = tmp_path / "example"
    assert not (directory / MANIFEST_NAME).exists()
    lines = (directory / PARTIAL_NAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3  # the key, then the two clips finished before the stop
    assert sorted(path.name for path in (directory / "audio").iterdir()) == [
        "example_1.wav",
        "example_2.wav",
    ]


def test_a_stopped_preparation_closes_its_progress_bar(tmp_path, monkeypatch):
    closed = []

    class Bar:
        def update(self, n=1):
            pass

        def close(self):
            closed.append(True)

    monkeypatch.setattr(build, "progress", lambda **_: Bar())
    request_stop()
    with pytest.raises(Stopped), ManifestBuilder(tmp_path / "example", verbose=False) as builder:
        builder.add("example:1", tone, **FIELDS)
    assert closed


def test_the_next_preparation_takes_the_finished_clips_without_reading_them(tmp_path, run_again):
    directory = tmp_path / "example"
    with pytest.raises(Stopped):
        prepare(directory, ["example:1", "example:2", "example:3"], stop_after=2)
    run_again()
    first = (directory / PARTIAL_NAME).read_text(encoding="utf-8").splitlines()[1]

    builder = ManifestBuilder(directory, verbose=False)
    builder.add("example:1", never_read, **FIELDS)
    builder.add("example:2", never_read, **FIELDS)
    builder.add("example:3", tone, **FIELDS)
    builder.write()

    clips = read_manifest(directory)
    assert [clip.id for clip in clips] == ["example:1", "example:2", "example:3"]
    assert clips[0].manifest_line() == first
    assert not (directory / PARTIAL_NAME).exists()


def test_a_clip_whose_fields_changed_is_prepared_again(tmp_path, run_again):
    """The text, the license, every field takes part: a wrong reuse is a bug (D-017)."""
    directory = tmp_path / "example"
    with pytest.raises(Stopped):
        prepare(directory, ["example:1", "example:2"], stop_after=1)
    run_again()
    builder = ManifestBuilder(directory, verbose=False)
    read = []
    builder.add("example:1", lambda: read.append(1) or tone(), **{**FIELDS, "text": "Salut."})
    assert read == [1]
    builder.write()
    assert read_manifest(directory)[0].text == "Salut."


def test_a_clip_whose_audio_file_is_gone_is_prepared_again(tmp_path, run_again):
    directory = tmp_path / "example"
    with pytest.raises(Stopped):
        prepare(directory, ["example:1", "example:2"], stop_after=1)
    run_again()
    (directory / "audio" / "example_1.wav").unlink()
    builder = ManifestBuilder(directory, verbose=False)
    builder.add("example:1", tone, **FIELDS)
    assert (directory / "audio" / "example_1.wav").is_file()


def test_clips_written_by_other_code_are_prepared_again(tmp_path, monkeypatch, run_again):
    """The samples of a WAV come from the code that decoded them, which the key names."""
    directory = tmp_path / "example"
    with pytest.raises(Stopped):
        prepare(directory, ["example:1", "example:2"], stop_after=1)
    run_again()
    monkeypatch.setattr(build, "preparation_key", lambda: "another version of the code")
    builder = ManifestBuilder(directory, verbose=False)
    assert builder.prepared("example:1", **FIELDS) is None
    read = []
    builder.add("example:1", lambda: read.append(1) or tone(level=0.5), **FIELDS)
    builder.write()
    assert read == [1]
    assert len(read_manifest(directory)) == 1


def test_a_torn_last_line_is_skipped_and_its_clip_prepared_again(tmp_path, run_again):
    directory = tmp_path / "example"
    with pytest.raises(Stopped):
        prepare(directory, ["example:1", "example:2", "example:3"], stop_after=2)
    run_again()
    partial = directory / PARTIAL_NAME
    partial.write_text(partial.read_text(encoding="utf-8")[:-20], encoding="utf-8")
    builder = ManifestBuilder(directory, verbose=False)
    assert builder.prepared("example:1", **FIELDS) is not None
    assert builder.prepared("example:2", **FIELDS) is None


def test_the_key_names_the_code_and_the_decoding_libraries(tmp_path, monkeypatch):
    assert build.preparation_key() == build.preparation_key()
    assert len(build.preparation_key()) == 64

    # A copy of the package, so that a test can edit its code.
    package = Path(build.__file__).parent.parent
    copy = tmp_path / "audiolab"
    for path in [
        *package.glob("data/*.py"),
        *package.glob("data/*.json"),
        package / "core/audio.py",
    ]:
        (copy / path.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copy / path.relative_to(package))
    monkeypatch.setattr(build, "__file__", str(copy / "data" / "build.py"))
    before = build.preparation_key()
    assert before == build.preparation_key()

    talks = copy / "data" / "talks.py"
    talks.write_text(talks.read_text(encoding="utf-8") + "\n# an edit\n", encoding="utf-8")
    edited = build.preparation_key()
    assert edited != before, "an edit to a data module prepares again"

    real_version = build.version
    monkeypatch.setattr(
        build, "version", lambda name: "0.0" if name == "soxr" else real_version(name)
    )
    assert build.preparation_key() != edited, "another resampler prepares again"


# --- training sources: the cap is decided again on what was written --------------


def make_take(directory: Path) -> Take:
    return Take(
        name="example-train",
        builder=ManifestBuilder(directory, verbose=False),
        language="fr",
        source_split="train",
        license="CC-BY-4.0",
        text_has_casing_punct=True,
        cap=3,
    )


def test_a_stop_is_honoured_while_rows_are_left_out(tmp_path):
    take = make_take(tmp_path / "example-train")
    assert not take.offer("1", tone(seconds=0.1), text="Trop court.", group="a")
    request_stop()
    with pytest.raises(Stopped, match="preparation of example-train stopped"):
        take.offer("2", tone(seconds=0.1), text="Trop court.", group="a")
    assert take.rows_read == 1


def test_a_resumed_training_source_decides_on_the_written_length_without_reading(
    tmp_path, run_again
):
    directory = tmp_path / "example-train"
    take = make_take(directory)
    rows = [("a", 1.0), ("b", 30.0), ("c", 2.0), ("d", 1.5)]  # b is too long
    with pytest.raises(Stopped):
        for source_id, seconds in rows:
            if source_id == "d":  # c is written, then the stop comes
                request_stop()
            take.offer(source_id, lambda s=seconds: tone(s), text="Un mot.", group="spk")
    run_again()

    resumed = make_take(directory)
    reads = []

    def read(seconds):
        reads.append(seconds)
        return tone(seconds)

    for source_id, seconds in rows:
        resumed.offer(source_id, lambda s=seconds: read(s), text="Un mot.", group="spk")
    resumed.builder.write()
    assert reads == [30.0, 1.5]  # what was not written before: the long row, and d
    assert resumed.kept == 3 and resumed.too_long == 1
    assert [clip.id for clip in read_manifest(directory)] == [
        "example-train:a",
        "example-train:c",
        "example-train:d",
    ]


@dataclass(frozen=True)
class Row:
    id: str
    text: str
    start: float
    end: float
    speaker: str


def test_a_recording_whose_clips_are_all_written_is_not_read_again(tmp_path):
    directory = tmp_path / "example-train"
    rows = [Row("m1-0", "Un.", 0.0, 1.0, "spk"), Row("m1-1", "Deux.", 1.0, 2.0, "spk")]
    take = make_take(directory)
    for row in rows:
        take.decide(text=row.text, group=row.speaker, seconds=row.end - row.start)
    write_clips(take, "m1", lambda: tone(3.0), rows)

    again = make_take(directory)
    write_clips(again, "m1", never_read, rows)
    again.builder.write()
    assert len(read_manifest(directory)) == 2
