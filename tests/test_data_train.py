"""What a training source takes from its split, exercised on rows built here.

Every fetcher of a training source filters the same way and stops at the same
cap, so the rules are checked once, on rows made by the test, and each fetcher's
own test only checks what its source does differently.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import make_clip  # noqa: E402

from audiolab.core.audio import SAMPLE_RATE, Audio  # noqa: E402
from audiolab.data.build import ManifestBuilder  # noqa: E402
from audiolab.data.store import read_manifest, write_manifest  # noqa: E402
from audiolab.data.train import (  # noqa: E402
    CLIP_CAP_PER_GROUP,
    MAX_SECONDS,
    MIN_GROUPS,
    MIN_SECONDS,
    TRAIN_CLIP_CAP,
    Take,
    evaluation_groups,
    plan_clips,
)


def tone(seconds: float, rate: int = SAMPLE_RATE) -> Audio:
    samples = np.zeros(int(seconds * rate), dtype=np.float32)
    return Audio(samples=samples, sample_rate=rate)


def make_take(tmp_path, **fields) -> Take:
    values = {
        "name": "example-train",
        "builder": ManifestBuilder(tmp_path, verbose=False),
        "language": "fr",
        "source_split": "train",
        "license": "CC-BY-4.0",
        "text_has_casing_punct": True,
    }
    return Take(**(values | fields))


def test_the_cap_is_the_same_number_of_clips_for_every_source():
    """The screening compares sources, so the size they are cut to must not vary."""
    assert TRAIN_CLIP_CAP == 12_000


def test_take_keeps_rows_until_the_cap_and_stops_reading(tmp_path):
    take = make_take(tmp_path, cap=3)
    for index in range(5):
        take.offer(f"u{index}", tone(3.0), text=f"bonjour {index}", group=f"spk{index}")
    assert take.kept == 3
    assert take.rows_read == 3
    assert take.full()
    assert not take.offer("u9", tone(3.0), text="bonjour", group="spk9")
    assert take.kept == 3


def test_take_leaves_out_rows_outside_the_duration_window(tmp_path):
    take = make_take(tmp_path, cap=10)
    take.offer("short", tone(MIN_SECONDS - 0.1), text="bonjour", group="spk1")
    take.offer("long", tone(MAX_SECONDS + 0.1), text="bonjour", group="spk2")
    take.offer("edge_low", tone(MIN_SECONDS), text="bonjour", group="spk3")
    take.offer("edge_high", tone(MAX_SECONDS), text="bonjour", group="spk4")
    assert take.kept == 2
    assert take.too_short == 1
    assert take.too_long == 1
    assert take.rows_read == 4


def test_take_leaves_out_a_row_without_text(tmp_path):
    take = make_take(tmp_path, cap=10)
    assert not take.offer("blank", tone(3.0), text="   ", group="spk1")
    assert not take.offer("empty", tone(3.0), text="", group="spk1")
    assert take.without_text == 2
    assert take.builder.clips == []


def test_take_leaves_out_a_group_an_evaluation_set_already_uses(tmp_path):
    """A speaker scored in an evaluation set is never trained on (decision 0002)."""
    take = make_take(tmp_path, cap=10, excluded=frozenset({"1234"}))
    take.offer("a", tone(3.0), text="bonjour", group="1234")
    take.offer("b", tone(3.0), text="bonjour", group="5678")
    assert take.in_evaluation_set == 1
    assert take.kept == 1
    assert [clip.group for clip in take.builder.clips] == ["example-train:5678"]


def test_take_writes_the_fields_a_later_recipe_needs(tmp_path):
    take = make_take(tmp_path, cap=10, group_kind="recording", text_has_casing_punct=False)
    take.offer("u1", tone(2.5), text="bonjour", group="video-1")
    path = take.builder.write()
    written = read_manifest(path.parent)
    assert len(written) == 1
    assert written[0].id == "example-train:u1"
    assert written[0].group == "example-train:video-1"
    assert written[0].group_kind == "recording"
    assert written[0].source_split == "train"
    assert written[0].license == "CC-BY-4.0"
    assert written[0].language == "fr"
    assert written[0].text_has_casing_punct is False
    assert written[0].duration_s == pytest.approx(2.5, abs=1e-3)
    assert written[0].audio == "audio/example-train_u1.wav"


def test_take_report_says_how_much_of_the_source_was_needed(tmp_path):
    take = make_take(tmp_path, cap=1)
    take.offer("u1", tone(0.2), text="bonjour", group="spk1")
    take.offer("u2", tone(3.0), text="bonjour", group="spk2")
    report = take.report()
    assert "1 clips (cap reached)" in report
    assert "1 shorter than 0.5 s" in report
    assert "without text, 0 already in an evaluation set" in report


def test_take_report_notices_a_source_that_is_smaller_than_the_cap(tmp_path):
    take = make_take(tmp_path, cap=100)
    take.offer("u1", tone(3.0), text="bonjour", group="spk1")
    assert "below the cap of 100" in take.report()


def test_evaluation_groups_reads_the_id_after_the_dataset_name(tmp_path):
    """A manifest group is ``<dataset>:<source group>``: the source id is what matters."""
    write_manifest(
        tmp_path / "mls-fr",
        [make_clip("mls-fr:a", group="mls-fr:1234"), make_clip("mls-fr:b", group="mls-fr:5678")],
    )
    assert evaluation_groups("mls-fr", root=tmp_path) == frozenset({"1234", "5678"})


def test_evaluation_groups_of_several_datasets_are_merged(tmp_path):
    write_manifest(tmp_path / "librispeech-en", [make_clip("x", group="librispeech-en:121")])
    write_manifest(tmp_path / "fleurs-fr", [make_clip("y", group="fleurs-fr:900000")])
    assert evaluation_groups("librispeech-en", "fleurs-fr", root=tmp_path) == frozenset(
        {"121", "900000"}
    )


def test_evaluation_groups_is_empty_for_a_dataset_that_is_not_prepared(tmp_path):
    """Nothing is excluded when the evaluation set is not on this machine."""
    assert evaluation_groups("voxpopuli-en-nonnative", root=tmp_path) == frozenset()


def test_one_group_may_only_hold_its_share_of_the_cap(tmp_path):
    """A corpus read speaker after speaker must not spend the cap on a few readers."""
    take = make_take(tmp_path, cap=4, cap_per_group=1)
    assert take.offer("a1", tone(3.0), text="bonjour", group="spkA") is True

    assert take.offer("a2", tone(3.0), text="bonjour", group="spkA") is False
    assert take.group_full("spkA") is True
    assert take.group_full("spkB") is False
    assert take.beyond_group_cap == 1

    assert take.offer("b1", tone(3.0), text="bonjour", group="spkB") is True
    assert take.kept == 2
    assert not take.full()


def test_the_share_of_a_group_spreads_the_cap_over_many_groups(tmp_path):
    """The screening compares sources, so each must cover a comparable number of groups."""
    assert MIN_GROUPS == 60
    assert CLIP_CAP_PER_GROUP == 200
    take = make_take(tmp_path)
    assert take.cap_per_group == CLIP_CAP_PER_GROUP


def test_take_report_says_what_the_share_of_a_group_left_out(tmp_path):
    take = make_take(tmp_path, cap=4, cap_per_group=1)
    take.offer("a1", tone(3.0), text="bonjour", group="spkA")
    take.offer("a2", tone(3.0), text="bonjour", group="spkA")

    assert "1 beyond the 1-clip share of a group" in take.report()


@dataclass(frozen=True)
class Row:
    """One row of a source as its own plan sees it: what it says, when, and whose."""

    text: str
    start: float
    end: float
    speaker: str


def row(text: str = "bonjour", start: float = 0.0, end: float = 3.0, speaker: str = "spkA"):
    return Row(text=text, start=start, end=end, speaker=speaker)


def test_planning_clips_applies_the_filters_of_the_cap_to_the_annotation():
    """A source that must read its whole annotation first decides from that alone."""
    take = make_take(Path("/tmp"), cap=10, cap_per_group=2)
    rows = [
        row("bonjour"),
        row("", speaker="spkB"),
        row(end=0.4, speaker="spkC"),
        row(end=25.0, speaker="spkD"),
        row("bonjour", speaker="spkA"),
        row("bonjour", speaker="spkA"),
        row("bonjour", speaker="spkA"),
    ]
    planned, counts = plan_clips(take, rows)

    assert [planned_row.speaker for planned_row in planned] == ["spkA", "spkA"]
    assert counts.rows_read == 7
    assert counts.without_text == 1
    assert counts.too_short == 1
    assert counts.too_long == 1
    assert counts.beyond_group_cap == 2
    assert counts.per_group == {"spkA": 2}


def test_planning_clips_stops_at_the_cap_and_keeps_the_source_order():
    take = make_take(Path("/tmp"), cap=2, cap_per_group=2)
    rows = [row(speaker=speaker) for speaker in ("spkA", "spkB", "spkC")]

    planned, counts = plan_clips(take, rows)

    assert [planned_row.speaker for planned_row in planned] == ["spkA", "spkB"]
    # The rows after the cap are not read at all: the preparation stops there.
    assert counts.rows_read == 2


def test_the_report_of_a_planned_source_says_what_a_one_pass_source_says(tmp_path):
    """The plan counted the rows without reading them, and the run adds those counts.

    The plan only reads the annotation, so it cannot know whether a row is really
    usable; the audio is offered afterwards and ``absorb`` adds what the plan
    decided about the rows the run never got to.
    """
    take = make_take(tmp_path, cap=4, cap_per_group=4)
    rows = [row(), row(end=0.4), row("", speaker="spkB"), row(speaker="spkC")]
    planned, counts = plan_clips(take, rows)
    assert len(planned) == 2

    for planned_row in planned:
        assert take.offer(
            planned_row.speaker,
            tone(planned_row.end - planned_row.start),
            text=planned_row.text,
            group=planned_row.speaker,
        )
    take.absorb(counts)

    assert take.kept == 2
    assert "4 rows read" in take.report()
    assert "1 shorter than 0.5 s" in take.report()
    assert "1 without text" in take.report()
