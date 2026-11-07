import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from helpers import make_clip, speech_clips  # noqa: E402

from audiolab.data.panels import (  # noqa: E402
    DEFAULT_SEED,
    FAST_CLIPS,
    FULL_CLIPS,
    NONSPEECH_CLIPS,
    PANEL_NAMES,
    build_panels,
    check_panel,
    load_dataset_clips,
    panel_path,
    read_panel,
    sealed_is_disjoint,
    select_panels,
)


def panels_of(clips, tmp_path, dataset="fake", built="2026-11-07"):
    return build_panels(dataset, clips, tmp_path / "configs" / "panels", built=built)


def test_speech_panels_have_the_spec_sizes():
    clips = speech_clips("fake", speakers=100, per_speaker=10)
    chosen = select_panels(clips, 1234)
    assert len(chosen["sealed"]) == 20 * 10  # 20 % of the 100 groups
    assert len(chosen["full"]) == FULL_CLIPS
    assert len(chosen["fast"]) == FAST_CLIPS
    assert chosen["full"] == sorted(chosen["full"])
    assert chosen["fast"] == sorted(chosen["fast"])


def test_speech_panels_are_disjoint_and_sealed_groups_are_whole():
    clips = speech_clips("fake", speakers=100, per_speaker=10)
    chosen = select_panels(clips, 1234)
    sealed = set(chosen["sealed"])
    full = set(chosen["full"])
    fast = set(chosen["fast"])
    assert not sealed & full and not sealed & fast and fast <= full
    assert sealed_is_disjoint(clips, chosen["sealed"])


def test_no_group_dominates_the_full_panel():
    clips = speech_clips("fake", speakers=100, per_speaker=10)
    chosen = select_panels(clips, 1234)
    per_group = Counter(clip_id.split(":", 1)[1][:3] for clip_id in chosen["full"])
    assert len(per_group) == 80  # every remaining group takes part
    assert max(per_group.values()) - min(per_group.values()) <= 1


def test_full_panel_is_shorter_when_the_groups_are_exhausted():
    # 20 % sealed leaves 320 utterances of a 400-utterance dataset: no room for 600.
    clips = speech_clips("fake", speakers=40, per_speaker=10)
    chosen = select_panels(clips, 1234)
    assert len(chosen["sealed"]) == 8 * 10
    assert len(chosen["full"]) == 32 * 10
    assert len(chosen["fast"]) == FAST_CLIPS
    assert sealed_is_disjoint(clips, chosen["sealed"])


def test_panels_are_deterministic():
    clips = speech_clips("fake", speakers=20, per_speaker=20)
    first = select_panels(clips, 1234)
    assert first == select_panels(list(reversed(clips)), 1234)
    assert first == select_panels(clips, 1234)
    assert first != select_panels(clips, 4321)


def battery_clips() -> list:
    return (
        [
            make_clip(f"ns:silence-{index:03d}", language="none", group=f"ns:silence-{index:03d}")
            for index in range(50)
        ]
        + [
            make_clip(f"ns:noise-{index:03d}", language="none", group=f"ns:noise-{index:03d}")
            for index in range(50)
        ]
        + [
            make_clip(f"ns:esc50-{index:03d}", language="none", group=f"ns:esc50-{index:03d}")
            for index in range(200)
        ]
    )


def test_nonspeech_panels_have_the_spec_sizes_and_stay_disjoint():
    chosen = select_panels(battery_clips(), 1234)
    assert len(chosen["sealed"]) == NONSPEECH_CLIPS["sealed"]
    assert len(chosen["full"]) == NONSPEECH_CLIPS["full"]
    assert len(chosen["fast"]) == NONSPEECH_CLIPS["fast"]

    sealed, full, fast = (set(chosen[name]) for name in "sealed full fast".split())
    assert not sealed & full and not sealed & fast
    assert fast <= full  # the battery is too small for three disjoint panels
    # silence first, then noise, then ESC-50: sealed takes all silences plus some noises
    kinds = Counter(clip_id.split(":")[1].split("-")[0] for clip_id in chosen["full"])
    assert kinds == {"noise": 40, "esc50": 200}


def test_build_panels_writes_frozen_files_with_a_header(tmp_path):
    clips = speech_clips("fake", speakers=20, per_speaker=20)
    panels = panels_of(clips, tmp_path)
    assert [panel.name for panel in panels] == list(PANEL_NAMES)
    path = panel_path("fake-full", tmp_path / "configs" / "panels")
    text = path.read_text(encoding="utf-8")
    assert "# dataset: fake" in text and "# built: 2026-11-07" in text
    assert "# manifest_sha256: " in text and f"# seed: {DEFAULT_SEED}" in text

    read_back = read_panel(path)
    assert read_back.ids == panels[1].ids
    assert read_back.seed == panels[1].seed
    assert read_back.dataset == "fake"


def test_build_panels_refuses_to_overwrite(tmp_path):
    clips = speech_clips("fake", speakers=10, per_speaker=10)
    panels_of(clips, tmp_path)
    with pytest.raises(FileExistsError, match="frozen"):
        panels_of(clips, tmp_path)


def test_check_panel_accepts_a_matching_manifest(tmp_path):
    clips = speech_clips("fake", speakers=10, per_speaker=10)
    panels_of(clips, tmp_path)
    path = panel_path("fake-full", tmp_path / "configs" / "panels")
    assert check_panel(path, clips) == []


def test_check_panel_detects_changed_data(tmp_path):
    clips = speech_clips("fake", speakers=10, per_speaker=10)
    panels_of(clips, tmp_path)
    path = panel_path("fake-full", tmp_path / "configs" / "panels")

    changed = clips + [make_clip("fake:extra", group="fake:spk999")]
    assert any("different manifest" in problem for problem in check_panel(path, changed))

    renamed = clips[:-1] + [make_clip("fake:000-999", group="fake:spk000")]
    assert check_panel(path, renamed)


def test_read_panel_refuses_a_bad_header(tmp_path):
    path = tmp_path / "fake-full.txt"
    path.write_text("fake:001\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no header field"):
        read_panel(path)


def test_load_dataset_clips_reads_from_the_data_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOLAB_DATA", str(tmp_path / "data"))
    from audiolab.data.store import write_manifest

    write_manifest(tmp_path / "data" / "fake", [make_clip("fake:001")])
    assert [clip.id for clip in load_dataset_clips("fake")] == ["fake:001"]
