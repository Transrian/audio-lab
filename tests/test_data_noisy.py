"""The noisy derived sets: deterministic mixing, even conditions, derived panels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from helpers import make_clip

from audiolab.core.audio import SAMPLE_RATE, Audio, to_int16
from audiolab.data import noisy
from audiolab.data.download import DownloadError
from audiolab.data.panels import build_panels, check_panel, panel_path, read_panel
from audiolab.data.store import read_manifest, write_manifest

# --- the mixing itself ---------------------------------------------------------


def _speech(seconds: float = 2.0, amplitude: float = 0.5) -> np.ndarray:
    """A tone for the first half, silence for the rest: the loud part is active."""
    samples = np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)
    loud = len(samples) // 2
    t = np.arange(loud) / SAMPLE_RATE
    samples[:loud] = (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return samples


def _noise(seconds: float = 2.0) -> np.ndarray:
    return (np.random.default_rng(7).normal(0, 0.2, int(seconds * SAMPLE_RATE))).astype(np.float32)


def _measured_snr(speech: np.ndarray, noise: np.ndarray, mix: np.ndarray) -> float:
    speech_power = noisy._active_speech_power(speech)
    noise_power = float(np.mean((mix - speech).astype(np.float64) ** 2))
    return 10 * float(np.log10(speech_power / noise_power))


@pytest.mark.parametrize("target", (10, 5, 0))
def test_snr_of_a_mix_holds_on_the_active_speech(target):
    speech, noise = _speech(), _noise()

    mix = noisy._add_noise(speech, noise, target)

    assert abs(_measured_snr(speech, noise, mix) - target) < 0.5


def test_mixing_twice_writes_identical_audio():
    speech, noise, rir = _speech(), _noise(), _speech(seconds=1.0, amplitude=0.8)

    first_mix = noisy._add_noise(speech, noise, 5)
    second_mix = noisy._add_noise(speech, noise, 5)
    first_reverb = noisy._reverb(speech, rir)
    second_reverb = noisy._reverb(speech, rir)

    assert to_int16(first_mix).tobytes() == to_int16(second_mix).tobytes()
    assert to_int16(first_reverb).tobytes() == to_int16(second_reverb).tobytes()


def test_reverb_keeps_the_length_and_puts_the_direct_path_first():
    speech = _speech()
    # A RIR with a delayed peak: the cut must move the peak to time zero.
    rir = np.zeros(1600, dtype=np.float32)
    rir[400] = 1.0
    rir[600] = 0.3

    out = noisy._reverb(speech, rir)

    assert len(out) == len(speech)
    # With only the direct path left at t=0, the mix starts with the speech
    # scaled by the normalised peak, not with silence.
    assert np.abs(out[0]) > 0


def _fake_material(path):
    """The audio a fake material path stands for: noise for n, tones otherwise."""
    if "n.wav" in str(path):
        return Audio(samples=_noise(seconds=0.5), sample_rate=SAMPLE_RATE)
    return Audio(samples=_speech(seconds=0.5, amplitude=0.8), sample_rate=SAMPLE_RATE)


def test_peak_is_scaled_down_to_under_one():
    speech = _speech(amplitude=0.95)
    clip = make_clip("src:one")

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(noisy, "load_audio", _fake_material)
        mix, _ = noisy._mix(
            "src:one",
            "noise-snr0",
            Audio(samples=speech, sample_rate=SAMPLE_RATE),
            [(Path("n.wav"), "public domain")],
            [Path("r.wav")],
            [([(Path("b.wav"), "src:g1")], "CC-BY-4.0")],
            clip,
        )

    assert np.abs(mix).max() <= 0.99


# --- the deterministic choices --------------------------------------------------


def test_conditions_spread_evenly_over_many_ids():
    counts = {condition: 0 for condition in noisy.CONDITIONS}
    for index in range(600):
        clip_id = f"fake-src:clip-{index:04d}"
        condition = noisy.CONDITIONS[noisy._h("noisy-condition", clip_id) % len(noisy.CONDITIONS)]
        counts[condition] += 1

    for condition, count in counts.items():
        assert 70 <= count <= 130, f"{condition}: {count} of 600"


def test_babble_uses_four_different_groups_and_alternates_pools():
    """Four speakers, four different groups; with two pools they alternate pools."""
    loaded: list[Path] = []
    fr_pool = [(Path(f"fr/g{k}/c{i}.wav"), f"fr:g{k}") for k in range(5) for i in range(2)]
    en_pool = [(Path(f"en/g{k}/c{i}.wav"), f"en:g{k}") for k in range(5) for i in range(2)]

    with pytest.MonkeyPatch().context() as mp:

        def load(source, sample_rate=SAMPLE_RATE):
            loaded.append(Path(source))
            return Audio(samples=_speech(seconds=0.5), sample_rate=sample_rate)

        mp.setattr(noisy, "load_audio", load)
        mixed, license_ = noisy._babble_segment(
            "src:one", int(1.0 * SAMPLE_RATE), [(fr_pool, "CC-BY-4.0"), (en_pool, "CC-BY-4.0")]
        )

    assert len(loaded) == 4
    groups = [group for pool in (fr_pool, en_pool) for path, group in pool if path in loaded]
    assert len(set(groups)) == 4
    assert [str(p).split("/")[0] for p in loaded] == ["fr", "en", "fr", "en"]
    assert license_ == "CC-BY-4.0"
    assert len(mixed) == int(1.0 * SAMPLE_RATE)


def test_babble_stops_when_a_pool_has_too_few_groups():
    pool = [(Path(f"g{k}/c0.wav"), f"g{k}") for k in range(3)]
    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(
            noisy,
            "load_audio",
            lambda source, sample_rate=SAMPLE_RATE: Audio(
                samples=_speech(seconds=0.5), sample_rate=SAMPLE_RATE
            ),
        )

        with pytest.raises(DownloadError, match="fewer than four free groups"):
            noisy._babble_segment("src:one", int(0.5 * SAMPLE_RATE), [(pool, "CC-BY-4.0")])


# --- what a mixed clip carries --------------------------------------------------


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("noise-snr10", "CC-BY-4.0 (speech) + public domain (noise)"),
        ("babble-snr5", "CC-BY-4.0 (speech) + CC-BY-4.0 (babble)"),
        ("reverb", f"CC-BY-4.0 (speech) + {noisy.RIRS_LICENSE} (reverb)"),
        (
            "reverb-noise-snr10",
            f"CC-BY-4.0 (speech) + {noisy.RIRS_LICENSE} (reverb) + public domain (noise)",
        ),
    ],
)
def test_the_license_names_the_material(condition, expected):
    speech = _speech(seconds=0.5)
    clip = make_clip("src:one", license="CC-BY-4.0")
    babble_pool = [(Path(f"b{k}.wav"), f"src:g{k}") for k in range(4)]

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(noisy, "load_audio", _fake_material)
        _, license_ = noisy._mix(
            "src:one",
            condition,
            Audio(samples=speech, sample_rate=SAMPLE_RATE),
            [(Path("n.wav"), "public domain")],
            [Path("r.wav")],
            [(babble_pool, "CC-BY-4.0")],
            clip,
        )

    assert license_ == expected


# --- the derived panels ---------------------------------------------------------


def _source_clips(prefix: str) -> list:
    return [
        make_clip(f"{prefix}:g{g}c{i}", group=f"{prefix}:g{g}") for g in (1, 2) for i in range(3)
    ]


def test_derived_panels_map_the_source_ids_and_keep_sealing(tmp_path):
    source = "fleurs-fake"
    derived = f"noisy-{source}"
    source_clips = _source_clips(source)
    derived_clips = [
        make_clip(
            f"{derived}:{clip.id.partition(':')[2]}",
            group=f"{derived}:{clip.group.partition(':')[2]}",
            sha256=clip.sha256,
        )
        for clip in source_clips
    ]
    build_panels(source, source_clips, panels_dir=tmp_path)

    panels = build_panels(derived, derived_clips, panels_dir=tmp_path)

    for panel in panels:
        source_panel = read_panel(panel_path(f"{source}-{panel.name}", tmp_path))
        assert panel.ids == sorted(
            f"{derived}:{clip_id.partition(':')[2]}" for clip_id in source_panel.ids
        )
        # The sealed group of the source is the sealed group of the derived set.
        source_sealed = set(read_panel(panel_path(f"{source}-sealed", tmp_path)).ids)
        derived_sealed = set(read_panel(panel_path(f"{derived}-sealed", tmp_path)).ids)
        sealed_source_groups = {clip.group for clip in source_clips if clip.id in source_sealed}
        sealed_derived_groups = {clip.group for clip in derived_clips if clip.id in derived_sealed}
        assert sealed_derived_groups == {
            f"{derived}:{group.partition(':')[2]}" for group in sealed_source_groups
        }
        # The derived panel checks clean against its own manifest.
        assert check_panel(panel_path(derived + f"-{panel.name}", tmp_path), derived_clips) == []


def test_derived_panels_refuse_a_missing_source_and_an_overwrite(tmp_path):
    source = "fleurs-fake"
    derived = f"noisy-{source}"
    derived_clips = [
        make_clip(
            f"{derived}:{clip.id.partition(':')[2]}",
            group=f"{derived}:{clip.group.partition(':')[2]}",
            sha256=clip.sha256,
        )
        for clip in _source_clips(source)
    ]
    # No source panels yet: the build is refused, naming the missing file.
    with pytest.raises(ValueError, match="fleurs-fake-fast.txt is missing"):
        build_panels(derived, derived_clips, panels_dir=tmp_path)
    # With the source panels: the build runs, and a second build is refused.
    build_panels(source, _source_clips(source), panels_dir=tmp_path)
    build_panels(derived, derived_clips, panels_dir=tmp_path)
    with pytest.raises(FileExistsError):
        build_panels(derived, derived_clips, panels_dir=tmp_path)


# --- the committed babble list -------------------------------------------------


def _pin_file(tmp_path, data):
    """A pin file in the shape of the committed babble list."""
    path = tmp_path / "babble_utterances.json"
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def _pin(**lists):
    return {"note": "a test pin", **lists}


def test_the_committed_babble_list_is_the_babble_of_the_noisy_sets():
    """The committed file reads back, and holds the babble of the prepared noisy sets."""
    babble = noisy.load_babble_utterances()
    assert set(babble) == set(noisy.NOISY_EXPECTED)
    assert {name: len(ids) for name, ids in babble.items()} == {
        "noisy-fleurs-fr": 431,
        "noisy-fleurs-en": 394,
        "noisy-servicenow-codeswitched-en-fr": 199,
    }
    # The flat list a training exclusion checks against: sorted, one id once.
    union = noisy.babble_union(babble)
    assert len(union) == 1012
    assert union == sorted(set(union))


def test_babble_union_is_the_sorted_union_without_repeats():
    assert noisy.babble_union({"noisy-fleurs-en": ["2", "1"], "noisy-fleurs-fr": ["1", "3"]}) == [
        "1",
        "2",
        "3",
    ]
    assert noisy.babble_union({}) == []


def test_the_loader_reads_a_well_formed_pin(tmp_path):
    lists = {
        "noisy-fleurs-en": ["103-1240-0001", "103-1240-0002"],
        "noisy-fleurs-fr": ["9121_800_000001"],
        "noisy-servicenow-codeswitched-en-fr": ["9121_900_000001"],
    }
    path = _pin_file(tmp_path, _pin(**lists))
    assert noisy.load_babble_utterances(path) == lists


def test_the_loader_refuses_a_list_that_is_not_sorted(tmp_path):
    path = _pin_file(tmp_path, _pin(**{"noisy-fleurs-en": ["103-1240-0002", "103-1240-0001"]}))
    with pytest.raises(ValueError, match="not sorted"):
        noisy.load_babble_utterances(path)


def test_the_loader_refuses_a_list_that_holds_an_id_twice(tmp_path):
    path = _pin_file(tmp_path, _pin(**{"noisy-fleurs-en": ["103-1240-0001", "103-1240-0001"]}))
    with pytest.raises(ValueError, match="twice"):
        noisy.load_babble_utterances(path)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (["noisy-fleurs-en"], "must be an object"),
        ({}, "holds no note"),
        ({"note": "  "}, "holds no note"),
        ({"note": "a test pin"}, "holds no list for"),
        # A file holding only some of the sets: its exclusion would silently
        # stop covering the rest, so the loader refuses it like a broken pin.
        (
            {"note": "a test pin", "noisy-fleurs-fr": ["9121_800_000001"]},
            "holds no list for",
        ),
        ({"note": "a test pin", "noisy-fake": ["1"]}, "is not a noisy clip set"),
        ({"note": "a test pin", "noisy-fleurs-en": []}, "holds no babble"),
        ({"note": "a test pin", "noisy-fleurs-en": [1]}, "list of utterance ids"),
        ({"note": "a test pin", "noisy-fleurs-en": {"a": "b"}}, "list of utterance ids"),
    ],
)
def test_the_loader_refuses_a_malformed_pin(tmp_path, data, match):
    path = _pin_file(tmp_path, data)
    with pytest.raises(ValueError, match=match):
        noisy.load_babble_utterances(path)


def test_the_loader_refuses_a_missing_or_broken_file(tmp_path):
    with pytest.raises(ValueError, match="unreadable"):
        noisy.load_babble_utterances(tmp_path / "absent.json")
    path = _pin_file(tmp_path, _pin(**{"noisy-fleurs-en": ["1"]}))
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable"):
        noisy.load_babble_utterances(path)


# --- the babble a preparation would mix ----------------------------------------


def _synthetic_pool():
    """Eight speakers, four utterances each, sorted the way a manifest read is."""
    pool = [
        (f"u{utterance:02d}-sp{speaker:02d}", f"sp{speaker:02d}")
        for speaker in range(8)
        for utterance in range(4)
    ]
    return sorted(pool)


def _clips_in_condition(condition: str, count: int) -> list[str]:
    """Clip ids whose condition is the one asked for: found by the hash itself."""
    found = []
    for index in range(10_000):
        clip_id = f"src:clip-{index:05d}"
        if noisy._condition(clip_id) == condition:
            found.append(clip_id)
            if len(found) == count:
                return found
    raise AssertionError(f"no clip found with condition {condition}")


def test_chosen_babble_utterances_is_the_mixed_utterances():
    """The pure form picks what the mix itself loads, and nothing else."""
    babble_clips = _clips_in_condition("babble-snr5", 3)
    other_clips = _clips_in_condition("noise-snr10", 2)
    pool = _synthetic_pool()

    expected = noisy.chosen_babble_utterances(babble_clips + other_clips, [pool])

    loaded: list[Path] = []

    def load(source, sample_rate=SAMPLE_RATE):
        loaded.append(Path(source))
        return Audio(samples=_speech(seconds=0.5), sample_rate=sample_rate)

    with pytest.MonkeyPatch().context() as mp:
        mp.setattr(noisy, "load_audio", load)
        for clip_id in babble_clips:
            noisy._babble_segment(
                clip_id,
                int(0.5 * SAMPLE_RATE),
                [
                    (
                        [(Path(f"audio/{utterance}.wav"), group) for utterance, group in pool],
                        "CC-BY-4.0",
                    )
                ],
            )

    assert {path.stem for path in loaded} == expected


def test_chosen_babble_utterances_ignores_the_other_conditions():
    babble_clips = _clips_in_condition("babble-snr5", 2)
    other_clips = _clips_in_condition("noise-snr0", 3)
    pool = _synthetic_pool()
    assert noisy.chosen_babble_utterances(
        babble_clips + other_clips, [pool]
    ) == noisy.chosen_babble_utterances(babble_clips, [pool])
    assert noisy.chosen_babble_utterances(other_clips, [pool]) == set()


def test_a_babble_clip_takes_two_speakers_from_each_of_two_pools():
    clip = _clips_in_condition("babble-snr5", 1)[0]
    fr = sorted(
        (f"u{utterance:02d}-sp{speaker:02d}", f"sp{speaker:02d}")
        for speaker in range(4)
        for utterance in range(6)
    )
    en = sorted(
        (f"103-{1240 + speaker:04d}-000{utterance + 1}", f"103-{1240 + speaker}")
        for speaker in range(4)
        for utterance in range(3)
    )

    expected = noisy.chosen_babble_utterances([clip], [fr, en])

    # The speakers alternate pools, and no group is taken twice.
    assert len(expected) == 4
    fr_groups = {group for utterance, group in fr if utterance in expected}
    en_groups = {group for utterance, group in en if utterance in expected}
    assert len(fr_groups) == 2 and len(en_groups) == 2


# --- the check that refuses a changed pool --------------------------------------


def test_check_babble_passes_when_the_pools_mix_the_committed_list(monkeypatch):
    babble_clips = _clips_in_condition("babble-snr5", 2)
    pool = _synthetic_pool()
    committed = {"noisy-fleurs-en": sorted(noisy.chosen_babble_utterances(babble_clips, [pool]))}
    monkeypatch.setattr(noisy, "load_babble_utterances", lambda path=None: committed)

    noisy.check_babble("noisy-fleurs-en", babble_clips, [pool])


def test_check_babble_refuses_a_pool_that_changed(monkeypatch):
    babble_clips = _clips_in_condition("babble-snr5", 2)
    pool = _synthetic_pool()
    committed = {"noisy-fleurs-en": sorted(noisy.chosen_babble_utterances(babble_clips, [pool]))}
    monkeypatch.setattr(noisy, "load_babble_utterances", lambda path=None: committed)
    # One utterance is gone from the pool: the choice changes with it.
    changed = pool[1:]
    assert noisy.chosen_babble_utterances(babble_clips, [changed]) != set(
        committed["noisy-fleurs-en"]
    )

    with pytest.raises(DownloadError, match="no longer equals"):
        noisy.check_babble("noisy-fleurs-en", babble_clips, [changed])


def test_check_babble_refuses_a_set_with_no_committed_list(monkeypatch):
    monkeypatch.setattr(
        noisy,
        "load_babble_utterances",
        lambda path=None: {"noisy-fleurs-fr": ["9121_8878_000347"]},
    )

    with pytest.raises(DownloadError, match="no committed babble list"):
        noisy.check_babble(
            "noisy-fleurs-en", _clips_in_condition("babble-snr5", 1), [_synthetic_pool()]
        )


# --- the preparation's check of the committed babble, end to end ---------------


def _panel_files(panels_dir: Path, source: str, ids: list[str]) -> None:
    """Three panel files for one source, each holding the same ids."""
    panels_dir.mkdir(exist_ok=True)
    header = "\n".join(
        [
            f"# dataset: {source} ({len(ids)} clips in {len(ids)} groups; test {len(ids)})",
            f"# items: {len(ids)}",
            "# clips_sha256: 000000000000",
            "# manifest_sha256: 000000000000",
            "# seed: 1",
            "# built: 2026-01-01",
        ]
    )
    for name in ("fast", "full", "sealed"):
        (panels_dir / f"{source}-{name}.txt").write_text(
            header + "\n" + "\n".join(ids) + "\n", encoding="utf-8"
        )


def _noisy_world(tmp_path, monkeypatch):
    """A fake prepared world holding one small noisy preparation of noisy-fleurs-fr.

    The source manifest, its panels, the babble pool, the noise and the RIRs all
    live under a temp data dir, and the audio is the fake material of the
    mixing tests. The ids are picked by the condition hash itself, so the set
    holds babble clips to mix.
    """
    name = "noisy-fleurs-fr"
    source = "fleurs-fr"
    candidates = [f"{source}:clip-{i:03d}" for i in range(300)]
    ids = sorted(
        [c for c in candidates if noisy._condition(c) == "babble-snr5"][:3]
        + [c for c in candidates if noisy._condition(c).startswith("noise")][:3]
    )
    data = tmp_path / "data"
    write_manifest(data / source, [make_clip(clip_id) for clip_id in ids])
    pool_clips = [
        make_clip(f"mls-fr-train:9121_{800 + k}_{i:06d}", group=f"mls-fr-train:9121_{800 + k}")
        for k in range(4)
        for i in range(2)
    ]
    write_manifest(data / "mls-fr-train", pool_clips)
    write_manifest(
        data / "musan-noise",
        [make_clip("musan-noise:n1", language="none", source_split="eval", group="g1")],
    )
    write_manifest(
        data / "rirs-noises",
        [
            make_clip(
                "rirs-noises:r1",
                language="none",
                source_split="eval",
                category="real-rir",
                group="g1",
            )
        ],
    )
    monkeypatch.setattr(noisy, "data_dir", lambda: data)
    monkeypatch.setattr(noisy, "load_audio", _fake_material)
    panels_dir = tmp_path / "panels"
    _panel_files(panels_dir, source, ids)
    monkeypatch.setattr(noisy, "DEFAULT_PANELS_DIR", panels_dir)
    monkeypatch.setattr(noisy, "NOISY_EXPECTED", {name: len(ids)})
    pool_ids = [
        (clip.id.split(":", 1)[1], clip.group)
        for clip in sorted(pool_clips, key=lambda clip: clip.id)
    ]
    return name, ids, pool_ids


def _committed_pin(tmp_path, monkeypatch, name: str, ids: list[str], omit: tuple = ()) -> None:
    """A committed-list pin the loader points at: one key per set, unless omitted."""
    pin = {"note": "a test pin"}
    for key in noisy.NOISY_EXPECTED:
        if key not in omit:
            pin[key] = sorted(ids) if key == name else ["9121_800_000001"]
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(pin, indent=1), encoding="utf-8")
    monkeypatch.setattr(noisy, "BABBLE_FILE", path)


def test_a_preparation_runs_when_the_pools_mix_the_committed_babble(tmp_path, monkeypatch):
    name, ids, pool_ids = _noisy_world(tmp_path, monkeypatch)
    expected = noisy.chosen_babble_utterances(ids, [pool_ids])
    assert expected
    _committed_pin(tmp_path, monkeypatch, name, sorted(expected))

    count = noisy.fetch_noisy(name, tmp_path / "cache", tmp_path / "out", verbose=False)

    assert count == len(ids)
    clips = {clip.id for clip in read_manifest(tmp_path / "out")}
    assert clips == {f"{name}:{clip_id.split(':', 1)[1]}" for clip_id in ids}
    assert (tmp_path / "out" / "manifest.jsonl").is_file()
    assert list((tmp_path / "out" / "audio").glob("*.wav"))


def test_a_preparation_is_refused_when_the_pools_mix_another_babble(tmp_path, monkeypatch):
    name, ids, pool_ids = _noisy_world(tmp_path, monkeypatch)
    expected = sorted(noisy.chosen_babble_utterances(ids, [pool_ids]))
    assert len(expected) >= 2
    # One utterance is gone from the pin: the pool no longer mixes the committed babble.
    _committed_pin(tmp_path, monkeypatch, name, expected[:-1])
    out = tmp_path / "out"

    with pytest.raises(DownloadError, match="no longer equals"):
        noisy.fetch_noisy(name, tmp_path / "cache", out, verbose=False)

    # The refusal happens before any output is written.
    assert not (out / "manifest.jsonl").exists()
    assert not (out / "manifest.partial.jsonl").exists()
    assert not (out.is_dir() and any(out.iterdir()))


def test_a_preparation_is_refused_when_the_pin_lacks_the_set(tmp_path, monkeypatch):
    name, ids, _pool_ids = _noisy_world(tmp_path, monkeypatch)
    _committed_pin(tmp_path, monkeypatch, name, [], omit=(name,))
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="holds no list for"):
        noisy.fetch_noisy(name, tmp_path / "cache", out, verbose=False)

    assert not (out / "manifest.jsonl").exists()
    assert not (out / "manifest.partial.jsonl").exists()
