"""Noise and reverberation augmentation, in the parts the plain install can test.

The perturbation itself runs inside NeMo's data workers and needs the heavy
stack; everything that decides *what* a run augments - the config, the bank
manifests, the NeMo dataset dict and the run record - is tested here, CPU only,
with the heavy step replaced as in the other train tests.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402
from helpers import (  # noqa: E402
    MINIMAL,
    FakeStep,
    make_clip,
    point_the_guard_at_a_test_suite,
)

from audiolab.data.store import write_manifest  # noqa: E402
from audiolab.train import runner  # noqa: E402
from audiolab.train.config import TrainConfig, TrainConfigError  # noqa: E402
from audiolab.train.data import (  # noqa: E402
    BANK_WINDOW_SECONDS,
    TrainDataError,
    write_bank_manifest,
)
from audiolab.train.run import (  # noqa: E402
    NOISE_MANIFEST_NAME,
    REVERB_MANIFEST_NAME,
    TRAIN_JSON,
)
from audiolab.train.settings import dataset_config  # noqa: E402


@pytest.fixture(autouse=True)
def no_gpu_needed(monkeypatch):
    """The training step is faked, so no GPU is either: CI machines have none."""
    monkeypatch.setattr(runner, "select_gpu", lambda spec: None)


@pytest.fixture(autouse=True)
def the_guard_checks_a_suite_the_tests_control(tmp_path, monkeypatch):
    """The guard still runs on every sequence test, but against a suite the tests own."""
    point_the_guard_at_a_test_suite(monkeypatch, tmp_path)


# --- the config --------------------------------------------------------------------


def config(**overrides) -> TrainConfig:
    return TrainConfig.from_mapping({**MINIMAL, **overrides}, name="example")


def test_the_augmentation_fields_read_their_defaults_when_absent():
    """Absent fields mean the run is exactly the run of today: no augmentation."""
    loaded = config()
    assert loaded.noise_prob == 0.0 and loaded.noise_bank == ""
    assert loaded.noise_min_snr_db == 0.0 and loaded.noise_max_snr_db == 20.0
    assert loaded.reverb_prob == 0.0 and loaded.reverb_bank == ""


def test_a_config_with_a_full_augmentation_block_is_accepted():
    loaded = config(
        noise_prob=0.4,
        noise_bank="musan-noise",
        noise_min_snr_db=0.0,
        noise_max_snr_db=20.0,
        reverb_prob=0.2,
        reverb_bank="rirs-noises",
    )
    assert loaded.noise_prob == 0.4 and loaded.noise_bank == "musan-noise"
    assert loaded.noise_min_snr_db == 0.0 and loaded.noise_max_snr_db == 20.0
    assert loaded.reverb_prob == 0.2 and loaded.reverb_bank == "rirs-noises"


def test_a_probability_outside_the_unit_interval_is_refused():
    with pytest.raises(TrainConfigError, match="noise_prob must be a probability"):
        config(noise_prob=1.5)
    with pytest.raises(TrainConfigError, match="reverb_prob must be a probability"):
        config(reverb_prob=-0.1)


def test_an_snr_range_that_is_not_a_range_is_refused():
    with pytest.raises(TrainConfigError, match="must be above noise_min_snr_db"):
        config(
            noise_prob=0.4,
            noise_bank="musan-noise",
            noise_min_snr_db=10.0,
            noise_max_snr_db=10.0,
        )
    with pytest.raises(TrainConfigError, match="must be above noise_min_snr_db"):
        config(
            noise_prob=0.4, noise_bank="musan-noise", noise_min_snr_db=15.0, noise_max_snr_db=10.0
        )


def test_a_probability_without_its_bank_is_refused():
    """There would be no noise to mix, or no RIR to convolve with."""
    with pytest.raises(TrainConfigError, match="needs noise_bank"):
        config(noise_prob=0.4)
    with pytest.raises(TrainConfigError, match="needs reverb_bank"):
        config(reverb_prob=0.2)


def test_a_bank_name_that_cannot_name_a_prepared_dataset_is_refused():
    with pytest.raises(TrainConfigError, match="noise_bank .* is not a prepared dataset name"):
        config(noise_prob=0.4, noise_bank="no such/dataset")
    with pytest.raises(TrainConfigError, match="reverb_bank .* is not a prepared dataset name"):
        config(reverb_prob=0.2, reverb_bank="Upper")


def test_an_unknown_key_is_still_refused_among_the_augmentation_fields(tmp_path):
    """Adding the fields must not loosen the rule that keeps a typo from costing hours."""
    raw = {**MINIMAL, "noise_probs": 0.4, "noise_bank": "musan-noise"}
    path = tmp_path / "typo.yaml"
    path.write_text("".join(f"{key}: {value}\n" for key, value in raw.items()), encoding="utf-8")
    with pytest.raises(TrainConfigError, match="unknown training keys: noise_probs"):
        TrainConfig.from_file(path)


def test_a_disabled_augmentation_records_exactly_nothing_new():
    """A run with every probability at 0 is today's run, and its record says so."""
    plain = config().to_dict()
    spelled_out = config(
        noise_prob=0.0,
        noise_bank="",
        noise_min_snr_db=0.0,
        noise_max_snr_db=20.0,
        reverb_prob=0.0,
        reverb_bank="",
    ).to_dict()
    assert spelled_out == plain
    for key in (
        "noise_prob",
        "noise_bank",
        "noise_min_snr_db",
        "noise_max_snr_db",
        "reverb_prob",
        "reverb_bank",
    ):
        assert key not in plain


def test_the_recorded_fields_follow_the_probability_that_enabled_them():
    recorded = config(noise_prob=0.4, noise_bank="musan-noise").to_dict()
    assert recorded["noise_prob"] == 0.4 and recorded["noise_bank"] == "musan-noise"
    assert "reverb_prob" not in recorded and "reverb_bank" not in recorded
    reverb_only = config(reverb_prob=0.2, reverb_bank="rirs-noises").to_dict()
    assert reverb_only["reverb_prob"] == 0.2 and reverb_only["reverb_bank"] == "rirs-noises"
    for key in ("noise_prob", "noise_bank", "noise_min_snr_db", "noise_max_snr_db"):
        assert key not in reverb_only


# --- the bank manifests -------------------------------------------------------------


def write_bank(root: Path, name: str, rows: list[dict]) -> Path:
    """A prepared noise bank: the manifest rows it holds, as clips.

    ``rows`` are the bare fields a bank row carries over from a speech clip
    (``id`` names the file, as the real banks' rows do); every row gets the
    fields a bank row always has.
    """
    clips = []
    for row in rows:
        clip_id = row.get("id", f"{name}:file-{len(clips):04d}")
        fields = {key: value for key, value in row.items() if key != "id"}
        clips.append(make_clip(clip_id, language="none", group=clip_id, **fields))
    write_manifest(root / name, clips)
    return root


def test_a_bank_manifest_keeps_only_the_train_half_sorted_by_id(tmp_path):
    root = write_bank(
        tmp_path / "data",
        "noise-bank",
        [
            {"id": "noise-bank:z-last", "source_split": "train", "duration_s": 5.0},
            {"id": "noise-bank:a-first", "source_split": "train", "duration_s": 3.0},
            {"id": "noise-bank:m-middle", "source_split": "eval", "duration_s": 9.0},
        ],
    )
    path = tmp_path / "noise.jsonl"
    count, digest = write_bank_manifest(path, root / "noise-bank")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert count == 2
    # The eval row is nowhere in the file: it is what the noisy panels score.
    assert "m-middle" not in path.read_text(encoding="utf-8")
    assert [line["audio_filepath"] for line in lines] == [
        str((root / "noise-bank" / "audio" / "noise-bank_a-first.wav").resolve()),
        str((root / "noise-bank" / "audio" / "noise-bank_z-last.wav").resolve()),
    ]
    for line in lines:
        assert set(line) == {"audio_filepath", "duration", "offset", "text"}
        assert line["text"] == "" and line["offset"] == 0.0
        assert Path(line["audio_filepath"]).is_absolute()
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def windows_for(lines: list[dict], file: str) -> list[dict]:
    """The manifest lines of the one bank file named ``file``."""
    return [line for line in lines if line["audio_filepath"].endswith(file)]


def test_a_long_noise_file_becomes_windows_of_at_most_the_window(tmp_path):
    """NeMo reads one window per augmented clip, never a whole multi-minute file."""
    root = write_bank(
        tmp_path / "data",
        "noise-bank",
        [
            {"id": "noise-bank:long", "source_split": "train", "duration_s": 45.0},
            {"id": "noise-bank:exact", "source_split": "train", "duration_s": 40.0},
            {"id": "noise-bank:short", "source_split": "train", "duration_s": 10.0},
        ],
    )
    path = tmp_path / "noise.jsonl"
    count, _ = write_bank_manifest(path, root / "noise-bank")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert count == 6
    long_windows = windows_for(lines, "noise-bank_long.wav")
    assert [(line["offset"], line["duration"]) for line in long_windows] == [
        (0.0, 20.0),
        (20.0, 20.0),
        (40.0, 5.0),
    ]
    # The windows cover the file exactly: from zero to its full length, once.
    assert long_windows[0]["offset"] == 0.0
    assert long_windows[-1]["offset"] + long_windows[-1]["duration"] == 45.0
    for line in lines:
        assert line["duration"] <= BANK_WINDOW_SECONDS
    exact = windows_for(lines, "noise-bank_exact.wav")
    assert len(exact) == 2 and all(line["duration"] == 20.0 for line in exact)
    short = windows_for(lines, "noise-bank_short.wav")
    assert len(short) == 1 and short[0]["duration"] == 10.0


def test_a_bank_with_no_train_row_is_refused(tmp_path):
    root = write_bank(
        tmp_path / "data",
        "noise-bank",
        [{"id": "noise-bank:only-eval", "source_split": "eval", "duration_s": 5.0}],
    )
    with pytest.raises(TrainDataError, match="no row with source_split == 'train'"):
        write_bank_manifest(tmp_path / "noise.jsonl", root / "noise-bank")


def test_a_bank_manifest_is_stable_for_the_same_bank(tmp_path):
    """Same bank, two writes: the same lines and the same hash the record keeps."""
    rows = [
        {"id": "noise-bank:a", "source_split": "train", "duration_s": 25.0},
        {"id": "noise-bank:b", "source_split": "train", "duration_s": 2.0},
        {"id": "noise-bank:c", "source_split": "eval", "duration_s": 8.0},
    ]
    first_root = write_bank(tmp_path / "one", "noise-bank", rows)
    second_root = write_bank(tmp_path / "two", "noise-bank", rows)
    first = tmp_path / "one" / "noise.jsonl"
    second = tmp_path / "two" / "noise.jsonl"
    count_a, digest_a = write_bank_manifest(first, first_root / "noise-bank")
    count_b, digest_b = write_bank_manifest(second, second_root / "noise-bank")
    # Different roots, different absolute paths, different hashes - the hashes
    # of one bank under one root are stable across writes.
    assert count_a == count_b
    again = tmp_path / "one" / "noise-again.jsonl"
    _, digest_again = write_bank_manifest(again, first_root / "noise-bank")
    assert digest_a == digest_again
    assert first.read_text(encoding="utf-8") == again.read_text(encoding="utf-8")


# --- the NeMo dataset dict ----------------------------------------------------------


def test_the_dataset_config_has_no_augmentor_when_the_run_does_not_augment(tmp_path):
    loaded = config()
    settings = dataset_config(loaded, tmp_path / "manifest.jsonl")
    assert "augmentor" not in settings
    assert settings["manifest_filepath"] == str(tmp_path / "manifest.jsonl")


def test_the_dataset_config_names_the_perturbations_the_way_nemo_expects(tmp_path):
    """The keys and the kwargs are NeMo's, checked against perturb.py."""
    noise = tmp_path / "noise.jsonl"
    reverb = tmp_path / "reverb.jsonl"
    loaded = config(
        noise_prob=0.4,
        noise_bank="musan-noise",
        noise_min_snr_db=0.0,
        noise_max_snr_db=20.0,
        reverb_prob=0.2,
        reverb_bank="rirs-noises",
    )
    settings = dataset_config(
        loaded, tmp_path / "manifest.jsonl", noise_manifest=noise, reverb_manifest=reverb
    )
    assert settings["augmentor"] == {
        "noise": {
            "prob": 0.4,
            "manifest_path": str(noise),
            "min_snr_db": 0.0,
            "max_snr_db": 20.0,
        },
        "impulse": {
            "prob": 0.2,
            "manifest_path": str(reverb),
            "normalize_impulse": True,
            "shift_impulse": True,
        },
    }
    # No rng: the run seeds itself, and each worker's random is seeded at init.
    for entry in settings["augmentor"].values():
        assert "rng" not in entry


def test_noise_alone_and_reverb_alone_each_add_only_their_entry(tmp_path):
    noise = tmp_path / "noise.jsonl"
    reverb = tmp_path / "reverb.jsonl"
    noise_only = dataset_config(
        config(noise_prob=0.4, noise_bank="musan-noise"),
        tmp_path / "manifest.jsonl",
        noise_manifest=noise,
    )
    assert set(noise_only["augmentor"]) == {"noise"}
    reverb_only = dataset_config(
        config(reverb_prob=0.2, reverb_bank="rirs-noises"),
        tmp_path / "manifest.jsonl",
        reverb_manifest=reverb,
    )
    assert set(reverb_only["augmentor"]) == {"impulse"}


def test_a_probability_without_its_manifest_cannot_reach_nemo(tmp_path):
    """The runner always writes the manifest it names; this is the tripwire."""
    with pytest.raises(ValueError, match="no noise manifest"):
        dataset_config(
            config(noise_prob=0.4, noise_bank="musan-noise"), tmp_path / "manifest.jsonl"
        )
    with pytest.raises(ValueError, match="no reverb manifest"):
        dataset_config(
            config(reverb_prob=0.2, reverb_bank="rirs-noises"),
            tmp_path / "manifest.jsonl",
            noise_manifest=tmp_path / "noise.jsonl",
        )


# --- the run directory and the record -----------------------------------------------


def train_json(directory: Path) -> dict:
    return json.loads((directory / TRAIN_JSON).read_text(encoding="utf-8"))


def write_dataset(root: Path, name: str, count: int) -> Path:
    write_manifest(root / name, [make_clip(f"{name}:{index:04d}") for index in range(count)])
    return root


def run_config(tmp_path: Path, **overrides) -> TrainConfig:
    base = tmp_path / "base.nemo"
    base.write_bytes(b"base weights")
    small = {"clips": 6, "updates": 4, **overrides}
    return config(base_model=str(base), **small)


def test_a_run_that_does_not_augment_leaves_no_trace_of_it(tmp_path, monkeypatch):
    """No bank manifests in the directory, no block in the record, nothing new in the config."""
    monkeypatch.setattr(runner, "train_checkpoint", FakeStep())
    directory = runner.run_training(
        run_config(tmp_path),
        data_dir=write_dataset(tmp_path / "data", "fleurs-fr", 20),
        runs_root=tmp_path / "runs",
        print_line=lambda _: None,
    )
    assert not (directory / NOISE_MANIFEST_NAME).exists()
    assert not (directory / REVERB_MANIFEST_NAME).exists()
    record = train_json(directory)
    assert "augmentation" not in record
    for key in (
        "noise_prob",
        "noise_bank",
        "noise_min_snr_db",
        "noise_max_snr_db",
        "reverb_prob",
        "reverb_bank",
    ):
        assert key not in record["config"]


def test_a_run_that_augments_writes_its_banks_and_records_them(tmp_path, monkeypatch):
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    data_root = write_dataset(tmp_path / "data", "fleurs-fr", 20)
    # Two bank rows each: one train and one eval, so only the train half may reach the run.
    data_root = write_bank(
        data_root,
        "musan-noise",
        [
            {"id": "musan-noise:train", "source_split": "train", "duration_s": 25.0},
            {"id": "musan-noise:eval", "source_split": "eval", "duration_s": 7.0},
        ],
    )
    data_root = write_bank(
        data_root,
        "rirs-noises",
        [
            {"id": "rirs-noises:train", "source_split": "train", "duration_s": 1.0},
            {"id": "rirs-noises:eval", "source_split": "eval", "duration_s": 2.0},
        ],
    )
    directory = runner.run_training(
        run_config(
            tmp_path,
            noise_prob=0.4,
            noise_bank="musan-noise",
            reverb_prob=0.2,
            reverb_bank="rirs-noises",
        ),
        data_dir=data_root,
        runs_root=tmp_path / "runs",
        print_line=lambda _: None,
    )
    noise_path = directory / NOISE_MANIFEST_NAME
    reverb_path = directory / REVERB_MANIFEST_NAME
    assert noise_path.is_file() and reverb_path.is_file()
    noise_lines = [json.loads(line) for line in noise_path.read_text().splitlines()]
    reverb_lines = [json.loads(line) for line in reverb_path.read_text().splitlines()]
    # The 25 s train file is two windows; the eval file is not there at all.
    assert len(noise_lines) == 2
    assert sum(line["duration"] for line in noise_lines) == 25.0
    assert "musan-noise:eval" not in noise_path.read_text()
    assert len(reverb_lines) == 1 and reverb_lines[0]["duration"] == 1.0
    assert "rirs-noises:eval" not in reverb_path.read_text()

    record = train_json(directory)
    assert record["augmentation"] == {
        "noise": {
            "prob": 0.4,
            "min_snr_db": 0.0,
            "max_snr_db": 20.0,
            "bank": "musan-noise",
            "bank_lines": 2,
            "manifest_sha256": hashlib.sha256(noise_path.read_bytes()).hexdigest(),
        },
        "reverb": {
            "prob": 0.2,
            "bank": "rirs-noises",
            "bank_lines": 1,
            "manifest_sha256": hashlib.sha256(reverb_path.read_bytes()).hexdigest(),
        },
    }
    # The config record keeps the fields the enabled probabilities named.
    assert record["config"]["noise_prob"] == 0.4
    assert record["config"]["reverb_prob"] == 0.2
    # And the heavy step was handed a run that carries the block.
    assert step.calls[0].augmentation == record["augmentation"]


def test_noise_only_records_no_reverb_block_and_vice_versa(tmp_path, monkeypatch):
    data_root = write_dataset(tmp_path / "data", "fleurs-fr", 20)
    data_root = write_bank(
        data_root,
        "musan-noise",
        [{"id": "musan-noise:train", "source_split": "train", "duration_s": 5.0}],
    )
    monkeypatch.setattr(runner, "train_checkpoint", FakeStep())
    directory = runner.run_training(
        run_config(tmp_path, noise_prob=0.4, noise_bank="musan-noise"),
        data_dir=data_root,
        runs_root=tmp_path / "runs",
        print_line=lambda _: None,
    )
    assert (directory / NOISE_MANIFEST_NAME).is_file()
    assert not (directory / REVERB_MANIFEST_NAME).exists()
    record = train_json(directory)
    assert set(record["augmentation"]) == {"noise"}
