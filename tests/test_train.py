"""Training: the config, the clips a run draws, the run directory, the sequence.

None of this needs PyTorch or NeMo: the fine-tune itself sits behind one call the
tests replace, which is the point of keeping it in its own module.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402
import yaml  # noqa: E402
from helpers import make_clip  # noqa: E402

from audiolab.core.checkpoint import CheckpointError  # noqa: E402
from audiolab.data.store import write_manifest  # noqa: E402
from audiolab.train import runner  # noqa: E402
from audiolab.train.config import TrainConfig, TrainConfigError  # noqa: E402
from audiolab.train.data import TrainDataError, dataset_dir, select_clips  # noqa: E402  # noqa: E402
from audiolab.train.run import (  # noqa: E402
    LOG_NAME,
    MANIFEST_NAME,
    MODEL_CONFIG,
    MODEL_NAME,
    RESUME_NAME,
    TRAIN_JSON,
    TrainRunError,
    is_finished,
    prepare_run_dir,
    resume_checkpoint,
    run_dir,
    train_json,
    write_model_config,
)
from audiolab.train.runner import Training  # noqa: E402

MINIMAL = {"base_model": "/models/base.nemo", "dataset": "fleurs-fr"}


def config(**overrides) -> TrainConfig:
    """A small run: two clips per update, so a test needs only a handful of clips."""
    return TrainConfig.from_mapping({**MINIMAL, "batch_size": 2, **overrides}, name="example")


def write_config(path: Path, raw: dict) -> Path:
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def run_config(tmp_path: Path, **overrides) -> TrainConfig:
    """A config small enough to run in a test, starting from a base model that exists."""
    base = tmp_path / "base.nemo"
    base.write_bytes(b"base weights")
    small = {"clips": 6, "updates": 4, **overrides}
    return config(base_model=str(base), **small)


def write_dataset(root: Path, name: str, count: int) -> Path:
    """A prepared dataset: a manifest of plausible clips, which is all a run reads.

    Returns the data *root* a run is pointed at, since that is what
    ``--data-dir`` and ``$AUDIOLAB_DATA`` name.
    """
    write_manifest(root / name, [make_clip(f"{name}:{index:04d}") for index in range(count)])
    return root


# --- the config -------------------------------------------------------------------


def test_a_config_reads_its_defaults_and_the_file_name_names_the_run(tmp_path):
    loaded = TrainConfig.from_file(write_config(tmp_path / "my-first-run.yaml", MINIMAL))
    assert loaded.name == "my-first-run"
    assert loaded.clips == 4000 and loaded.batch_size == 16
    assert loaded.grad_accumulation == 1 and loaded.schedule == "constant"
    assert loaded.lr_encoder == 1.0e-5 and loaded.lr_heads == 3.0e-5
    assert loaded.precision == "bf16-mixed" and loaded.seed == 20261107
    assert loaded.resume_every == 500 and loaded.log_every == 50
    assert loaded.workers == 4 and loaded.gpu == "0"


def test_an_unknown_key_in_a_training_config_is_refused(tmp_path):
    """A typo in a training config costs GPU hours, so it stops the run instead."""
    path = write_config(tmp_path / "typo.yaml", {**MINIMAL, "learning_rate": 1.0e-5})
    with pytest.raises(TrainConfigError, match="unknown training keys: learning_rate"):
        TrainConfig.from_file(path)


def test_a_config_file_that_does_not_exist_says_so(tmp_path):
    with pytest.raises(TrainConfigError, match="no such training config"):
        TrainConfig.from_file(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"base_model": "/models/base.nemo"}, "missing required keys: dataset"),
        ({**MINIMAL, "updates": 0}, "updates must be at least 1"),
        ({**MINIMAL, "clips": 8, "batch_size": 16}, "smaller than batch_size"),
        ({**MINIMAL, "schedule": "slow"}, "unknown schedule"),
        ({**MINIMAL, "precision": "fp8"}, "unknown precision"),
        ({**MINIMAL, "updates": 100, "warmup_updates": 100}, "below updates"),
        ({**MINIMAL, "lr_encoder": 0.0}, "learning rates must be above 0"),
        ({**MINIMAL, "grad_clip": -1.0}, "cannot be negative"),
        ({**MINIMAL, "dataset": "not a dataset"}, "not a prepared dataset name"),
    ],
)
def test_a_config_that_could_only_fail_later_is_refused_now(raw, message):
    with pytest.raises(TrainConfigError, match=message):
        TrainConfig.from_mapping(raw, name="example")


def test_a_run_name_that_is_not_a_plain_name_is_refused(tmp_path):
    """The name becomes a directory, so anything else is refused, not escaped."""
    with pytest.raises(TrainConfigError, match="cannot name a run"):
        TrainConfig.from_file(write_config(tmp_path / "Upper.yaml", MINIMAL))


def test_the_base_model_can_be_a_local_file_named_by_the_environment(tmp_path, monkeypatch):
    local = tmp_path / "local.nemo"
    local.write_bytes(b"weights")
    monkeypatch.setenv("AUDIOLAB_BASE", str(local))
    loaded = config(base_model="nvidia/parakeet-tdt-0.6b-v3", base_model_env="AUDIOLAB_BASE")
    assert loaded.weights().path == str(local)
    monkeypatch.delenv("AUDIOLAB_BASE")
    assert loaded.weights().hf_id == "nvidia/parakeet-tdt-0.6b-v3"


def test_the_record_of_a_run_holds_its_choices_not_its_source(tmp_path):
    loaded = TrainConfig.from_file(
        write_config(tmp_path / "recorded.yaml", {**MINIMAL, "clips": 128, "updates": 64})
    )
    recorded = loaded.to_dict()
    assert recorded["clips"] == 128 and recorded["updates"] == 64
    assert "source" not in recorded
    assert str(tmp_path) not in json.dumps(recorded)


# --- the clips a run draws --------------------------------------------------------


def clips(count: int, **fields):
    return [make_clip(f"ds:{index:04d}", **fields) for index in range(count)]


def test_the_same_seed_draws_the_same_clips_however_the_dataset_is_ordered():
    dataset = clips(50)
    first = select_clips(dataset, count=10, seed=7, max_seconds=20.0)
    second = select_clips(list(reversed(dataset)), count=10, seed=7, max_seconds=20.0)
    assert [c.id for c in first.clips] == [c.id for c in second.clips]
    assert len(first.clips) == 10
    assert [c.id for c in select_clips(dataset, count=10, seed=8, max_seconds=20.0).clips] != [
        c.id for c in first.clips
    ]


def test_clips_that_are_too_long_or_have_no_text_are_left_out():
    dataset = [
        make_clip("ds:short", duration_s=0.1),
        make_clip("ds:good", duration_s=3.0),
        make_clip("ds:long", duration_s=40.0),
        make_clip("ds:empty", text="  "),
        make_clip("ds:also-good", duration_s=19.9),
    ]
    drawn = select_clips(dataset, count=2, seed=1, max_seconds=20.0)
    assert {clip.id for clip in drawn.clips} == {"ds:good", "ds:also-good"}
    assert (drawn.outside_duration, drawn.without_text, drawn.available) == (2, 1, 2)


def test_a_dataset_with_too_few_usable_clips_says_what_to_do():
    dataset = [make_clip("ds:one"), make_clip("ds:too-long", duration_s=30.0)]
    with pytest.raises(TrainDataError, match="asks for 2"):
        select_clips(dataset, count=2, seed=1, max_seconds=20.0)


def test_the_drawn_clips_are_identified_by_their_ids():
    dataset = clips(10)
    drawn = select_clips(dataset, count=5, seed=3, max_seconds=20.0)
    assert (
        drawn.ids_sha256
        == select_clips(list(reversed(dataset)), count=5, seed=3, max_seconds=20.0).ids_sha256
    )
    assert drawn.ids_sha256 != select_clips(dataset, count=5, seed=4, max_seconds=20.0).ids_sha256


def test_a_dataset_that_was_never_prepared_says_how_to_prepare_it(tmp_path):
    with pytest.raises(TrainDataError, match="audiolab data fetch nope"):
        dataset_dir("nope", tmp_path)
    assert dataset_dir("fleurs-fr", write_dataset(tmp_path / "data", "fleurs-fr", 2)).name == (
        "fleurs-fr"
    )


# --- the run directory ------------------------------------------------------------


def test_a_finished_run_is_not_trained_again(tmp_path):
    directory = prepare_run_dir(run_dir("example", tmp_path))
    (directory / MODEL_NAME).write_bytes(b"weights")
    (directory / TRAIN_JSON).write_text("{}", encoding="utf-8")
    assert is_finished(directory)
    with pytest.raises(TrainRunError, match="already holds a finished run"):
        prepare_run_dir(run_dir("example", tmp_path))
    prepare_run_dir(run_dir("example", tmp_path), fresh=True)
    assert not is_finished(directory)
    assert directory.is_dir()


def test_a_run_that_stopped_keeps_its_resume_slot(tmp_path):
    directory = prepare_run_dir(run_dir("example", tmp_path))
    assert resume_checkpoint(directory) is None
    (directory / RESUME_NAME).write_bytes(b"optimizer state")
    assert resume_checkpoint(directory) == directory / RESUME_NAME
    assert not is_finished(directory)  # no record yet: the run did not finish


def test_a_cleared_run_directory_is_always_the_configs_own(tmp_path):
    elsewhere = tmp_path / "somewhere" / "example"
    elsewhere.mkdir(parents=True)
    (elsewhere / "keep-me").write_text("precious", encoding="utf-8")
    with pytest.raises(TrainRunError, match="refusing to clear"):
        prepare_run_dir(elsewhere, fresh=True)
    assert (elsewhere / "keep-me").is_file()


def test_a_run_writes_the_model_config_that_evaluates_it(tmp_path):
    directory = prepare_run_dir(run_dir("example", tmp_path))
    path = write_model_config(directory, name="example", gpu="GPU-abc")
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    assert raw["serve"]["checkpoint"] == str(directory / MODEL_NAME)
    assert raw["serve"]["gpu"] == "GPU-abc"
    assert raw["name"] == "example" and raw["model"] == "example"
    # No endpoint and no address: this config serves a local checkpoint (D-016).
    assert "base_url_env" not in text and "http" not in text


# --- the whole sequence, with the fine-tune replaced ------------------------------


@dataclass
class FakeResult:
    final_loss: float = 1.5
    updates: int = 64
    wall_seconds: float = 32.0
    peak_vram_gib: float = 7.25
    device: str = "Fake GPU"


class FakeStep:
    """The heavy step, replaced: it writes the two files a real run writes."""

    def __init__(self) -> None:
        self.calls: list[Training] = []

    def __call__(self, training: Training, *, print_line=print):
        self.calls.append(training)
        (training.directory / MODEL_NAME).write_bytes(b"fine-tuned weights")
        (training.directory / LOG_NAME).write_text(
            '{"update": 64, "loss": 1.5}\n', encoding="utf-8"
        )
        return FakeResult()


def test_a_run_writes_the_manifest_the_records_and_the_serving_config(tmp_path, monkeypatch):
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    base = tmp_path / "base.nemo"
    base.write_bytes(b"base weights")
    directory = runner.run_training(
        run_config(tmp_path),
        data_dir=write_dataset(tmp_path / "data", "fleurs-fr", 20),
        runs_root=tmp_path / "runs",
        print_line=lambda _: None,
    )
    assert directory == run_dir("example", tmp_path / "runs")
    assert is_finished(directory)
    assert (directory / MANIFEST_NAME).is_file()
    assert (directory / MODEL_CONFIG).is_file()
    assert not (directory / RESUME_NAME).exists()

    manifest = [json.loads(line) for line in (directory / MANIFEST_NAME).read_text().splitlines()]
    assert len(manifest) == 6
    assert set(manifest[0]) == {"audio_filepath", "duration", "text"}
    assert Path(manifest[0]["audio_filepath"]).is_absolute()

    record = train_json(directory)
    assert record["name"] == "example"
    assert record["clips"] == 6 and record["updates"] == 64
    assert record["config"]["batch_size"] == 2
    assert record["base_model"] == str(base)
    # The base model is recorded as what it is, not as where it sits.
    assert record["base_model_identity"].startswith("sha256=")
    assert str(tmp_path) not in record["base_model_identity"]
    assert record["updates_per_second"] == 2.0  # 64 updates in 32 seconds
    assert record["peak_vram_gib"] == 7.25
    assert record["device"] == "Fake GPU"
    assert record["sampled_ids_sha256"] and len(record["sampled_ids_sha256"]) == 64
    assert step.calls[0].resuming is False


def test_nothing_is_written_before_the_clips_can_be_drawn(tmp_path, monkeypatch):
    """A run that cannot start must not leave half a run directory behind."""
    monkeypatch.setattr(runner, "train_checkpoint", FakeStep())
    with pytest.raises(TrainDataError, match="asks for 99"):
        runner.run_training(
            run_config(tmp_path, clips=99),
            data_dir=write_dataset(tmp_path / "data", "fleurs-fr", 5),
            runs_root=tmp_path / "runs",
        )
    assert not run_dir("example", tmp_path / "runs").exists()


def test_a_finished_run_says_so_and_fresh_starts_over(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "train_checkpoint", FakeStep())
    dataset = write_dataset(tmp_path / "data", "fleurs-fr", 20)
    loaded = run_config(tmp_path, clips=4, updates=2)
    runner.run_training(
        loaded, data_dir=dataset, runs_root=tmp_path / "runs", print_line=lambda _: None
    )
    with pytest.raises(TrainRunError, match="--fresh"):
        runner.run_training(loaded, data_dir=dataset, runs_root=tmp_path / "runs")
    runner.run_training(
        loaded, data_dir=dataset, runs_root=tmp_path / "runs", fresh=True, print_line=lambda _: None
    )
    assert is_finished(run_dir("example", tmp_path / "runs"))


def test_a_run_that_stopped_is_continued_from_its_slot(tmp_path, monkeypatch):
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    dataset = write_dataset(tmp_path / "data", "fleurs-fr", 20)
    directory = run_dir("example", tmp_path / "runs")
    prepare_run_dir(directory)
    (directory / RESUME_NAME).write_bytes(b"optimizer state")
    runner.run_training(
        run_config(tmp_path, clips=4, updates=2),
        data_dir=dataset,
        runs_root=tmp_path / "runs",
        print_line=lambda _: None,
    )
    assert step.calls[0].resuming is True


def test_a_missing_base_checkpoint_is_reported_before_the_gpu_is_touched(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "train_checkpoint", FakeStep())
    with pytest.raises(CheckpointError, match="does not exist"):
        runner.run_training(
            config(base_model="/nope/missing.nemo"),
            data_dir=write_dataset(tmp_path / "data", "fleurs-fr", 20),
            runs_root=tmp_path / "runs",
        )


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def test_the_missing_heavy_stack_is_reported_with_a_way_out(tmp_path):
    """Without PyTorch and NeMo, the message must say how to get them."""
    if _has("torch"):
        pytest.skip("this environment has PyTorch, so the plain-install path cannot be tested")
    training = Training(
        config=config(),
        directory=tmp_path,
        clips=select_clips(clips(2), count=1, seed=1, max_seconds=20.0),
        base=config().weights(),
        base_identity="sha256=0",
        resuming=False,
    )
    with pytest.raises(TrainRunError, match="scripts/nemo-venv.sh"):
        runner.train_checkpoint(training)


# --- the command line -------------------------------------------------------------


def test_the_train_command_runs_one_config(tmp_path, monkeypatch, capsys):
    from audiolab.cli import main

    dataset = write_dataset(tmp_path / "data", "fleurs-fr", 20)
    monkeypatch.setattr(runner, "train_checkpoint", FakeStep())
    monkeypatch.setenv("AUDIOLAB_RUNS", str(tmp_path / "runs"))
    base = tmp_path / "base.nemo"
    base.write_bytes(b"base weights")
    path = write_config(
        tmp_path / "example.yaml",
        {**MINIMAL, "base_model": str(base), "clips": 4, "updates": 2, "batch_size": 2},
    )
    assert main(["--data-dir", str(dataset), "train", "--config", str(path)]) == 0
    printed = capsys.readouterr().out
    assert "train.json" in printed and "audiolab eval" in printed
    assert is_finished(run_dir("example", tmp_path / "runs"))


def test_a_training_config_with_a_mistake_exits_with_its_message(tmp_path, capsys):
    from audiolab.cli import main

    path = write_config(tmp_path / "example.yaml", {**MINIMAL, "epochs": 3})
    assert main(["train", "--config", str(path)]) == 2
    assert "unknown training keys: epochs" in capsys.readouterr().err


def test_serving_a_config_without_a_serve_block_says_what_to_do(tmp_path, capsys):
    from audiolab.cli import main

    path = write_config(
        tmp_path / "remote.yaml", {"name": "remote", "model": "id", "base_url_env": "STT_BASE_URL"}
    )
    assert main(["serve", "--model", str(path)]) == 2
    assert "no 'serve' block" in capsys.readouterr().err
