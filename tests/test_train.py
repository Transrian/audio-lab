"""Training: the config, the clips a run draws, the run directory, the sequence.

None of this needs PyTorch or NeMo: the fine-tune itself sits behind one call the
tests replace, which is the point of keeping it in its own module.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402
import yaml  # noqa: E402
from helpers import (  # noqa: E402
    MINIMAL,
    MIX_MINIMAL,
    FakeResult,
    FakeStep,
    make_clip,
    point_the_guard_at_a_test_suite,
)

from audiolab.core.checkpoint import CheckpointError  # noqa: E402
from audiolab.core.stop import Stopped, request_stop  # noqa: E402
from audiolab.data.store import write_manifest  # noqa: E402
from audiolab.train import runner  # noqa: E402
from audiolab.train.config import DatasetShare, TrainConfig, TrainConfigError  # noqa: E402
from audiolab.train.data import TrainDataError, dataset_dir, select_clips  # noqa: E402
from audiolab.train.run import (  # noqa: E402
    MANIFEST_NAME,
    MODEL_CONFIG,
    MODEL_NAME,
    RESUME_NAME,
    TRAIN_JSON,
    TrainRunError,
    is_finished,
    prepare_run_dir,
    read_run_config,
    refuse_finished_run,
    resume_checkpoint,
    run_dir,
    write_model_config,
    write_run_config,
)
from audiolab.train.runner import Training  # noqa: E402
from audiolab.train.settings import dataset_config  # noqa: E402


@pytest.fixture(autouse=True)
def no_gpu_needed(monkeypatch):
    """The training step is faked, so no GPU is either: CI machines have none."""
    monkeypatch.setattr(runner, "select_gpu", lambda spec: None)


@pytest.fixture(autouse=True)
def the_guard_checks_a_suite_the_tests_control(tmp_path, monkeypatch):
    """The guard still runs on every sequence test, but against a suite the tests own.

    The real guard's suite is the repository's full suite, whose evaluation
    datasets (fleurs-fr, mls-fr, …) the tests would have to dodge by name.
    Against a suite of one dataset the tests never prepare, the guard warns,
    never refuses, and its block in train.json is still written and checkable.
    """
    point_the_guard_at_a_test_suite(monkeypatch, tmp_path)


def train_json(directory: Path) -> dict:
    """The record of a finished run, read the way a person would read it."""
    return json.loads((directory / TRAIN_JSON).read_text(encoding="utf-8"))


def config(**overrides) -> TrainConfig:
    """A small run: two clips per update, so a test needs only a handful of clips."""
    return TrainConfig.from_mapping({**MINIMAL, **overrides}, name="example")


def write_config(path: Path, raw: dict) -> Path:
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


BASELINE = Path(__file__).parent / "data" / "train-config-baseline.json"
REPO_ROOT = Path(__file__).parent.parent


def mix_config(**overrides) -> TrainConfig:
    return TrainConfig.from_mapping({**MIX_MINIMAL, **overrides}, name="example")


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
    assert loaded.clips == 4 and loaded.batch_size == 2
    assert loaded.grad_accumulation == 1 and loaded.schedule == "constant"
    assert loaded.weight_decay == 1.0e-3 and loaded.warmup_updates == 0
    assert loaded.precision == "bf16-mixed" and loaded.grad_clip == 1.0
    assert loaded.max_clip_seconds == 20.0 and loaded.seed == 20261107
    assert loaded.resume_every == 500 and loaded.log_every == 50
    assert loaded.workers == 4 and loaded.gpu == "0"


def test_the_size_of_a_run_and_its_rates_have_to_be_written_down():
    """No default for what a run trains on, how long, or how fast it moves."""
    with pytest.raises(
        TrainConfigError,
        match="missing required keys: clips, updates, batch_size, lr_encoder, lr_heads",
    ):
        TrainConfig.from_mapping(
            {"base_model": "/models/base.nemo", "dataset": "fleurs-fr"}, name="example"
        )


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
        ({"base_model": "/models/base.nemo"}, "give either dataset\\+clips or datasets"),
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


# --- the mix form of the config ---------------------------------------------------


def test_a_mix_config_is_accepted_and_its_clips_are_the_sum_of_its_shares():
    loaded = mix_config(
        datasets=[
            {"name": "granary-fr-train-40k", "clips": 30000},
            {"name": "ds-b", "clips": 20000},
        ]
    )
    assert loaded.datasets == (
        DatasetShare(name="granary-fr-train-40k", clips=30000),
        DatasetShare(name="ds-b", clips=20000),
    )
    assert loaded.clips == 50000
    assert loaded.dataset == ""


def test_the_mix_checks_run_on_a_config_that_was_built_by_hand():
    """_check is the gate every mix config passes, whatever built it.

    from_mapping reads the shape of the ``datasets`` list and computes ``clips``
    from it; the rules about a share - a name that can be a dataset, a count of
    at least one, no name twice - live in _check alone, so a config assembled
    without from_mapping is refused in the same words.
    """
    shares = (
        DatasetShare(name="ds-a", clips=4),
        DatasetShare(name="ds-b", clips=4),
    )
    good = TrainConfig(
        name="example",
        base_model="/models/base.nemo",
        dataset="",
        clips=8,
        updates=4,
        batch_size=2,
        lr_encoder=1.0e-5,
        lr_heads=3.0e-5,
        datasets=shares,
    )
    good._check()  # a well-formed mix is accepted

    def built(datasets: tuple[DatasetShare, ...]) -> TrainConfig:
        return TrainConfig(
            name="example",
            base_model="/models/base.nemo",
            dataset="",
            clips=sum(share.clips for share in datasets),
            updates=4,
            batch_size=2,
            lr_encoder=1.0e-5,
            lr_heads=3.0e-5,
            datasets=datasets,
        )

    with pytest.raises(TrainConfigError, match="is not a prepared dataset name"):
        built((DatasetShare(name="Not A Dataset", clips=4),))._check()
    with pytest.raises(TrainConfigError, match="must be an integer of at least 1"):
        built((DatasetShare(name="ds-a", clips=0), DatasetShare(name="ds-b", clips=8)))._check()
    with pytest.raises(TrainConfigError, match="given twice"):
        built((DatasetShare(name="ds-a", clips=4), DatasetShare(name="ds-a", clips=4)))._check()
    # The checks on the run as a whole still read the mix's total, the sum of
    # the shares: one clip cannot cover a batch of two.
    with pytest.raises(TrainConfigError, match="smaller than batch_size"):
        built((DatasetShare(name="ds-a", clips=1),))._check()


def test_giving_both_forms_of_the_data_is_refused():
    with pytest.raises(TrainConfigError, match="not both"):
        mix_config(dataset="fleurs-fr", clips=4)


def test_giving_both_forms_of_the_data_is_refused_even_when_one_is_empty():
    with pytest.raises(TrainConfigError, match="not both"):
        TrainConfig.from_mapping({**MIX_MINIMAL, "dataset": "", "clips": 4}, name="example")


def test_duplicate_dataset_names_in_a_mix_are_refused():
    with pytest.raises(TrainConfigError, match="given twice"):
        mix_config(datasets=[{"name": "ds-a", "clips": 4}, {"name": "ds-a", "clips": 2}])


def test_an_entry_key_other_than_name_and_clips_is_refused():
    with pytest.raises(TrainConfigError, match="datasets\\[1\\] has unknown keys: seed"):
        mix_config(datasets=[{"name": "ds-a", "clips": 4}, {"name": "ds-b", "clips": 4, "seed": 3}])


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ({"name": "ds-a", "clips": 0}, "must be an integer of at least 1"),
        ({"name": "ds-a", "clips": -2}, "must be an integer of at least 1"),
        ({"name": "ds-a", "clips": 2.5}, "must be an integer of at least 1"),
        ({"name": "ds-a"}, "must be an integer of at least 1"),
        ({"clips": 4}, "not a prepared dataset name"),
        ({"name": "Not A Dataset", "clips": 4}, "not a prepared dataset name"),
    ],
)
def test_a_mix_entry_that_cannot_be_a_share_is_refused_now(entry, message):
    with pytest.raises(TrainConfigError, match=message):
        mix_config(datasets=[entry])


@pytest.mark.parametrize(
    "raw", [{**MIX_MINIMAL, "datasets": []}, {**MIX_MINIMAL, "datasets": "ds-a"}]
)
def test_an_empty_or_not_a_list_of_datasets_is_refused(raw):
    with pytest.raises(TrainConfigError, match="non-empty list"):
        TrainConfig.from_mapping(raw, name="example")


def test_the_single_form_of_to_dict_is_exactly_todays_output():
    """A single-dataset config must record exactly what it always did.

    The expected dict is the one ``to_dict`` gave
    ``configs/train/screen-peertube-fr.yaml`` before the mix support existed
    (captured in tests/data/train-config-baseline.json at that state).
    """
    expected = json.loads(BASELINE.read_text(encoding="utf-8"))["screen-peertube-fr.yaml"]
    assert (
        TrainConfig.from_file(REPO_ROOT / "configs/train/screen-peertube-fr.yaml").to_dict()
        == expected
    )


def test_loading_every_screen_config_gives_the_same_to_dict_as_before():
    """Mix support changed what a config may say, not what a single-dataset one records.

    The baseline names the configs it was captured from, so the check runs over
    those names: each of them must still be there and still record the same
    thing. A config added since is not in the baseline and stays free - a screen
    config for a new source, or a mix, must not have to match it.
    """
    expected = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert expected, "the baseline no longer holds the configs it was captured from"
    for name, recorded in sorted(expected.items()):
        path = REPO_ROOT / "configs" / "train" / name
        assert path.is_file(), f"{name}: the config the baseline was captured from is gone"
        assert TrainConfig.from_file(path).to_dict() == recorded, name


def test_the_mix_form_of_to_dict_records_the_shares_and_nothing_about_the_single_form():
    recorded = mix_config(
        datasets=[{"name": "ds-a", "clips": 4}, {"name": "ds-b", "clips": 6}]
    ).to_dict()
    assert recorded["datasets"] == [{"name": "ds-a", "clips": 4}, {"name": "ds-b", "clips": 6}]
    assert "dataset" not in recorded and "clips" not in recorded


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
        refuse_finished_run(run_dir("example", tmp_path))
    # The same rule guards the directory itself, whatever calls it.
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


def test_a_run_writes_the_model_config_that_evaluates_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a runs root given as a relative path
    directory = prepare_run_dir(run_dir("example", Path("runs")))
    path = write_model_config(directory, name="example", gpu="GPU-abc")
    text = path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    # Absolute, so the config works from anywhere, whatever runs root was given.
    expected = tmp_path / "runs" / "train" / "example" / MODEL_NAME
    assert raw["serve"]["checkpoint"] == str(expected)
    assert raw["serve"]["gpu"] == "GPU-abc"
    assert raw["name"] == "example" and raw["model"] == "example"
    # No endpoint and no address: this config serves a local checkpoint (D-016).
    assert "base_url_env" not in text and "http" not in text


# --- the whole sequence, with the fine-tune replaced ------------------------------


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
    # A single-dataset record keeps exactly the keys it always had, plus the
    # leakage guard's block: what was checked, and what could not be.
    assert set(record) == {
        "name",
        "config",
        "dataset",
        "clips",
        "clips_available",
        "sampled_ids_sha256",
        "base_model",
        "base_model_identity",
        "resumed",
        "git_commit",
        "leakage_guard",
        "final_loss",
        "updates",
        "wall_seconds",
        "updates_per_second",
        "peak_vram_gib",
        "device",
    }
    assert record["leakage_guard"] == {
        "evaluation_datasets_checked": [],
        "missing": ["guard-eval"],
        "clips_checked": 6,
    }
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


class InterruptedStep:
    """A run that stops the way a crash does: after writing a resume slot."""

    def __call__(self, training: Training, *, print_line=print):
        (training.directory / RESUME_NAME).write_bytes(b"optimizer state")
        raise RuntimeError("the machine went down")


def stopped_run(tmp_path, dataset, **overrides) -> Path:
    """Leave a run directory holding a resume slot and nothing finished."""
    monkeypatch_hint = overrides.pop("monkeypatch")
    monkeypatch_hint.setattr(runner, "train_checkpoint", InterruptedStep())
    with pytest.raises(RuntimeError, match="went down"):
        runner.run_training(
            run_config(tmp_path, clips=4, updates=2, **overrides),
            data_dir=dataset,
            runs_root=tmp_path / "runs",
            print_line=lambda _: None,
        )
    return run_dir("example", tmp_path / "runs")


def test_a_run_that_stopped_is_continued_from_its_slot(tmp_path, monkeypatch):
    dataset = write_dataset(tmp_path / "data", "fleurs-fr", 20)
    stopped_run(tmp_path, dataset, monkeypatch=monkeypatch)
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    directory = runner.run_training(
        run_config(tmp_path, clips=4, updates=2),
        data_dir=dataset,
        runs_root=tmp_path / "runs",
        print_line=lambda _: None,
    )
    assert step.calls[0].resuming is True
    assert train_json(directory)["resumed"] is True


class StoppedStep:
    """A run asked to stop: it writes its slot and says it stopped."""

    def __call__(self, training: Training, *, print_line=print):
        (training.directory / RESUME_NAME).write_bytes(b"optimizer state at update 1")
        return FakeResult(updates=1, stopped=True)


def test_a_stopped_run_keeps_its_slot_writes_no_record_and_the_same_command_resumes(
    tmp_path, monkeypatch
):
    dataset = write_dataset(tmp_path / "data", "fleurs-fr", 20)
    loaded = run_config(tmp_path, clips=4, updates=2)
    monkeypatch.setattr(runner, "train_checkpoint", StoppedStep())
    with pytest.raises(Stopped, match="stopped after update 1 of 2") as stopped:
        runner.run_training(
            loaded, data_dir=dataset, runs_root=tmp_path / "runs", print_line=lambda _: None
        )
    assert "run the same command again to resume" in str(stopped.value)
    directory = run_dir("example", tmp_path / "runs")
    assert (directory / RESUME_NAME).is_file()
    assert not is_finished(directory)
    assert not (directory / MODEL_CONFIG).exists()

    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    runner.run_training(
        loaded, data_dir=dataset, runs_root=tmp_path / "runs", print_line=lambda _: None
    )
    assert step.calls[0].resuming is True
    assert is_finished(directory) and not (directory / RESUME_NAME).exists()


def test_a_stop_asked_before_training_starts_trains_nothing(tmp_path, monkeypatch):
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    request_stop()
    with pytest.raises(Stopped, match="before it started"):
        runner.run_training(
            run_config(tmp_path, clips=4, updates=2),
            data_dir=write_dataset(tmp_path / "data", "fleurs-fr", 20),
            runs_root=tmp_path / "runs",
            print_line=lambda _: None,
        )
    assert step.calls == []


def test_what_a_run_was_started_as_is_on_disk_before_it_trains(tmp_path, monkeypatch):
    """``config.json`` is what a later resume is checked against."""
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    dataset = write_dataset(tmp_path / "data", "fleurs-fr", 20)
    loaded = run_config(tmp_path, clips=4, updates=2)
    directory = runner.run_training(
        loaded, data_dir=dataset, runs_root=tmp_path / "runs", print_line=lambda _: None
    )
    started = read_run_config(directory)
    assert started["config"] == loaded.to_dict()
    assert started["base_model_identity"].startswith("sha256=")
    assert started["sampled_ids_sha256"] == step.calls[0].clips.ids_sha256
    assert started["clips"] == len(step.calls[0].clips.clips)
    # The guard's block is in train.json only: config.json is compared field by
    # field to resume an old run, and a new key there would refuse it.
    assert "leakage_guard" not in started
    # The run that went on from the base model is not a resumed one.
    assert train_json(directory)["resumed"] is False


def test_a_resume_with_another_config_is_refused(tmp_path, monkeypatch):
    """A slot belongs to one run; continuing it with another recipe is a third run."""
    dataset = write_dataset(tmp_path / "data", "fleurs-fr", 20)
    stopped_run(tmp_path, dataset, monkeypatch=monkeypatch)
    monkeypatch.setattr(runner, "train_checkpoint", FakeStep())
    with pytest.raises(TrainRunError, match="config.updates changed"):
        runner.run_training(
            run_config(tmp_path, clips=4, updates=3),
            data_dir=dataset,
            runs_root=tmp_path / "runs",
            print_line=lambda _: None,
        )
    # The old config still resumes, and --fresh starts the new one over it.
    runner.run_training(
        run_config(tmp_path, clips=4, updates=2),
        data_dir=dataset,
        runs_root=tmp_path / "runs",
        print_line=lambda _: None,
    )
    runner.run_training(
        run_config(tmp_path, clips=4, updates=3),
        data_dir=dataset,
        runs_root=tmp_path / "runs",
        fresh=True,
        print_line=lambda _: None,
    )
    assert train_json(run_dir("example", tmp_path / "runs"))["config"]["updates"] == 3


def test_a_slot_that_says_nothing_about_its_run_is_refused(tmp_path, monkeypatch):
    """An unverifiable slot is not resumed: nothing says it is this config's run."""
    monkeypatch.setattr(runner, "train_checkpoint", FakeStep())
    dataset = write_dataset(tmp_path / "data", "fleurs-fr", 20)
    directory = run_dir("example", tmp_path / "runs")
    prepare_run_dir(directory)
    (directory / RESUME_NAME).write_bytes(b"optimizer state")
    with pytest.raises(TrainRunError, match="no config.json"):
        runner.run_training(
            run_config(tmp_path, clips=4, updates=2),
            data_dir=dataset,
            runs_root=tmp_path / "runs",
        )


def test_a_new_config_over_a_stopped_run_that_has_no_slot_starts_over(tmp_path, monkeypatch):
    """There is nothing to continue, so the new config simply runs."""
    monkeypatch.setattr(runner, "train_checkpoint", FakeStep())
    dataset = write_dataset(tmp_path / "data", "fleurs-fr", 20)
    directory = run_dir("example", tmp_path / "runs")
    prepare_run_dir(directory)
    write_run_config(directory, {"config": {"updates": 99}})
    runner.run_training(
        run_config(tmp_path, clips=4, updates=2),
        data_dir=dataset,
        runs_root=tmp_path / "runs",
        print_line=lambda _: None,
    )
    assert read_run_config(directory)["config"]["updates"] == 2


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


# --- the two learning rates, checked before training starts -----------------------


class FakeParameter:
    """Something an optimizer group can hold; the check compares identities and flags."""

    def __init__(self, requires_grad: bool = True):
        self.requires_grad = requires_grad


class FakeModule:
    def __init__(self, parameters):
        self._parameters = list(parameters)

    def parameters(self):
        return self._parameters


class FakeModel:
    def __init__(self, groups, **parts):
        self._optimizer = type("Optimizer", (), {"param_groups": groups})()
        for name, module in parts.items():
            setattr(self, name, module)

    def named_parameters(self):
        """Every weight the model holds, under the name a real model would give it."""
        return [
            (f"{name}.weight", parameter)
            for name, module in vars(self).items()
            if isinstance(module, FakeModule)
            for parameter in module.parameters()
        ]


# The two rates of the example config: the encoder, then the heads.
RATES = {"encoder": 1.0e-5, "decoder": 3.0e-5, "joint": 3.0e-5}


def fake_model(rates: dict | None = None, weightless: tuple = ()) -> FakeModel:
    """A model whose named parts sit in one group each, at the rate given.

    ``weightless`` names parts that carry no parameters, which the feature
    extractor of this family is.
    """
    groups, parts = [], {}
    for part, rate in (rates or RATES).items():
        parameter = FakeParameter()
        groups.append({"params": [parameter], "lr": rate})
        parts[part] = FakeModule([parameter])
    for part in weightless:
        parts[part] = FakeModule([])
    return FakeModel(groups, **parts)


def test_the_dataset_a_run_uses_is_the_one_the_recipe_asks_for(tmp_path):
    """The manifest is read the project's way, not NeMo's default way."""
    path = tmp_path / "manifest.jsonl"
    built = dataset_config(config(clips=16, batch_size=8), path)
    assert built["manifest_filepath"] == str(path)
    assert built["batch_size"] == 8
    assert built["use_lhotse"] is False and built["is_tarred"] is False
    # <bos>/<eos> are not part of a transcript: the tokenizer of this family has
    # none, and a fix-tune must not depend on that staying true.
    assert built["use_start_end_token"] is False


def test_the_rate_check_reads_the_rate_the_schedule_started_from():
    """NeMo steps its warmup policy once while building it, so ``lr`` has moved.

    Comparing ``lr`` would refuse a run whose rates were exactly right.
    """
    from audiolab.train.settings import check_rates

    model = fake_model(weightless=("preprocessor",))
    for group in model._optimizer.param_groups:
        group["initial_lr"] = group["lr"]
        group["lr"] = group["lr"] / 10  # what one warmup step leaves behind
    check_rates(model, config())


def test_the_rate_check_accepts_a_model_grouped_the_way_the_run_asks():
    from audiolab.train.settings import check_rates

    check_rates(fake_model(weightless=("preprocessor",)), config())


def test_a_part_without_weights_is_allowed_to_have_an_empty_group():
    """The feature extractor of this family computes its filters, so it has none."""
    from audiolab.train.settings import check_rates

    model = fake_model(weightless=("preprocessor",))
    model._optimizer.param_groups.insert(0, {"params": [], "lr": 3.0e-5})
    check_rates(model, config())


def test_a_rate_that_landed_on_the_wrong_part_is_refused():
    from audiolab.train.settings import check_rates

    model = fake_model(weightless=("preprocessor",))
    # The encoder's parameters end up in a group that carries the heads' rate.
    encoder = {id(p) for p in model.encoder.parameters()}
    for group in model._optimizer.param_groups:
        if encoder & {id(p) for p in group["params"]}:
            group["lr"] = 3.0e-5
    with pytest.raises(ValueError, match="train at"):
        check_rates(model, config())


def test_a_model_without_one_of_the_named_parts_is_refused():
    from audiolab.train.settings import check_rates

    model = fake_model({"encoder": 1.0e-5, "joint": 3.0e-5}, weightless=("preprocessor",))
    with pytest.raises(ValueError, match="no part named 'decoder'"):
        check_rates(model, config())


def test_a_third_rate_in_the_optimizer_is_refused():
    from audiolab.train.settings import check_rates

    model = fake_model(weightless=("preprocessor",))
    model._optimizer.param_groups.append({"params": [FakeParameter()], "lr": 1.0})
    with pytest.raises(ValueError, match="did not ask for"):
        check_rates(model, config())


def test_a_parameter_no_named_part_owns_is_refused():
    """It would train at the fallback rate, and the run would not mean what it says."""
    from audiolab.train.settings import check_rates

    model = fake_model(weightless=("preprocessor",))
    model.extra_head = FakeModule([FakeParameter()])
    with pytest.raises(ValueError, match="belongs to no part this run sets a rate for"):
        check_rates(model, config())


def test_a_frozen_parameter_is_refused():
    """A run that says full fine-tune must not leave weights behind."""
    from audiolab.train.settings import check_rates

    model = fake_model(weightless=("preprocessor",))
    model.encoder._parameters[0].requires_grad = False
    with pytest.raises(ValueError, match="is frozen"):
        check_rates(model, config())


def test_a_part_whose_weights_are_frozen_is_refused():
    """NeMo leaves a frozen parameter out of the optimizer: no rate can reach it."""
    from audiolab.train.settings import check_rates

    model = fake_model(weightless=("preprocessor",))
    for group in model._optimizer.param_groups:
        group["params"] = []  # what NeMo builds when a part's weights are all frozen
    with pytest.raises(ValueError, match="holds no parameter of"):
        check_rates(model, config())


def test_an_optimizer_parameter_of_no_named_part_is_refused():
    """Even at one of the two rates, nothing outside the named parts may train."""
    from audiolab.train.settings import check_rates

    model = fake_model(weightless=("preprocessor",))
    model._optimizer.param_groups.append({"params": [FakeParameter()], "lr": 3.0e-5})
    with pytest.raises(ValueError, match="parameters of no named part"):
        check_rates(model, config())
