"""Training on a mix of datasets: the per-dataset draw, the leakage guard, the records.

None of this needs PyTorch, NeMo or a GPU: the fine-tune sits behind one call the
tests replace, the guard reads manifests, and the config is pure data (tested in
test_train.py). The synthetic datasets are what the tests write, with ``make_clip``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402
from helpers import make_clip  # noqa: E402

from audiolab.data.panels import (  # noqa: E402
    DEFAULT_PANELS_DIR,
    Panel,
    SuiteError,
    load_suite,
)
from audiolab.data.store import manifest_id_sha256, write_manifest  # noqa: E402
from audiolab.train import runner  # noqa: E402
from audiolab.train.config import DatasetShare, TrainConfig  # noqa: E402
from audiolab.train.data import (  # noqa: E402
    TrainDataError,
    mix_seed,
    select_clips,
    select_mix,
    write_nemo_manifest,
)
from audiolab.train.guard import check_leakage, evaluation_datasets  # noqa: E402
from audiolab.train.run import (  # noqa: E402
    MANIFEST_NAME,
    MODEL_NAME,
    TRAIN_JSON,
    TrainRunError,
    is_finished,
    read_run_config,
    run_dir,
)
from audiolab.train.runner import Training  # noqa: E402

MINIMAL = {
    "base_model": "/models/base.nemo",
    "updates": 4,
    "batch_size": 2,
    "lr_encoder": 1.0e-5,
    "lr_heads": 3.0e-5,
}
MIX_MINIMAL = {**MINIMAL, "datasets": [{"name": "ds-a", "clips": 4}, {"name": "ds-b", "clips": 4}]}


# --- the draw -------------------------------------------------------------------


def write_dataset(root: Path, name: str, count: int) -> None:
    write_manifest(root / name, [make_clip(f"{name}:{index:04d}") for index in range(count)])


def test_select_mix_is_deterministic_for_the_same_seed(tmp_path):
    write_dataset(tmp_path, "ds-a", 20)
    write_dataset(tmp_path, "ds-b", 20)
    shares = (DatasetShare("ds-a", 5), DatasetShare("ds-b", 5))
    first = select_mix(shares, seed=7, max_seconds=20.0, data_dir=tmp_path)
    second = select_mix(shares, seed=7, max_seconds=20.0, data_dir=tmp_path)
    assert [clip.id for clip in first.clips] == [clip.id for clip in second.clips]
    assert first.ids_sha256 == second.ids_sha256
    assert (
        first.ids_sha256
        != select_mix(shares, seed=8, max_seconds=20.0, data_dir=tmp_path).ids_sha256
    )


def test_each_dataset_of_a_mix_is_drawn_as_if_it_were_alone_at_its_own_seed(tmp_path):
    """The per-dataset seed is the whole point: the mix is a union of solo draws."""
    write_dataset(tmp_path, "ds-a", 20)
    write_dataset(tmp_path, "ds-b", 20)
    mixed = select_mix(
        (DatasetShare("ds-a", 5), DatasetShare("ds-b", 5)),
        seed=7,
        max_seconds=20.0,
        data_dir=tmp_path,
    )
    for per in mixed.datasets:
        solo = select_clips(
            [make_clip(f"{per.name}:{index:04d}") for index in range(20)],
            count=5,
            seed=mix_seed(7, per.name),
            max_seconds=20.0,
        )
        assert [clip.id for clip in per.clips] == [clip.id for clip in solo.clips]


def test_adding_a_third_dataset_leaves_the_clips_drawn_from_the_first_two_unchanged(tmp_path):
    for name in ("ds-a", "ds-b", "ds-c"):
        write_dataset(tmp_path, name, 20)
    two = select_mix(
        (DatasetShare("ds-a", 5), DatasetShare("ds-b", 5)),
        seed=7,
        max_seconds=20.0,
        data_dir=tmp_path,
    )
    three = select_mix(
        (DatasetShare("ds-a", 5), DatasetShare("ds-b", 5), DatasetShare("ds-c", 5)),
        seed=7,
        max_seconds=20.0,
        data_dir=tmp_path,
    )
    by_name = {per.name: [clip.id for clip in per.clips] for per in three.datasets}
    assert by_name["ds-a"] == [clip.id for clip in two.datasets[0].clips]
    assert by_name["ds-b"] == [clip.id for clip in two.datasets[1].clips]
    assert len(three.clips) == 15


def test_a_mix_dataset_with_too_few_usable_clips_names_that_dataset(tmp_path):
    write_dataset(tmp_path, "ds-a", 2)
    write_dataset(tmp_path, "ds-b", 20)
    with pytest.raises(TrainDataError, match="ds-a"):
        select_mix(
            (DatasetShare("ds-a", 5), DatasetShare("ds-b", 5)),
            seed=7,
            max_seconds=20.0,
            data_dir=tmp_path,
        )


def test_a_mix_dataset_that_was_never_prepared_says_so(tmp_path):
    write_dataset(tmp_path, "ds-a", 20)
    with pytest.raises(TrainDataError, match="ds-b"):
        select_mix(
            (DatasetShare("ds-a", 5), DatasetShare("ds-b", 5)),
            seed=7,
            max_seconds=20.0,
            data_dir=tmp_path,
        )


def test_duplicate_audio_across_two_datasets_of_a_mix_is_refused(tmp_path):
    """The same recording in two datasets would train the same audio twice."""
    shared = [hashlib.sha256(f"recording {i}".encode()).hexdigest() for i in range(6)]
    write_manifest(
        tmp_path / "ds-a", [make_clip(f"ds-a:{i:04d}", sha256=shared[i]) for i in range(6)]
    )
    write_manifest(
        tmp_path / "ds-b", [make_clip(f"ds-b:{i:04d}", sha256=shared[i]) for i in range(6)]
    )
    with pytest.raises(
        TrainDataError, match="6 recordings appear in more than one dataset, first 5"
    ):
        select_mix(
            (DatasetShare("ds-a", 6), DatasetShare("ds-b", 6)),
            seed=7,
            max_seconds=20.0,
            data_dir=tmp_path,
        )


def test_the_same_audio_twice_inside_one_dataset_is_not_the_mixs_business(tmp_path):
    """A repetition inside one dataset is that dataset's to fix, not the mix's to refuse."""
    shared = hashlib.sha256(b"twice").hexdigest()
    write_manifest(
        tmp_path / "ds-a",
        [
            make_clip("ds-a:0001", sha256=shared),
            make_clip("ds-a:0002", sha256=shared),
            make_clip("ds-a:0003"),
        ],
    )
    write_dataset(tmp_path, "ds-b", 4)
    mix = select_mix(
        (DatasetShare("ds-a", 3), DatasetShare("ds-b", 2)),
        seed=1,
        max_seconds=20.0,
        data_dir=tmp_path,
    )
    assert len(mix.clips) == 5


def test_the_mix_clips_are_sorted_by_id_and_per_dataset_results_are_kept(tmp_path):
    write_dataset(tmp_path, "ds-a", 10)
    write_dataset(tmp_path, "ds-b", 10)
    mix = select_mix(
        (DatasetShare("ds-a", 3), DatasetShare("ds-b", 3)),
        seed=3,
        max_seconds=20.0,
        data_dir=tmp_path,
    )
    assert [clip.id for clip in mix.clips] == sorted(clip.id for clip in mix.clips)
    assert [per.name for per in mix.datasets] == ["ds-a", "ds-b"]
    for per in mix.datasets:
        assert per.ids_sha256 == manifest_id_sha256(per.clips)
    assert mix.ids_sha256 == manifest_id_sha256(mix.clips)


def test_write_nemo_manifest_pairs_put_each_clip_in_its_own_dataset(tmp_path):
    """A mix's manifest is one format: the pair says which directory the audio is in.

    A single-dataset run is the same format over the one directory - a list of
    pairs of the one directory - so its bytes are what they always were.
    """
    clips_a = [make_clip(f"ds-a:{i:04d}") for i in range(2)]
    clips_b = [make_clip(f"ds-b:{i:04d}") for i in range(2)]
    path = tmp_path / "manifest.jsonl"
    written = write_nemo_manifest(
        path, [(clip, tmp_path / clip.id.split(":")[0]) for clip in [*clips_a, *clips_b]]
    )
    assert written == 4
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {Path(line["audio_filepath"]).parent.parent.name for line in lines} == {"ds-a", "ds-b"}
    assert all(Path(line["audio_filepath"]).is_absolute() for line in lines)
    # The single form: the pairs form over the one directory, same fields and paths.
    single = [make_clip(f"ds:{i:04d}") for i in range(3)]
    assert write_nemo_manifest(path, [(clip, tmp_path / "ds") for clip in single]) == 3
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [line["audio_filepath"] for line in lines] == [
        str((tmp_path / "ds" / clip.audio).resolve()) for clip in single
    ]
    assert set(lines[0]) == {"audio_filepath", "duration", "text"}


# --- the leakage guard ------------------------------------------------------------


def guard_suite(tmp_path: Path, names: list[str]) -> tuple[Path, Path]:
    """A suite with one full panel per evaluation dataset, for the guard to check."""
    panels_dir = tmp_path / "panels"
    panels_dir.mkdir(exist_ok=True)
    for name in names:
        (panels_dir / f"{name}-full.txt").write_text(
            Panel(
                name="full",
                dataset=name,
                ids=[f"{name}:0000"],
                items=1,
                seed=1,
                manifest_sha256="0" * 64,
                clips_sha256="0" * 64,
                dataset_line=f"{name} (1 clip in 1 group; test 1)",
                built="2026-01-01",
            ).text(),
            encoding="utf-8",
        )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: guard\npanels:\n" + "\n".join(f"  - {name}-full" for name in names) + "\n",
        encoding="utf-8",
    )
    return suite, panels_dir


def test_the_guard_defaults_resolve_to_the_repos_full_suite_and_its_panels():
    """The default path - the one the sequence tests monkeypatch around - is the repo's.

    Pinned so that moving or renaming the suite or the panels the guard relies on
    fails here, not at the first production training run. Both are given the way
    the eval command gives them - relative to the working directory, which is the
    repository root the documented commands run from - and the panels directory
    is the constant that command defaults to.

    The datasets behind the panels are not pinned: one panel added to the suite
    (a new evaluation set, plan 01) is the suite's business, and the guard checks
    whatever it names. Reading them here proves the suite and every panel it
    names are there, and that each dataset is listed once.
    """
    from audiolab.train.guard import FULL_PANELS, FULL_SUITE

    repo = Path(__file__).resolve().parents[1]
    assert FULL_SUITE == Path("configs/suites/full.yaml")
    assert FULL_PANELS == DEFAULT_PANELS_DIR
    assert (repo / FULL_SUITE).is_file()
    assert (repo / FULL_PANELS).is_dir()
    datasets = evaluation_datasets(FULL_SUITE, FULL_PANELS)
    assert datasets and all(datasets), "the full suite names no dataset"
    assert len(set(datasets)) == len(datasets)


def test_the_guard_reads_the_suite_the_way_the_eval_command_does(tmp_path):
    """One reader for both: a suite the eval command refuses is refused here too.

    The guard checks the file the eval command scores, so a change to the suite
    format must reach the two together instead of letting them drift apart.
    """
    suite = tmp_path / "suite.yaml"
    suite.write_text("name: guard\npanels: []\n", encoding="utf-8")
    with pytest.raises(SuiteError, match="must hold a non-empty 'panels' list"):
        load_suite(suite)
    with pytest.raises(TrainRunError, match="must hold a non-empty 'panels' list"):
        evaluation_datasets(suite, tmp_path)
    with pytest.raises(TrainRunError, match="is missing"):
        evaluation_datasets(tmp_path / "nope.yaml", tmp_path)


def test_no_committed_training_config_names_an_evaluation_dataset():
    """The configs in configs/train/ train, they are not scored: none names an eval dataset.

    The guard refuses a run whose clip ids are prefixed by an evaluation dataset's
    name, so the first command the documentation shows must not ask for one. The
    check needs no data and no run: it compares the names the committed configs
    ask for with the datasets behind the full suite's panels.
    """
    from audiolab.train.guard import FULL_PANELS, FULL_SUITE

    scored = set(evaluation_datasets(FULL_SUITE, FULL_PANELS))
    repo = Path(__file__).resolve().parents[1]
    configs = sorted((repo / "configs" / "train").glob("*.yaml"))
    assert configs, "the committed training configs are gone"
    for path in configs:
        config = TrainConfig.from_file(path)
        names = [share.name for share in config.datasets] or [config.dataset]
        assert names, f"{path.name} names no dataset"
        assert not (scored & set(names)), (
            f"{path.name} trains on {names}, and {sorted(scored & set(names))} is scored by "
            "the full suite"
        )


def test_a_training_clip_that_is_the_audio_of_an_evaluation_clip_is_refused(tmp_path):
    data = tmp_path / "data"
    eval_clips = [make_clip(f"eval-ds:{i:04d}") for i in range(3)]
    write_manifest(data / "eval-ds", eval_clips)
    suite, panels = guard_suite(tmp_path, ["eval-ds"])
    training = [make_clip("train-ds:0001", sha256=eval_clips[0].sha256)]
    with pytest.raises(TrainDataError, match="is the audio of an evaluation clip"):
        check_leakage(
            training, data_dir=data, suite_path=suite, panels_dir=panels, print_line=lambda _: None
        )


def test_a_training_clip_named_after_an_evaluation_dataset_is_refused(tmp_path):
    """The name alone is the leak: the dataset does not even have to be prepared here."""
    suite, panels = guard_suite(tmp_path, ["eval-ds"])
    with pytest.raises(TrainDataError, match="named after the evaluation dataset"):
        check_leakage(
            [make_clip("eval-ds:0001")],
            data_dir=tmp_path / "data",
            suite_path=suite,
            panels_dir=panels,
            print_line=lambda _: None,
        )


def test_an_evaluation_dataset_that_is_not_prepared_here_only_warns(tmp_path):
    data = tmp_path / "data"
    write_manifest(data / "eval-ds", [make_clip(f"eval-ds:{i:04d}") for i in range(5)])
    suite, panels = guard_suite(tmp_path, ["eval-ds", "absent-ds"])
    lines: list[str] = []
    result = check_leakage(
        [make_clip(f"train-ds:{i:04d}") for i in range(3)],
        data_dir=data,
        suite_path=suite,
        panels_dir=panels,
        print_line=lines.append,
    )
    assert result.evaluation_datasets_checked == ("eval-ds",)
    assert result.missing == ("absent-ds",)
    assert result.clips_checked == 3
    assert "absent-ds is not prepared here" in " ".join(lines)


def test_the_summary_line_says_what_was_checked_and_what_is_missing(tmp_path):
    data = tmp_path / "data"
    write_manifest(data / "eval-ds", [make_clip(f"eval-ds:{i:04d}") for i in range(5)])
    suite, panels = guard_suite(tmp_path, ["eval-ds", "absent-ds"])
    lines: list[str] = []
    check_leakage(
        [make_clip(f"train-ds:{i:04d}") for i in range(3)],
        data_dir=data,
        suite_path=suite,
        panels_dir=panels,
        print_line=lines.append,
    )
    assert lines[-1] == (
        "leakage guard: 3 training clips checked against 5 evaluation clips of 1 datasets "
        "(missing: absent-ds)"
    )


def test_the_guard_checks_only_the_suites_datasets_in_the_suites_order(tmp_path):
    data = tmp_path / "data"
    write_manifest(data / "one", [make_clip("one:0000")])
    write_manifest(data / "two", [make_clip("two:0000")])
    write_manifest(data / "three", [make_clip("three:0000")])
    suite, panels = guard_suite(tmp_path, ["two", "one", "three"])
    result = check_leakage(
        [make_clip("train-ds:0001")],
        data_dir=data,
        suite_path=suite,
        panels_dir=panels,
        print_line=lambda _: None,
    )
    assert result.evaluation_datasets_checked == ("two", "one", "three")


def test_group_ids_are_not_compared_across_corpora(tmp_path):
    """A training clip that shares an evaluation clip's group is refused on no account of it."""
    data = tmp_path / "data"
    eval_clip = make_clip("eval-ds:0001", group="speaker-x")
    write_manifest(data / "eval-ds", [eval_clip])
    suite, panels = guard_suite(tmp_path, ["eval-ds"])
    result = check_leakage(
        [make_clip("train-ds:0001", group="speaker-x")],
        data_dir=data,
        suite_path=suite,
        panels_dir=panels,
        print_line=lambda _: None,
    )
    assert result.clips_checked == 1


# --- the whole sequence, with the fine-tune replaced --------------------------------


@dataclass
class FakeResult:
    final_loss: float = 1.5
    updates: int = 64
    wall_seconds: float = 32.0
    peak_vram_gib: float = 7.25
    device: str = "Fake GPU"
    stopped: bool = False


class FakeStep:
    """The heavy step, replaced: it writes the two files a real run writes."""

    def __init__(self) -> None:
        self.calls: list[Training] = []

    def __call__(self, training: Training, *, print_line=print):
        self.calls.append(training)
        (training.directory / MODEL_NAME).write_bytes(b"fine-tuned weights")
        (training.directory / "train-log.jsonl").write_text(
            '{"update": 64, "loss": 1.5}\n', encoding="utf-8"
        )
        return FakeResult()


@pytest.fixture(autouse=True)
def no_gpu_needed(monkeypatch):
    monkeypatch.setattr(runner, "select_gpu", lambda spec: None)


@pytest.fixture(autouse=True)
def the_guard_checks_a_suite_the_tests_control(tmp_path, monkeypatch):
    """As in test_train.py: the guard runs, against a suite the tests own."""
    suite, panels = guard_suite(tmp_path, ["guard-eval"])
    monkeypatch.setattr(
        runner, "check_leakage", partial(check_leakage, suite_path=suite, panels_dir=panels)
    )


def base_model(tmp_path: Path) -> Path:
    base = tmp_path / "base.nemo"
    base.write_bytes(b"base weights")
    return base


def mix_config(tmp_path: Path, **overrides) -> TrainConfig:
    return TrainConfig.from_mapping(
        {**MIX_MINIMAL, "base_model": str(base_model(tmp_path)), **overrides}, name="mixexample"
    )


def write_mix_data(root: Path) -> Path:
    write_dataset(root, "ds-a", 20)
    write_dataset(root, "ds-b", 20)
    return root


def test_a_mix_run_draws_per_dataset_and_records_the_mix(tmp_path, monkeypatch):
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    data = write_mix_data(tmp_path / "data")
    directory = runner.run_training(
        mix_config(tmp_path), data_dir=data, runs_root=tmp_path / "runs", print_line=lambda _: None
    )
    assert directory == run_dir("mixexample", tmp_path / "runs")
    assert is_finished(directory)
    mix = step.calls[0].clips
    assert [per.name for per in mix.datasets] == ["ds-a", "ds-b"]
    assert len(mix.clips) == 8

    # The manifest has one line per clip, each under its own dataset's directory.
    manifest = [json.loads(line) for line in (directory / MANIFEST_NAME).read_text().splitlines()]
    assert len(manifest) == 8
    directories = {Path(line["audio_filepath"]).parent.parent.name for line in manifest}
    assert directories == {"ds-a", "ds-b"}
    assert all(Path(line["audio_filepath"]).is_absolute() for line in manifest)

    record = json.loads((directory / TRAIN_JSON).read_text(encoding="utf-8"))
    # A mix record holds the per-dataset numbers and the whole, and no dataset key.
    assert "dataset" not in record
    assert record["clips"] == 8
    assert record["sampled_ids_sha256"] == mix.ids_sha256
    assert [entry["name"] for entry in record["datasets"]] == ["ds-a", "ds-b"]
    for entry, per in zip(record["datasets"], mix.datasets, strict=True):
        assert entry == {
            "name": per.name,
            "clips": len(per.clips),
            "clips_available": per.available,
            "sampled_ids_sha256": per.ids_sha256,
        }
    assert record["leakage_guard"] == {
        "evaluation_datasets_checked": [],
        "missing": ["guard-eval"],
        "clips_checked": 8,
    }

    # config.json: the mix config, the per-dataset hashes, and no guard block.
    started = read_run_config(directory)
    assert "leakage_guard" not in started
    assert started["config"]["datasets"] == [
        {"name": "ds-a", "clips": 4},
        {"name": "ds-b", "clips": 4},
    ]
    assert "dataset" not in started["config"] and "clips" not in started["config"]
    assert started["clips"] == 8
    assert started["sampled_ids_sha256"] == record["sampled_ids_sha256"]
    assert started["datasets"] == [
        {"name": "ds-a", "sampled_ids_sha256": mix.datasets[0].ids_sha256},
        {"name": "ds-b", "sampled_ids_sha256": mix.datasets[1].ids_sha256},
    ]


def test_a_mix_run_that_would_train_on_evaluation_audio_is_refused_before_anything_is_written(
    tmp_path, monkeypatch
):
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    data = tmp_path / "data"
    write_mix_data(data)
    write_dataset(data, "eval-ds", 20)
    # This run's guard checks a suite that scores eval-ds: training on it is the leak.
    suite, panels = guard_suite(tmp_path, ["eval-ds"])
    monkeypatch.setattr(
        runner, "check_leakage", partial(check_leakage, suite_path=suite, panels_dir=panels)
    )
    config = mix_config(
        tmp_path, datasets=[{"name": "eval-ds", "clips": 4}, {"name": "ds-b", "clips": 4}]
    )
    with pytest.raises(TrainDataError, match="leakage guard"):
        runner.run_training(config, data_dir=data, runs_root=tmp_path / "runs")
    assert step.calls == []
    assert not run_dir("mixexample", tmp_path / "runs").exists()


def test_the_single_form_still_prints_its_usable_clips_line_and_the_mix_prints_one_per_dataset(
    tmp_path, monkeypatch
):
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    data = tmp_path / "data"
    write_dataset(data, "fleurs-fr", 20)
    lines: list[str] = []
    runner.run_training(
        TrainConfig.from_mapping(
            {
                **MINIMAL,
                "base_model": str(base_model(tmp_path)),
                "dataset": "fleurs-fr",
                "clips": 4,
            },
            name="example",
        ),
        data_dir=data,
        runs_root=tmp_path / "runs",
        print_line=lines.append,
    )
    assert any("4 clips of fleurs-fr" in line and "were usable" in line for line in lines)

    write_mix_data(data)
    lines.clear()
    runner.run_training(
        mix_config(tmp_path), data_dir=data, runs_root=tmp_path / "runs", print_line=lines.append
    )
    assert any("4 clips of ds-a" in line for line in lines)
    assert any("4 clips of ds-b" in line for line in lines)


def test_a_single_dataset_run_records_nothing_about_a_mix(tmp_path, monkeypatch):
    """The single form's record must not carry the mix's keys, even when a guard runs."""
    monkeypatch.setattr(runner, "train_checkpoint", FakeStep())
    data = tmp_path / "data"
    write_dataset(data, "fleurs-fr", 20)
    directory = runner.run_training(
        TrainConfig.from_mapping(
            {
                **MINIMAL,
                "base_model": str(base_model(tmp_path)),
                "dataset": "fleurs-fr",
                "clips": 4,
            },
            name="example",
        ),
        data_dir=data,
        runs_root=tmp_path / "runs",
        print_line=lambda _: None,
    )
    record = json.loads((directory / TRAIN_JSON).read_text(encoding="utf-8"))
    assert record["dataset"] == "fleurs-fr"
    assert "datasets" not in record and "clips_available" in record
    started = read_run_config(directory)
    assert "datasets" not in started and started["config"]["dataset"] == "fleurs-fr"
