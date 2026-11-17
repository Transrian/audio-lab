"""Training on a mix of datasets: the per-dataset draw, the leakage guard, the records.

None of this needs PyTorch, NeMo or a GPU: the fine-tune sits behind one call the
tests replace, the guard reads manifests, and the config is pure data (tested in
test_train.py). The synthetic datasets are what the tests write, with ``make_clip``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest  # noqa: E402
from helpers import (  # noqa: E402
    MINIMAL,
    MIX_MINIMAL,
    FakeStep,
    guard_suite,
    make_clip,
    point_the_guard_at_a_test_suite,
)

from audiolab.data.duplicates import duplicate_classes, peertube_talks  # noqa: E402
from audiolab.data.panels import (  # noqa: E402
    DEFAULT_PANELS_DIR,
    SuiteError,
    load_suite,
)
from audiolab.data.store import (  # noqa: E402
    manifest_id_sha256,
    read_manifest,
    write_manifest,
)
from audiolab.train import runner  # noqa: E402
from audiolab.train.config import DatasetShare, TrainConfig, TrainConfigError  # noqa: E402
from audiolab.train.data import (  # noqa: E402
    MixExclusion,
    TrainDataError,
    mix_seed,
    select_clips,
    select_mix,
    write_nemo_manifest,
)
from audiolab.train.guard import check_leakage, evaluation_datasets  # noqa: E402
from audiolab.train.run import (  # noqa: E402
    MANIFEST_NAME,
    RESUME_NAME,
    TRAIN_JSON,
    TrainRunError,
    config_differences,
    is_finished,
    prepare_run_dir,
    read_run_config,
    run_dir,
    write_run_config,
)

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
        assert [clip.id for clip in per.drawn.clips] == [clip.id for clip in solo.clips]


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
    by_name = {per.name: [clip.id for clip in per.drawn.clips] for per in three.datasets}
    assert by_name["ds-a"] == [clip.id for clip in two.datasets[0].drawn.clips]
    assert by_name["ds-b"] == [clip.id for clip in two.datasets[1].drawn.clips]
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
        assert per.drawn.ids_sha256 == manifest_id_sha256(per.drawn.clips)
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


def test_the_guard_defaults_resolve_to_the_repos_full_suite_and_its_panels():
    """The default path - the one the sequence tests monkeypatch around - is the repo's.

    Pinned so that moving or renaming the suite or the panels the guard relies on
    fails here, not at the first production training run. The defaults are
    asserted as the eval command gives them - relative to the working directory,
    which is the repository root the documented commands run from - and the
    panels directory is the constant that command defaults to. The files are
    read through the repo root, so this test passes from any working directory.

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
    datasets = evaluation_datasets(repo / FULL_SUITE, repo / FULL_PANELS)
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

    repo = Path(__file__).resolve().parents[1]
    scored = set(evaluation_datasets(repo / FULL_SUITE, repo / FULL_PANELS))
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


@pytest.fixture(autouse=True)
def no_gpu_needed(monkeypatch):
    monkeypatch.setattr(runner, "select_gpu", lambda spec: None)


@pytest.fixture(autouse=True)
def the_guard_checks_a_suite_the_tests_control(tmp_path, monkeypatch):
    """As in test_train.py: the guard runs, against a suite the tests own."""
    point_the_guard_at_a_test_suite(monkeypatch, tmp_path)


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

    # The manifest has one line per clip, each under its own dataset's directory,
    # in the order of the drawn clips, which is sorted by id.
    manifest = [json.loads(line) for line in (directory / MANIFEST_NAME).read_text().splitlines()]
    assert len(manifest) == 8
    directories = {Path(line["audio_filepath"]).parent.parent.name for line in manifest}
    assert directories == {"ds-a", "ds-b"}
    assert all(Path(line["audio_filepath"]).is_absolute() for line in manifest)
    assert [Path(line["audio_filepath"]).name for line in manifest] == [
        Path(clip.audio).name for clip in mix.clips
    ]

    record = json.loads((directory / TRAIN_JSON).read_text(encoding="utf-8"))
    # A mix record holds the per-dataset numbers and the whole, and no dataset key.
    assert "dataset" not in record
    assert record["clips"] == 8
    assert record["sampled_ids_sha256"] == mix.ids_sha256
    assert [entry["name"] for entry in record["datasets"]] == ["ds-a", "ds-b"]
    for entry, per in zip(record["datasets"], mix.datasets, strict=True):
        assert entry == {
            "name": per.name,
            "clips": len(per.drawn.clips),
            "clips_available": per.drawn.available,
            "sampled_ids_sha256": per.drawn.ids_sha256,
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
        {"name": "ds-a", "sampled_ids_sha256": mix.datasets[0].drawn.ids_sha256},
        {"name": "ds-b", "sampled_ids_sha256": mix.datasets[1].drawn.ids_sha256},
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
    """One line per dataset, in the words a single-dataset run uses for its own."""
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    data = tmp_path / "data"
    write_dataset(data, "fleurs-fr", 20)
    lines: list[str] = []
    runner.run_training(
        TrainConfig.from_mapping(
            {**MINIMAL, "base_model": str(base_model(tmp_path))},
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
    for name in ("ds-a", "ds-b"):
        assert (
            f"4 clips of {name}: 20 were usable (0 outside the duration window, 0 without text)"
            in lines
        )


def test_a_single_dataset_run_records_nothing_about_a_mix(tmp_path, monkeypatch):
    """The single form's record must not carry the mix's keys, even when a guard runs."""
    monkeypatch.setattr(runner, "train_checkpoint", FakeStep())
    data = tmp_path / "data"
    write_dataset(data, "fleurs-fr", 20)
    directory = runner.run_training(
        TrainConfig.from_mapping(
            {**MINIMAL, "base_model": str(base_model(tmp_path))},
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


# --- exclusive: the rows subtract the earlier rows' datasets ----------------------


@pytest.fixture(scope="module")
def shared_talks() -> tuple[str, str, str, str]:
    """One talk of the pinned PeerTube corpus, and three more of other classes.

    The subtraction runs on the duplicate class of the talk a clip's group
    names, over the pinned listings a preparation reads. The synthetic rows of
    this section hold clips whose groups name these talks, so what is shared
    and what is not is decided by the same rule a preparation applies: the
    first talk is the one two rows share, the others are the rows' own talks,
    one per class.
    """
    talks = peertube_talks()
    classes = duplicate_classes(talks, domain="peertube")
    shared = min(record["id"] for record in talks)
    own: list[str] = []
    for record in talks:
        class_id = classes[record["id"]]
        if class_id != classes[shared] and all(class_id != classes[talk] for talk in own):
            own.append(record["id"])
        if len(own) == 3:
            break
    return shared, *own


def write_talk_dataset(
    root: Path, name: str, shared: str, own: str, n_shared: int, n_own: int
) -> None:
    """A synthetic talk dataset whose copy of the shared talk is other prepared bytes.

    The clip ids differ by the dataset name, and the test clips take their audio
    hashes from their ids: the same talk prepared by two rows holds the same
    speech in other bytes, so only the duplicate class - not the audio hash -
    can say the two rows hold it.
    """
    clips = [
        make_clip(f"{name}:{index:04d}", group=f"{name}:{shared}") for index in range(n_shared)
    ]
    clips += [
        make_clip(f"{name}:{n_shared + index:04d}", group=f"{name}:{own}") for index in range(n_own)
    ]
    write_manifest(root / name, clips)


def test_exclusive_subtracts_the_talk_a_later_row_shares_with_an_earlier_row(
    tmp_path, shared_talks
):
    """The same talk in two rows, prepared with other bytes: the later row loses it.

    The inter-dataset check is on the audio, and the two rows hold other bytes of
    the same talk, so only the class subtraction keeps the later row from
    training the talk a second time.
    """
    shared, own_a, own_b, _ = shared_talks
    data = tmp_path / "data"
    write_talk_dataset(data, "peertube-fr-train-40k", shared, own_a, n_shared=2, n_own=10)
    write_talk_dataset(data, "peertube-edu-fr-train-40k", shared, own_b, n_shared=4, n_own=8)
    shares = (
        DatasetShare("peertube-fr-train-40k", 6),
        DatasetShare("peertube-edu-fr-train-40k", 6),
    )
    mix = select_mix(shares, seed=7, max_seconds=20.0, data_dir=data, exclusive=True)
    first, second = mix.datasets
    # The earlier row subtracts from nothing yet; the later row loses the shared talk.
    assert first.exclusion == MixExclusion(pool_before=12, pool_after=12, clips_removed=0)
    assert second.exclusion == MixExclusion(pool_before=12, pool_after=8, clips_removed=4)
    # The pools the draws saw: the first whole, the second the remainder.
    assert first.drawn.available == 12
    assert second.drawn.available == 8
    classes = duplicate_classes(peertube_talks(), domain="peertube")
    shared_class = classes[shared]
    assert all(
        classes.get(clip.group.partition(":")[2]) != shared_class for clip in second.drawn.clips
    )
    # The earlier row drew what it draws alone, and the mix holds no clip twice.
    solo = select_clips(
        read_manifest(data / "peertube-fr-train-40k"),
        count=6,
        seed=mix_seed(7, "peertube-fr-train-40k"),
        max_seconds=20.0,
    )
    assert [clip.id for clip in first.drawn.clips] == [clip.id for clip in solo.clips]
    assert len(mix.clips) == 12
    assert mix.ids_sha256 == manifest_id_sha256(mix.clips)


def test_without_exclusive_the_same_mix_is_not_refused_for_other_bytes(tmp_path, shared_talks):
    """The gap the option fills: same speech, other prepared bytes.

    The inter-dataset check is on the audio, and the two rows hold other bytes of
    the same talk, so nothing refuses the mix without the flag: it would train
    the talk twice. exclusive is what subtracts it.
    """
    shared, own_a, own_b, _ = shared_talks
    data = tmp_path / "data"
    write_talk_dataset(data, "peertube-fr-train-40k", shared, own_a, n_shared=2, n_own=10)
    write_talk_dataset(data, "peertube-edu-fr-train-40k", shared, own_b, n_shared=4, n_own=8)
    mix = select_mix(
        (
            DatasetShare("peertube-fr-train-40k", 6),
            DatasetShare("peertube-edu-fr-train-40k", 6),
        ),
        seed=7,
        max_seconds=20.0,
        data_dir=data,
    )
    assert len(mix.clips) == 12
    for per in mix.datasets:
        assert per.exclusion is None


def test_swapping_the_rows_moves_the_loss_to_the_later_row(tmp_path, shared_talks):
    shared, own_a, own_b, _ = shared_talks
    data = tmp_path / "data"
    write_talk_dataset(data, "peertube-fr-train-40k", shared, own_a, n_shared=2, n_own=10)
    write_talk_dataset(data, "peertube-edu-fr-train-40k", shared, own_b, n_shared=4, n_own=8)
    first = select_mix(
        (
            DatasetShare("peertube-fr-train-40k", 6),
            DatasetShare("peertube-edu-fr-train-40k", 6),
        ),
        seed=7,
        max_seconds=20.0,
        data_dir=data,
        exclusive=True,
    )
    swapped = select_mix(
        (
            DatasetShare("peertube-edu-fr-train-40k", 6),
            DatasetShare("peertube-fr-train-40k", 6),
        ),
        seed=7,
        max_seconds=20.0,
        data_dir=data,
        exclusive=True,
    )
    by_name = {per.name: per for per in first.datasets}
    assert by_name["peertube-fr-train-40k"].exclusion.clips_removed == 0
    assert by_name["peertube-edu-fr-train-40k"].exclusion.clips_removed == 4
    by_name = {per.name: per for per in swapped.datasets}
    assert by_name["peertube-edu-fr-train-40k"].exclusion.clips_removed == 0
    assert by_name["peertube-fr-train-40k"].exclusion.clips_removed == 2


def test_without_exclusive_shared_audio_is_refused_and_exclusive_subtracts_it(tmp_path):
    """The same recording prepared into two rows: refused as it was, subtracted with the flag.

    A dataset without a corpus of talks subtracts by the audio sha256 it holds,
    and the subtraction reads the whole datasets of the earlier rows.
    """
    data = tmp_path / "data"
    shared = [hashlib.sha256(f"recording {i}".encode()).hexdigest() for i in range(3)]
    for name in ("ds-a", "ds-b"):
        write_manifest(
            data / name,
            [make_clip(f"{name}:{i:04d}", sha256=shared[i]) for i in range(3)]
            + [make_clip(f"{name}:{i:04d}") for i in range(3, 8)],
        )
    with pytest.raises(TrainDataError, match="3 recordings appear in more than one dataset"):
        select_mix(
            (DatasetShare("ds-a", 8), DatasetShare("ds-b", 8)),
            seed=7,
            max_seconds=20.0,
            data_dir=data,
        )
    mix = select_mix(
        (DatasetShare("ds-a", 5), DatasetShare("ds-b", 5)),
        seed=7,
        max_seconds=20.0,
        data_dir=data,
        exclusive=True,
    )
    first, second = mix.datasets
    assert first.exclusion == MixExclusion(pool_before=8, pool_after=8, clips_removed=0)
    assert second.exclusion == MixExclusion(pool_before=8, pool_after=5, clips_removed=3)
    assert second.drawn.available == 5
    shas = [clip.sha256 for clip in mix.clips]
    assert len(set(shas)) == len(shas)


def test_a_talk_row_subtracts_its_audio_from_a_later_row_without_a_corpus(tmp_path, shared_talks):
    """Every row's audio is what a later row without a corpus subtracts by.

    The first row holds a corpus of talks, so its clips subtract by class; the
    second row has none, so its clip with the first row's bytes is dropped by
    the audio sha256 it holds in its whole dataset, not only in its draw.
    """
    _, own_a, _, _ = shared_talks
    data = tmp_path / "data"
    audio = hashlib.sha256(b"shared recording").hexdigest()
    write_manifest(
        data / "peertube-fr-train-40k",
        [
            make_clip(
                "peertube-fr-train-40k:0000",
                group="peertube-fr-train-40k:shared-talk",
                sha256=audio,
            )
        ]
        + [
            make_clip(
                f"peertube-fr-train-40k:{index:04d}",
                group=f"peertube-fr-train-40k:{own_a}",
            )
            for index in range(1, 7)
        ],
    )
    write_manifest(
        data / "ds-b",
        [make_clip("ds-b:0000", sha256=audio)]
        + [make_clip(f"ds-b:{index:04d}") for index in range(1, 7)],
    )
    mix = select_mix(
        (DatasetShare("peertube-fr-train-40k", 4), DatasetShare("ds-b", 4)),
        seed=7,
        max_seconds=20.0,
        data_dir=data,
        exclusive=True,
    )
    first, second = mix.datasets
    assert first.exclusion == MixExclusion(pool_before=7, pool_after=7, clips_removed=0)
    assert second.exclusion == MixExclusion(pool_before=7, pool_after=6, clips_removed=1)
    assert second.drawn.available == 6
    assert all(clip.sha256 != audio for clip in second.drawn.clips)


def test_exclusive_is_deterministic_for_the_same_recipe_and_manifests(tmp_path, shared_talks):
    shared, own_a, own_b, _ = shared_talks
    data = tmp_path / "data"
    write_talk_dataset(data, "peertube-fr-train-40k", shared, own_a, n_shared=2, n_own=10)
    write_talk_dataset(data, "peertube-edu-fr-train-40k", shared, own_b, n_shared=4, n_own=8)
    shares = (
        DatasetShare("peertube-fr-train-40k", 6),
        DatasetShare("peertube-edu-fr-train-40k", 6),
    )
    first = select_mix(shares, seed=7, max_seconds=20.0, data_dir=data, exclusive=True)
    second = select_mix(shares, seed=7, max_seconds=20.0, data_dir=data, exclusive=True)
    assert [clip.id for clip in first.clips] == [clip.id for clip in second.clips]
    assert first.ids_sha256 == second.ids_sha256
    assert [per.exclusion for per in first.datasets] == [per.exclusion for per in second.datasets]


def test_exclusive_applies_to_a_mix_only(tmp_path):
    with pytest.raises(TrainConfigError, match="exclusive is a mix option"):
        TrainConfig.from_mapping(
            {**MINIMAL, "base_model": str(base_model(tmp_path)), "exclusive": True},
            name="example",
        )


def test_exclusive_must_be_a_boolean(tmp_path):
    with pytest.raises(TrainConfigError, match="exclusive must be true or false"):
        mix_config(tmp_path, exclusive="yes")


def test_a_mix_with_exclusive_false_is_the_mix_it_did_not_name(tmp_path):
    config = mix_config(tmp_path, exclusive=False)
    assert config.exclusive is False
    assert "exclusive" not in config.to_dict()


def test_an_exclusive_mix_run_records_and_logs_the_subtraction_per_row(
    tmp_path, monkeypatch, shared_talks
):
    """One line per row in the start log, one block per row in train.json, and the
    recipe's flag and the draw's hashes in config.json."""
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    shared, own_a, own_b, _ = shared_talks
    data = tmp_path / "data"
    write_talk_dataset(data, "peertube-fr-train-40k", shared, own_a, n_shared=2, n_own=10)
    write_talk_dataset(data, "peertube-edu-fr-train-40k", shared, own_b, n_shared=4, n_own=8)
    config = mix_config(
        tmp_path,
        datasets=[
            {"name": "peertube-fr-train-40k", "clips": 6},
            {"name": "peertube-edu-fr-train-40k", "clips": 6},
        ],
        exclusive=True,
    )
    lines: list[str] = []
    directory = runner.run_training(
        config, data_dir=data, runs_root=tmp_path / "runs", print_line=lines.append
    )
    mix = step.calls[0].clips
    # The start log: the clips line each row always gets, and one subtraction line.
    assert (
        "6 clips of peertube-fr-train-40k: 12 were usable "
        "(0 outside the duration window, 0 without text)" in lines
    )
    assert (
        "exclusive mix: peertube-fr-train-40k: 0 clips removed as shared with the "
        "earlier rows: pool 12 -> 12 usable clips" in lines
    )
    assert (
        "exclusive mix: peertube-edu-fr-train-40k: 4 clips removed as shared with the "
        "earlier rows: pool 12 -> 8 usable clips" in lines
    )
    # train.json: the per-row block, the rest of the record as always.
    record = json.loads((directory / TRAIN_JSON).read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in record["datasets"]}
    assert by_name["peertube-fr-train-40k"] == {
        "name": "peertube-fr-train-40k",
        "clips": 6,
        "clips_available": 12,
        "sampled_ids_sha256": mix.datasets[0].drawn.ids_sha256,
        "subtraction": {"pool_before": 12, "pool_after": 12, "clips_removed": 0},
    }
    assert by_name["peertube-edu-fr-train-40k"]["subtraction"] == {
        "pool_before": 12,
        "pool_after": 8,
        "clips_removed": 4,
    }
    assert record["clips"] == 12
    assert record["sampled_ids_sha256"] == mix.ids_sha256
    # config.json: the recipe says exclusive, and the per-dataset hashes are what
    # a resume is checked against.
    started = read_run_config(directory)
    assert started["config"]["exclusive"] is True
    assert started["datasets"] == [
        {"name": per.name, "sampled_ids_sha256": per.drawn.ids_sha256} for per in mix.datasets
    ]
    assert started["sampled_ids_sha256"] == record["sampled_ids_sha256"]


def test_the_subtraction_enters_the_resume_check(tmp_path, shared_talks):
    """Same recipe + same manifests resumes; a different subtraction is a different run.

    The resume check compares the run's ``config.json`` - the recipe and the
    per-dataset hashes of the draw - against what the same command recomputes.
    The subtraction is derived from the recipe and the manifests, so it enters
    the comparison in the recipe's flag and in the draw the flag changes.
    """
    shared, own_a, own_b, own_c = shared_talks
    data = tmp_path / "data"
    write_talk_dataset(data, "peertube-fr-train-40k", shared, own_a, n_shared=2, n_own=10)
    # The later row holds own_c in one clip only: a new clip of that talk in
    # the earlier row removes exactly that one clip from its pool.
    edu = [
        make_clip(f"peertube-edu-fr-train-40k:{i:04d}", group=f"peertube-edu-fr-train-40k:{shared}")
        for i in range(4)
    ]
    edu.append(
        make_clip("peertube-edu-fr-train-40k:0004", group=f"peertube-edu-fr-train-40k:{own_c}")
    )
    edu += [
        make_clip(
            f"peertube-edu-fr-train-40k:{5 + i:04d}", group=f"peertube-edu-fr-train-40k:{own_b}"
        )
        for i in range(7)
    ]
    write_manifest(data / "peertube-edu-fr-train-40k", edu)
    datasets = [
        {"name": "peertube-fr-train-40k", "clips": 6},
        {"name": "peertube-edu-fr-train-40k", "clips": 6},
    ]
    config = mix_config(tmp_path, datasets=datasets, exclusive=True)
    plain = mix_config(tmp_path, datasets=datasets)
    checkpoint = config.weights()
    identity = checkpoint.identity()
    drawn = select_mix(
        config.datasets,
        seed=config.seed,
        max_seconds=config.max_clip_seconds,
        data_dir=data,
        exclusive=True,
    )
    started = runner._start_payload(config, checkpoint, identity, drawn, {})
    # The same recipe over the same manifests recomputes the same start: a resume.
    redrawn = select_mix(
        config.datasets,
        seed=config.seed,
        max_seconds=config.max_clip_seconds,
        data_dir=data,
        exclusive=True,
    )
    restarted = runner._start_payload(config, checkpoint, identity, redrawn, {})
    assert config_differences(started, restarted) == []
    # The flag off draws the later row whole: the check sees it in the recipe.
    other_drawn = select_mix(
        plain.datasets, seed=plain.seed, max_seconds=plain.max_clip_seconds, data_dir=data
    )
    other_started = runner._start_payload(plain, checkpoint, identity, other_drawn, {})
    assert "config.exclusive" in config_differences(started, other_started)
    # The record a run started before the option wrote: no exclusive key in the
    # recipe, the same top-level keys, so it recomputes against the plain one.
    assert "exclusive" not in other_started["config"]
    assert set(other_started) == {
        "config",
        "base_model",
        "base_model_identity",
        "sampled_ids_sha256",
        "clips",
        "datasets",
    }
    # A new clip in the earlier row, holding a talk the later row's pool holds:
    # the later row's pool shrinks by that clip, its draw changes, and its
    # hash - what the resume check compares - changes with it.
    upstream = read_manifest(data / "peertube-fr-train-40k")
    write_manifest(
        data / "peertube-fr-train-40k",
        upstream
        + [make_clip("peertube-fr-train-40k:0012", group=f"peertube-fr-train-40k:{own_c}")],
    )
    changed = select_mix(
        config.datasets,
        seed=config.seed,
        max_seconds=config.max_clip_seconds,
        data_dir=data,
        exclusive=True,
    )
    assert changed.datasets[1].exclusion == MixExclusion(
        pool_before=12, pool_after=7, clips_removed=5
    )
    changed_started = runner._start_payload(config, checkpoint, identity, changed, {})
    assert "datasets" in config_differences(started, changed_started)
    assert "sampled_ids_sha256" in config_differences(started, changed_started)


def test_resuming_an_exclusive_run_refuses_a_changed_subtraction(
    tmp_path, monkeypatch, shared_talks
):
    """A stopped exclusive run resumes with the same recipe, and nothing else.

    The subtraction is what the resume check compares, so a resume that would
    subtract different clips - the flag off, the rows reordered - is a
    different run, and the refusal says which key moved.
    """
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    shared, own_a, own_b, _ = shared_talks
    data = tmp_path / "data"
    write_talk_dataset(data, "peertube-fr-train-40k", shared, own_a, n_shared=2, n_own=10)
    write_talk_dataset(data, "peertube-edu-fr-train-40k", shared, own_b, n_shared=4, n_own=8)
    datasets = [
        {"name": "peertube-fr-train-40k", "clips": 6},
        {"name": "peertube-edu-fr-train-40k", "clips": 6},
    ]
    config = mix_config(tmp_path, datasets=datasets, exclusive=True)
    plain = mix_config(tmp_path, datasets=datasets)
    swapped = mix_config(tmp_path, datasets=list(reversed(datasets)), exclusive=True)
    drawn = select_mix(
        config.datasets,
        seed=config.seed,
        max_seconds=config.max_clip_seconds,
        data_dir=data,
        exclusive=True,
    )
    started = runner._start_payload(
        config, config.weights(), config.weights().identity(), drawn, {}
    )

    def stopped_run(runs_root: Path) -> Path:
        """The stopped exclusive run: its start written, its slot, no finish."""
        directory = prepare_run_dir(run_dir(config.name, runs_root))
        write_run_config(directory, started)
        (directory / RESUME_NAME).write_bytes(b"optimizer state")
        return directory

    # The same recipe over the same manifests resumes it.
    directory = stopped_run(tmp_path / "runs" / "same")
    directory = runner.run_training(
        config, data_dir=data, runs_root=tmp_path / "runs" / "same", print_line=lambda _: None
    )
    assert is_finished(directory)
    assert step.calls[-1].resuming is True
    # The flag off would draw the later row whole: refused, and named.
    stopped_run(tmp_path / "runs" / "plain")
    with pytest.raises(TrainRunError, match="config.exclusive"):
        runner.run_training(
            plain, data_dir=data, runs_root=tmp_path / "runs" / "plain", print_line=lambda _: None
        )
    # The rows reordered subtract the other way: refused, and named.
    stopped_run(tmp_path / "runs" / "swapped")
    with pytest.raises(TrainRunError, match="config.datasets"):
        runner.run_training(
            swapped,
            data_dir=data,
            runs_root=tmp_path / "runs" / "swapped",
            print_line=lambda _: None,
        )
