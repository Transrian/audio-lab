"""The leakage guard's content rules, on CPU: the duplicate classes and the babble.

The two older rules - the audio sha256 and the dataset name - live in
``test_train_mix.py``'s guard section and pass unchanged. This file covers the
rules on the *content* of a recording: a training clip of a talk or NASA source
is refused when its talk or episode is in the duplicate class of a talk or
episode of a prepared evaluation set (the re-upload or re-release, whose
prepared audio is other bytes), and a run that is not a resume is refused when
its clips hold a babble utterance of the noisy evaluation sets.
"""

from __future__ import annotations

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
)

from audiolab.data import nasa  # noqa: E402
from audiolab.data.duplicates import duplicate_classes, peertube_talks  # noqa: E402
from audiolab.data.noisy import babble_union, load_babble_utterances  # noqa: E402
from audiolab.data.store import write_manifest  # noqa: E402
from audiolab.data.talk_listing import read_listing  # noqa: E402
from audiolab.train import runner  # noqa: E402
from audiolab.train.config import TrainConfig  # noqa: E402
from audiolab.train.data import TrainDataError, load_training_clips  # noqa: E402
from audiolab.train.guard import check_leakage  # noqa: E402
from audiolab.train.run import (  # noqa: E402
    RESUME_NAME,
    is_finished,
    prepare_run_dir,
    run_dir,
    write_run_config,
)

EVALUATIONS = json.loads(
    (
        Path(__file__).resolve().parent.parent / "src" / "audiolab" / "data" / "duplicates.json"
    ).read_text(encoding="utf-8")
)
EVAL_DATASET = "peertube-edu-captions-fr"


def _confirmed_pair(domain: str, reason_holds: str) -> tuple[str, str]:
    """One committed confirmed pair: (the evaluation talk, the copy a training set may hold)."""
    entry = next(
        entry
        for entry in EVALUATIONS
        if entry["domain"] == domain and reason_holds in entry["reason"]
    )
    return entry["ids"][0], entry["ids"][1]


EVAL_TALK, TWIN_TALK = _confirmed_pair("peertube", "evaluation talk")
EVAL_EPISODE, TWIN_EPISODE = _confirmed_pair("nasa", "re-released episode")


@pytest.fixture(scope="module")
def peertube_corpus() -> tuple[dict[str, str], str]:
    """The duplicate classes of the PeerTube corpus, and a talk of another class."""
    talks = peertube_talks()
    ids = {record["id"] for record in talks}
    assert EVAL_TALK in ids and TWIN_TALK in ids, "the pair left the pinned listings"
    classes = duplicate_classes(talks, domain="peertube")
    safe = next(
        record["id"]
        for record in talks
        if record["id"] not in (EVAL_TALK, TWIN_TALK)
        and classes[record["id"]] != classes[EVAL_TALK]
    )
    return classes, safe


@pytest.fixture(autouse=True)
def no_gpu_needed(monkeypatch):
    """The training step is faked, so no GPU is either: CI machines have none."""
    monkeypatch.setattr(runner, "select_gpu", lambda spec: None)


def _suite(tmp_path: Path, names: list[str]):
    """A suite the tests own, so the guard checks what the tests prepare."""
    return guard_suite(tmp_path, names)


def _babble_pin(tmp_path: Path, utterance: str) -> Path:
    """A committed babble file shaped like the real one, holding ``utterance``."""
    path = tmp_path / "babble.json"
    path.write_text(
        json.dumps(
            {
                "note": "the committed babble of the prepared noisy sets, for a test",
                "noisy-fleurs-fr": [utterance],
                "noisy-fleurs-en": ["1_1_000001"],
                "noisy-servicenow-codeswitched-en-fr": ["2_2_000002"],
            }
        ),
        encoding="utf-8",
    )
    return path


def _base_model(tmp_path: Path) -> Path:
    base = tmp_path / "base.nemo"
    base.write_bytes(b"base weights")
    return base


def _write_babble_pool(data: Path, utterance: str) -> None:
    """The babble pool as a prepared dataset: exactly the four clips a run draws."""
    write_manifest(
        data / "mls-fr-train",
        [
            make_clip(f"mls-fr-train:{utterance}"),
            make_clip("mls-fr-train:99999_99999_000001"),
            make_clip("mls-fr-train:99999_99999_000002"),
            make_clip("mls-fr-train:99999_99999_000003"),
        ],
    )


# --- the duplicate-class rule ---------------------------------------------------


def test_a_training_clip_of_a_twin_talk_of_an_evaluation_talk_is_refused(tmp_path, peertube_corpus):
    """The leak the class rule exists for: same speech, other prepared bytes."""
    data = tmp_path / "data"
    write_manifest(
        data / EVAL_DATASET,
        [make_clip(f"{EVAL_DATASET}:0001", group=f"{EVAL_DATASET}:{EVAL_TALK}")],
    )
    suite, panels = _suite(tmp_path, [EVAL_DATASET])
    training = [
        make_clip(
            "peertube-tech-fr-train-40k:00000001",
            group=f"peertube-tech-fr-train-40k:{TWIN_TALK}",
        )
    ]
    with pytest.raises(
        TrainDataError,
        match=(
            f"training clip peertube-tech-fr-train-40k:00000001 is a duplicate of the "
            f"evaluation talk {EVAL_TALK} of {EVAL_DATASET}"
        ),
    ):
        check_leakage(
            training,
            data_dir=data,
            suite_path=suite,
            panels_dir=panels,
            print_line=lambda _: None,
        )


def test_a_training_clip_of_another_talk_of_the_corpus_passes(tmp_path, peertube_corpus):
    """Same corpus, another class: the guard checks the talk, not the source."""
    _, safe_talk = peertube_corpus
    data = tmp_path / "data"
    write_manifest(
        data / EVAL_DATASET,
        [make_clip(f"{EVAL_DATASET}:0001", group=f"{EVAL_DATASET}:{EVAL_TALK}")],
    )
    suite, panels = _suite(tmp_path, [EVAL_DATASET])
    training = [
        make_clip("peertube-edu-fr-train:00000001", group=f"peertube-edu-fr-train:{safe_talk}")
    ]
    result = check_leakage(
        training,
        data_dir=data,
        suite_path=suite,
        panels_dir=panels,
        print_line=lambda _: None,
    )
    assert result.clips_checked == 1
    assert EVAL_DATASET in result.evaluation_datasets_checked


def test_a_training_clip_of_a_re_release_of_an_evaluation_episode_is_refused(
    tmp_path,
):
    """The NASA form of the same rule: the re-release, under another slug."""
    listed, episodes = read_listing(nasa.SOURCE)
    slugs = {episode.id for episode in episodes}
    assert EVAL_EPISODE in slugs and TWIN_EPISODE in slugs
    data = tmp_path / "data"
    write_manifest(
        data / "nasa-en-test",
        [make_clip("nasa-en-test:0001", group=f"nasa-en-test:{EVAL_EPISODE}")],
    )
    suite, panels = _suite(tmp_path, ["nasa-en-test"])
    training = [make_clip("nasa-en-train:0001", group=f"nasa-en-train:{TWIN_EPISODE}")]
    with pytest.raises(
        TrainDataError,
        match=(
            f"training clip nasa-en-train:0001 is a duplicate of the "
            f"evaluation talk {EVAL_EPISODE} of nasa-en-test"
        ),
    ):
        check_leakage(
            training,
            data_dir=data,
            suite_path=suite,
            panels_dir=panels,
            print_line=lambda _: None,
        )


def test_a_clip_of_a_relabelled_copy_of_the_source_is_checked_like_the_source(
    tmp_path,
):
    """The derived set: the same clips under another name still take the source's classes."""
    listed, episodes = read_listing(nasa.SOURCE)
    slugs = {episode.id for episode in episodes}
    assert EVAL_EPISODE in slugs and TWIN_EPISODE in slugs
    data = tmp_path / "data"
    write_manifest(
        data / "nasa-en-test",
        [make_clip("nasa-en-test:0001", group=f"nasa-en-test:{EVAL_EPISODE}")],
    )
    suite, panels = _suite(tmp_path, ["nasa-en-test"])
    # The twin, under the derived set's name: refused the way the source's is.
    training = [make_clip("nasa-en-train-qwen:0001", group=f"nasa-en-train-qwen:{TWIN_EPISODE}")]
    with pytest.raises(
        TrainDataError,
        match=(
            f"training clip nasa-en-train-qwen:0001 is a duplicate of the "
            f"evaluation talk {EVAL_EPISODE} of nasa-en-test"
        ),
    ):
        check_leakage(
            training,
            data_dir=data,
            suite_path=suite,
            panels_dir=panels,
            print_line=lambda _: None,
        )
    # An episode of another class, under the derived set's name: passed.
    classes = duplicate_classes(episodes, domain="nasa")
    safe = next(
        episode.id
        for episode in episodes
        if episode.id not in (EVAL_EPISODE, TWIN_EPISODE)
        and classes[episode.id] != classes[EVAL_EPISODE]
    )
    result = check_leakage(
        [make_clip("nasa-en-train-qwen:0002", group=f"nasa-en-train-qwen:{safe}")],
        data_dir=data,
        suite_path=suite,
        panels_dir=panels,
        print_line=lambda _: None,
    )
    assert result.clips_checked == 1


# --- the babble rule ------------------------------------------------------------


def test_a_new_run_holding_a_babble_utterance_is_refused_before_anything_is_written(
    tmp_path, monkeypatch
):
    """A single dataset whose drawn clips hold the pool's babble: refused, nothing written."""
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    data = tmp_path / "data"
    utterance = "10032_9061_000013"
    _write_babble_pool(data, utterance)
    config = TrainConfig.from_mapping(
        {
            **MINIMAL,
            "base_model": str(_base_model(tmp_path)),
            "dataset": "mls-fr-train",
            "clips": 4,
        },
        name="example",
    )
    suite, panels = _suite(tmp_path, ["guard-eval"])
    pin = _babble_pin(tmp_path, utterance)
    monkeypatch.setattr(
        runner,
        "check_leakage",
        partial(check_leakage, suite_path=suite, panels_dir=panels, babble_path=pin),
    )
    with pytest.raises(TrainDataError, match="babble utterance of the noisy evaluation sets"):
        runner.run_training(config, data_dir=data, runs_root=tmp_path / "runs")
    assert step.calls == []
    assert not run_dir("example", tmp_path / "runs").exists()


def test_a_mix_run_holding_a_babble_utterance_is_refused_before_anything_is_written(
    tmp_path, monkeypatch
):
    """The mix form: one row of the mix holds the babble, the run is refused."""
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    data = tmp_path / "data"
    utterance = "10032_9061_000013"
    _write_babble_pool(data, utterance)
    write_manifest(data / "ds-b", [make_clip(f"ds-b:{index:04d}") for index in range(4)])
    config = TrainConfig.from_mapping(
        {
            **MIX_MINIMAL,
            "base_model": str(_base_model(tmp_path)),
            "datasets": [
                {"name": "mls-fr-train", "clips": 4},
                {"name": "ds-b", "clips": 4},
            ],
        },
        name="mixexample",
    )
    suite, panels = _suite(tmp_path, ["guard-eval"])
    pin = _babble_pin(tmp_path, utterance)
    monkeypatch.setattr(
        runner,
        "check_leakage",
        partial(check_leakage, suite_path=suite, panels_dir=panels, babble_path=pin),
    )
    with pytest.raises(TrainDataError, match="babble utterance of the noisy evaluation sets"):
        runner.run_training(config, data_dir=data, runs_root=tmp_path / "runs")
    assert step.calls == []
    assert not run_dir("mixexample", tmp_path / "runs").exists()


def test_a_new_run_without_a_babble_utterance_passes_the_guard(tmp_path, monkeypatch):
    """The pool's clips with the babble one out: the run goes ahead and finishes."""
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    data = tmp_path / "data"
    write_manifest(
        data / "mls-fr-train",
        [make_clip(f"mls-fr-train:99999_99999_00000{index}") for index in range(1, 5)],
    )
    config = TrainConfig.from_mapping(
        {
            **MINIMAL,
            "base_model": str(_base_model(tmp_path)),
            "dataset": "mls-fr-train",
            "clips": 4,
        },
        name="example",
    )
    suite, panels = _suite(tmp_path, ["guard-eval"])
    pin = _babble_pin(tmp_path, "10032_9061_000013")
    monkeypatch.setattr(
        runner,
        "check_leakage",
        partial(check_leakage, suite_path=suite, panels_dir=panels, babble_path=pin),
    )
    directory = runner.run_training(config, data_dir=data, runs_root=tmp_path / "runs")
    assert len(step.calls) == 1
    assert is_finished(directory)


def test_a_resume_of_a_run_that_started_before_the_rule_is_not_refused_for_babble(
    tmp_path, monkeypatch
):
    """A new-run rule: the old run - with its babble - still resumes.

    The directory holds what an audiolab that knew no babble rule wrote: the
    config as started and the resume slot. The same clips, checked on the
    resume, must not be refused for the babble, and the run must finish.
    """
    step = FakeStep()
    monkeypatch.setattr(runner, "train_checkpoint", step)
    data = tmp_path / "data"
    utterance = "10032_9061_000013"
    _write_babble_pool(data, utterance)
    config = TrainConfig.from_mapping(
        {
            **MINIMAL,
            "base_model": str(_base_model(tmp_path)),
            "dataset": "mls-fr-train",
            "clips": 4,
        },
        name="example",
    )
    suite, panels = _suite(tmp_path, ["guard-eval"])
    pin = _babble_pin(tmp_path, utterance)
    monkeypatch.setattr(
        runner,
        "check_leakage",
        partial(check_leakage, suite_path=suite, panels_dir=panels, babble_path=pin),
    )
    # The same clips, the same refusal, on a start that is not a resume.
    with pytest.raises(TrainDataError, match="babble utterance"):
        runner.run_training(config, data_dir=data, runs_root=tmp_path / "runs")
    # The start an audiolab before the rule wrote, stopped with its slot.
    drawn = load_training_clips(
        runner.dataset_dir(config.dataset, data),
        count=config.clips,
        seed=config.seed,
        max_seconds=config.max_clip_seconds,
    )
    directory = prepare_run_dir(run_dir(config.name, tmp_path / "runs"))
    write_run_config(
        directory,
        runner._start_payload(config, config.weights(), config.weights().identity(), drawn, {}),
    )
    (directory / RESUME_NAME).write_bytes(b"optimizer state")
    # The same command again: the resume is checked for the other things alone.
    directory = runner.run_training(config, data_dir=data, runs_root=tmp_path / "runs")
    assert len(step.calls) == 1
    assert step.calls[0].resuming is True
    assert is_finished(directory)


def test_the_babble_rule_refuses_a_new_run_but_not_the_same_clips_on_a_resume(tmp_path):
    """The rule at the level of the check itself: new run refused, resume passed."""
    data = tmp_path / "data"
    suite, panels = _suite(tmp_path, ["guard-eval"])
    pin = _babble_pin(tmp_path, "10032_9061_000013")
    clip = make_clip("mls-fr-train:10032_9061_000013")
    with pytest.raises(TrainDataError, match="babble utterance"):
        check_leakage(
            [clip],
            data_dir=data,
            suite_path=suite,
            panels_dir=panels,
            babble_path=pin,
            print_line=lambda _: None,
        )
    result = check_leakage(
        [clip],
        data_dir=data,
        suite_path=suite,
        panels_dir=panels,
        babble_path=pin,
        resuming=True,
        print_line=lambda _: None,
    )
    assert result.clips_checked == 1


def test_a_run_checked_against_the_committed_babble_passes_without_one(tmp_path):
    """The committed babble file, loaded by default: its ids refuse, the rest pass."""
    data = tmp_path / "data"
    suite, panels = _suite(tmp_path, ["guard-eval"])
    committed = load_babble_utterances()
    babble_union_ids = set(babble_union(committed))
    # The first id of the first list of the committed file: the default path -
    # no babble_path given - refuses a run that holds it, so the file is the
    # one really read.
    babble_id = next(value for key, value in committed.items() if key != "note")[0]
    assert babble_id in babble_union_ids
    with pytest.raises(TrainDataError, match="babble utterance"):
        check_leakage(
            [make_clip(f"mls-fr-train:{babble_id}")],
            data_dir=data,
            suite_path=suite,
            panels_dir=panels,
            print_line=lambda _: None,
        )
    # A pool clip that is not in the committed lists passes the same default path.
    assert "99999_99999_000001" not in babble_union_ids
    result = check_leakage(
        [make_clip("mls-fr-train:99999_99999_000001")],
        data_dir=data,
        suite_path=suite,
        panels_dir=panels,
        print_line=lambda _: None,
    )
    assert result.clips_checked == 1
