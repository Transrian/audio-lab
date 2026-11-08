"""One training run, from a config file to a checkpoint and its record.

This module is the sequence, not the machinery: where the run lives, which clips
it trains on, the manifest NeMo reads, the call that trains, and what is written
down afterwards. It imports nothing heavy, so the whole sequence can be read and
tested with the plain install; the single call that needs PyTorch and NeMo goes
through :func:`train_checkpoint`, which the tests replace.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from audiolab.core.checkpoint import Checkpoint
from audiolab.core.gpu import select_gpu
from audiolab.eval.records import git_commit
from audiolab.train.config import TrainConfig
from audiolab.train.data import (
    TrainingClips,
    dataset_dir,
    load_training_clips,
    write_nemo_manifest,
)

if TYPE_CHECKING:  # what the heavy step returns; importing it would need PyTorch
    from audiolab.train.finetune import TrainResult

from audiolab.train.run import (
    LOG_NAME,
    MANIFEST_NAME,
    RESUME_NAME,
    TrainRunError,
    config_differences,
    prepare_run_dir,
    read_run_config,
    refuse_finished_run,
    resume_checkpoint,
    run_dir,
    write_model_config,
    write_run_config,
    write_train_json,
)

# What to say when the environment has no PyTorch and no NeMo. That is normal,
# and both ways out are worth spelling out.
MISSING_HEAVY_STACK = (
    "training needs PyTorch and NeMo, which the plain install does not carry. Build "
    "the local model environment with 'bash scripts/nemo-venv.sh' and run 'audiolab "
    "train' from it, or from any other interpreter where NeMo is installed "
    "(see requirements/nemo.txt)"
)


@dataclass(frozen=True)
class Training:
    """A run that is about to start: what it trains on, and where it writes."""

    config: TrainConfig
    directory: Path
    clips: TrainingClips
    base: Checkpoint
    base_identity: str
    resuming: bool


def train_checkpoint(
    training: Training, *, print_line: Callable[[str], None] = print
) -> TrainResult:
    """The one heavy step. Kept behind a name so the tests can replace it."""
    try:
        from audiolab.train import finetune
    except ImportError as failure:
        raise TrainRunError(MISSING_HEAVY_STACK) from failure
    return finetune.fine_tune(
        training.config,
        training.base,
        training.directory / MANIFEST_NAME,
        training.directory,
        training.directory / LOG_NAME,
        print_line,
    )


def run_training(
    config: TrainConfig,
    *,
    data_dir: Path | None = None,
    runs_root: Path | None = None,
    fresh: bool = False,
    print_line: Callable[[str], None] = print,
) -> Path:
    """Train one config and return the run directory that now holds the result."""
    directory = run_dir(config.name, runs_root)
    # Before anything is read, so a finished run costs a second rather than a
    # checkpoint load and a dataset scan.
    refuse_finished_run(directory, fresh=fresh)
    # The base checkpoint is resolved here rather than inside the heavy step, so
    # a missing file or an unset variable is reported before a GPU is touched.
    base = config.weights()
    base_identity = base.identity()
    with_data = dataset_dir(config.dataset, data_dir)
    clips = load_training_clips(
        with_data,
        count=config.clips,
        seed=config.seed,
        max_seconds=config.max_clip_seconds,
    )
    print_line(
        f"{len(clips.clips)} clips of {config.dataset}: {clips.available} were usable "
        f"({clips.outside_duration} outside the duration window, "
        f"{clips.without_text} without text)"
    )
    started = _start_payload(config, base, base_identity, clips)
    if not fresh and resume_checkpoint(directory) is not None:
        _refuse_a_different_run(directory, started)
    prepare_run_dir(directory, fresh=fresh)
    # Written before the first update, and never rewritten: this is what a later
    # resume is checked against.
    write_run_config(directory, started)
    write_nemo_manifest(directory / MANIFEST_NAME, clips.clips, with_data)

    # The GPU is chosen before the heavy step imports torch, because
    # CUDA_VISIBLE_DEVICES has to be set before CUDA is initialised (D-008).
    select_gpu(config.gpu)
    training = Training(
        config=config,
        directory=directory,
        clips=clips,
        base=base,
        base_identity=base_identity,
        resuming=not fresh and resume_checkpoint(directory) is not None,
    )
    print_line(
        "resuming from the last resume slot"
        if training.resuming
        else f"starting from {base} ({base_identity})"
    )
    result = train_checkpoint(training, print_line=print_line)

    # The record is written last, and the resume slot removed after it: a run
    # directory with no train.json is a run that stopped, whatever else is in it.
    write_train_json(directory, _record(config, training, result))
    (directory / RESUME_NAME).unlink(missing_ok=True)
    write_model_config(directory, name=config.name, gpu=config.gpu)
    return directory


def _start_payload(
    config: TrainConfig, base: Checkpoint, base_identity: str, clips: TrainingClips
) -> dict:
    """What a run is, for ``config.json``: its config, its base model, its clips.

    The clips are in it because a dataset can be prepared again under the same
    name: the same seed over other clips is another run, and this is what tells
    the two apart.
    """
    return {
        "config": config.to_dict(),
        "base_model": str(base),
        "base_model_identity": base_identity,
        "sampled_ids_sha256": clips.ids_sha256,
        "clips": len(clips.clips),
    }


def _refuse_a_different_run(directory: Path, started: dict) -> None:
    """A resume slot belongs to one run, and this is where that is enforced."""
    recorded = read_run_config(directory)
    if recorded is None:
        raise TrainRunError(
            f"{directory} holds a resume slot but no config.json, so the run it belongs to "
            "cannot be checked; go on with --fresh if the slot is not worth keeping"
        )
    differences = config_differences(recorded, started)
    if differences:
        raise TrainRunError(
            f"{directory} was started as a different run: {', '.join(differences)} changed. "
            "Put the old value back to continue it, or start a new run with --fresh "
            "(which deletes this directory and its resume slot)"
        )


def _record(config: TrainConfig, training: Training, result: TrainResult) -> dict:
    """What this run was and what it produced, as it goes into train.json."""
    updates = result.updates or config.updates
    seconds = result.wall_seconds or 0.0
    return {
        "name": config.name,
        "config": config.to_dict(),
        "dataset": config.dataset,
        "clips": len(training.clips.clips),
        "clips_available": training.clips.available,
        "sampled_ids_sha256": training.clips.ids_sha256,
        "base_model": str(training.base),
        "base_model_identity": training.base_identity,
        "resumed": training.resuming,
        "git_commit": git_commit(),
        "final_loss": result.final_loss,
        "updates": updates,
        "wall_seconds": round(seconds, 1),
        "updates_per_second": round(updates / seconds, 2) if seconds > 0 else 0.0,
        "peak_vram_gib": round(result.peak_vram_gib, 2),
        "device": result.device,
    }
