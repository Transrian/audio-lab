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
from audiolab.train.run import (
    LOG_NAME,
    MANIFEST_NAME,
    MODEL_NAME,
    RESUME_NAME,
    TRAIN_JSON,
    TrainRunError,
    is_finished,
    prepare_run_dir,
    resume_checkpoint,
    run_dir,
    write_model_config,
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


def train_checkpoint(training: Training, *, print_line: Callable[[str], None] = print):
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
    if is_finished(directory) and not fresh:
        raise TrainRunError(
            f"{directory} already holds a finished run ({MODEL_NAME} and {TRAIN_JSON}); "
            "train again with --fresh to start over"
        )
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
    prepare_run_dir(directory, fresh=fresh)
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
        resuming=resume_checkpoint(directory) is not None,
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


def _record(config: TrainConfig, training: Training, result: object) -> dict:
    """What this run was and what it produced, as it goes into train.json."""
    updates = getattr(result, "updates", config.updates) or config.updates
    seconds = getattr(result, "wall_seconds", 0.0) or 0.0
    return {
        "name": config.name,
        "config": config.to_dict(),
        "dataset": config.dataset,
        "clips": len(training.clips.clips),
        "clips_available": training.clips.available,
        "sampled_ids_sha256": training.clips.ids_sha256,
        "base_model": str(training.base),
        "base_model_identity": training.base_identity,
        "git_commit": git_commit(),
        "final_loss": getattr(result, "final_loss", float("nan")),
        "updates": updates,
        "wall_seconds": round(seconds, 1),
        "updates_per_second": round(updates / seconds, 2) if seconds > 0 else 0.0,
        "peak_vram_gib": round(getattr(result, "peak_vram_gib", 0.0), 2),
        "device": getattr(result, "device", ""),
    }
