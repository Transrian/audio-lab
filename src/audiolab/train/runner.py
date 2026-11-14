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
from audiolab.core.progress import say
from audiolab.core.stop import Stopped, raise_if_stopped
from audiolab.data.store import Clip
from audiolab.eval.records import git_commit
from audiolab.train.config import TrainConfig
from audiolab.train.data import (
    MixClips,
    TrainingClips,
    dataset_dir,
    load_training_clips,
    select_mix,
    write_bank_manifest,
    write_nemo_manifest,
)
from audiolab.train.guard import GuardResult, check_leakage

if TYPE_CHECKING:  # what the heavy step returns; importing it would need PyTorch
    from audiolab.train.finetune import TrainResult

from audiolab.train.run import (
    LOG_NAME,
    MANIFEST_NAME,
    NOISE_MANIFEST_NAME,
    RESUME_NAME,
    REVERB_MANIFEST_NAME,
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
    clips: TrainingClips | MixClips
    base: Checkpoint
    base_identity: str
    resuming: bool
    # The run's augmentation, written down as the banks were read: one entry
    # per enabled perturbation, with its probabilities, SNR range, the bank it
    # came from, and the hash of the manifest written into the run directory.
    # ``None`` when the run does not augment: the record then says no more than
    # the run that augmented nothing says today.
    augmentation: dict | None = None


def train_checkpoint(training: Training, *, print_line: Callable[[str], None] = say) -> TrainResult:
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
    print_line: Callable[[str], None] = say,
) -> Path:
    """Train one config and return the run directory that now holds the result.

    A run asked to stop raises :class:`~audiolab.core.stop.Stopped` once its
    resume slot is written; it writes neither a checkpoint nor a record, so the
    directory says what it is: a run that has not finished.
    """
    directory = run_dir(config.name, runs_root)
    # Before anything is read, so a finished run costs a second rather than a
    # checkpoint load and a dataset scan.
    refuse_finished_run(directory, fresh=fresh)
    # The base checkpoint is resolved here rather than inside the heavy step, so
    # a missing file or an unset variable is reported before a GPU is touched.
    base = config.weights()
    base_identity = base.identity()
    if config.datasets:
        drawn = select_mix(
            config.datasets,
            seed=config.seed,
            max_seconds=config.max_clip_seconds,
            data_dir=data_dir,
        )
        for per in drawn.datasets:
            print_line(_clips_line(per.name, per.drawn))
    else:
        with_data = dataset_dir(config.dataset, data_dir)
        drawn = load_training_clips(
            with_data,
            count=config.clips,
            seed=config.seed,
            max_seconds=config.max_clip_seconds,
        )
        print_line(_clips_line(config.dataset, drawn))
    # Before the model is loaded and before anything is written: a run that
    # would train on what the suite scores is refused, and a refusal costs no
    # directory and no GPU time.
    guard = check_leakage(drawn.clips, data_dir=data_dir, print_line=print_line)
    started = _start_payload(config, base, base_identity, drawn)
    if not fresh and resume_checkpoint(directory) is not None:
        _refuse_a_different_run(directory, started)
    prepare_run_dir(directory, fresh=fresh)
    # Written before the first update, and never rewritten: this is what a later
    # resume is checked against.
    write_run_config(directory, started)
    write_nemo_manifest(directory / MANIFEST_NAME, _manifest_entries(config, drawn, data_dir))
    # A run that augments writes its banks' NeMo manifests into the run
    # directory, so the record can hash what the perturbations actually read.
    augmentation = _write_bank_manifests(config, directory, data_dir)

    # The GPU is chosen before the heavy step imports torch, because
    # CUDA_VISIBLE_DEVICES has to be set before CUDA is initialised (D-008).
    select_gpu(config.gpu)
    training = Training(
        config=config,
        directory=directory,
        clips=drawn,
        base=base,
        base_identity=base_identity,
        resuming=not fresh and resume_checkpoint(directory) is not None,
        augmentation=augmentation,
    )
    print_line(
        "resuming from the last resume slot"
        if training.resuming
        else f"starting from {base} ({base_identity})"
    )
    raise_if_stopped(f"training of {config.name} stopped before it started")
    result = train_checkpoint(training, print_line=print_line)
    if result.stopped:
        raise Stopped(
            f"training of {config.name} stopped after update {result.updates} of "
            f"{config.updates}, kept in {directory / RESUME_NAME}"
        )

    # The record is written last, and the resume slot removed after it: a run
    # directory with no train.json is a run that stopped, whatever else is in it.
    write_train_json(directory, _record(config, training, result, guard))
    (directory / RESUME_NAME).unlink(missing_ok=True)
    write_model_config(directory, name=config.name, gpu=config.gpu)
    return directory


def _clips_line(name: str, drawn: TrainingClips) -> str:
    """What one dataset contributed to a run, single or one of a mix's."""
    return (
        f"{len(drawn.clips)} clips of {name}: {drawn.available} were usable "
        f"({drawn.outside_duration} outside the duration window, "
        f"{drawn.without_text} without text)"
    )


def _write_bank_manifests(
    config: TrainConfig, directory: Path, data_dir: Path | None
) -> dict | None:
    """The NeMo manifests of a run's banks, written into its run directory.

    One file per enabled perturbation, built from the train half of its bank,
    and the block the run record keeps: what it did to the audio, from which
    bank, and the hash of the file the perturbations read. ``None`` when the
    run does not augment, and no file is written then: the run directory of a
    run that augmented nothing is the run directory of today.
    """
    if config.noise_prob <= 0.0 and config.reverb_prob <= 0.0:
        return None
    blocks: dict = {}
    if config.noise_prob > 0.0:
        lines, digest = write_bank_manifest(
            directory / NOISE_MANIFEST_NAME, dataset_dir(config.noise_bank, data_dir)
        )
        blocks["noise"] = {
            "prob": config.noise_prob,
            "min_snr_db": config.noise_min_snr_db,
            "max_snr_db": config.noise_max_snr_db,
            "bank": config.noise_bank,
            "bank_lines": lines,
            "manifest_sha256": digest,
        }
    if config.reverb_prob > 0.0:
        lines, digest = write_bank_manifest(
            directory / REVERB_MANIFEST_NAME, dataset_dir(config.reverb_bank, data_dir)
        )
        blocks["reverb"] = {
            "prob": config.reverb_prob,
            "bank": config.reverb_bank,
            "bank_lines": lines,
            "manifest_sha256": digest,
        }
    return blocks


def _start_payload(
    config: TrainConfig, base: Checkpoint, base_identity: str, drawn: TrainingClips | MixClips
) -> dict:
    """What a run is, for ``config.json``: its config, its base model, its clips.

    The clips are in it because a dataset can be prepared again under the same
    name: the same seed over other clips is another run, and this is what tells
    the two apart. A mix run records the hash per dataset as well as the whole.
    """
    payload = {
        "config": config.to_dict(),
        "base_model": str(base),
        "base_model_identity": base_identity,
        "sampled_ids_sha256": drawn.ids_sha256,
        "clips": len(drawn.clips),
    }
    if isinstance(drawn, MixClips):
        payload["datasets"] = [
            {"name": per.name, "sampled_ids_sha256": per.drawn.ids_sha256} for per in drawn.datasets
        ]
    return payload


def _manifest_entries(
    config: TrainConfig, drawn: TrainingClips | MixClips, data_dir: Path | None
) -> list[tuple[Clip, Path]]:
    """The (clip, its dataset directory) pairs the NeMo manifest is written from.

    Both draws hand their clips sorted by id, so one pass over them gives the
    manifest its order: the manifest is read shuffled, but what a run trains on
    is identified by the ids, so the order it is written in is a fixed one.
    """
    if isinstance(drawn, MixClips):
        directory_of = {
            clip.id: per.directory for per in drawn.datasets for clip in per.drawn.clips
        }
        return [(clip, directory_of[clip.id]) for clip in drawn.clips]
    directory = dataset_dir(config.dataset, data_dir)
    return [(clip, directory) for clip in drawn.clips]


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


def _record(
    config: TrainConfig, training: Training, result: TrainResult, guard: GuardResult
) -> dict:
    """What this run was and what it produced, as it goes into train.json."""
    updates = result.updates or config.updates
    seconds = result.wall_seconds or 0.0
    if isinstance(training.clips, MixClips):
        data_record: dict = {
            "clips": len(training.clips.clips),
            "datasets": [
                {
                    "name": per.name,
                    "clips": len(per.drawn.clips),
                    "clips_available": per.drawn.available,
                    "sampled_ids_sha256": per.drawn.ids_sha256,
                }
                for per in training.clips.datasets
            ],
            "sampled_ids_sha256": training.clips.ids_sha256,
        }
    else:
        data_record = {
            "dataset": config.dataset,
            "clips": len(training.clips.clips),
            "clips_available": training.clips.available,
            "sampled_ids_sha256": training.clips.ids_sha256,
        }
    record = {
        "name": config.name,
        "config": config.to_dict(),
        **data_record,
        "base_model": str(training.base),
        "base_model_identity": training.base_identity,
        "resumed": training.resuming,
        "git_commit": git_commit(),
        # What the leakage guard checked. train.json only, never config.json:
        # config.json is compared to resume an old run, and a new key there
        # would refuse every run written before the guard existed.
        "leakage_guard": {
            "evaluation_datasets_checked": list(guard.evaluation_datasets_checked),
            "missing": list(guard.missing),
            "clips_checked": guard.clips_checked,
        },
        "final_loss": result.final_loss,
        "updates": updates,
        "wall_seconds": round(seconds, 1),
        "updates_per_second": round(updates / seconds, 2) if seconds > 0 else 0.0,
        "peak_vram_gib": round(result.peak_vram_gib, 2),
        "device": result.device,
    }
    # The augmentation block only for a run that augmented: a run that did not
    # keeps the record of a run that could not have had one.
    if training.augmentation:
        record["augmentation"] = training.augmentation
    return record
