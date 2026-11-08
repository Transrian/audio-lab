"""What a training run leaves behind, and where it lives.

One run directory per training config, under ``$AUDIOLAB_RUNS/train/<name>/``:

- ``config.json``     what the run was started as: its config, its base model and
                      the clips it drew, written before it trains;
- ``manifest.jsonl``  the clips this run trains on, in NeMo's manifest format;
- ``train-log.jsonl`` one line per logged interval, with the loss;
- ``resume.ckpt``     the one resume slot, rewritten as the run goes, deleted
                      when the run finishes;
- ``model.nemo``      the fine-tuned checkpoint, written at the end;
- ``train.json``      what was run, on what, and what came out;
- ``model.yaml``      a model config for ``audiolab eval``, already pointing at
                      ``model.nemo``.

A finished run is not trained again by accident: the directory is the record of
one run, and ``--fresh`` is the only way to start over it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from audiolab.core.paths import runs_dir
from audiolab.train.config import NAME_PATTERN

TRAIN_DIR = "train"
MANIFEST_NAME = "manifest.jsonl"
LOG_NAME = "train-log.jsonl"
RESUME_NAME = "resume.ckpt"
MODEL_NAME = "model.nemo"
TRAIN_JSON = "train.json"
CONFIG_NAME = "config.json"
MODEL_CONFIG = "model.yaml"


class TrainRunError(ValueError):
    """A run directory that cannot be used as asked."""


def run_dir(name: str, runs_root: str | Path | None = None) -> Path:
    """The directory of the run named by a training config."""
    if not NAME_PATTERN.match(name):
        raise TrainRunError(f"{name!r} cannot name a run")
    root = Path(runs_root) if runs_root else runs_dir()
    return root / TRAIN_DIR / name


def is_finished(directory: Path) -> bool:
    """True when a run in this directory reached its end."""
    return (directory / TRAIN_JSON).is_file() and (directory / MODEL_NAME).is_file()


def refuse_finished_run(directory: Path, *, fresh: bool = False) -> None:
    """Refuse to train over a finished run unless the caller asked to start over.

    The sequence calls this before it reads anything, and `prepare_run_dir` calls
    it before it touches the directory, so both refuse in the same words.
    """
    if is_finished(directory) and not fresh:
        raise TrainRunError(
            f"{directory} already holds a finished run ({MODEL_NAME} and {TRAIN_JSON}); "
            "train again with --fresh to start over, or remove the directory to rerun it"
        )


def prepare_run_dir(directory: Path, *, fresh: bool = False) -> Path:
    """Make the run directory ready to train in, keeping an unfinished run's slot."""
    refuse_finished_run(directory, fresh=fresh)
    if fresh and directory.exists():
        # Deleting a directory is the one destructive step here, so it is only
        # this run's own directory: <runs>/train/<name>/.
        if directory.parent.name != TRAIN_DIR:
            raise TrainRunError(
                f"refusing to clear {directory}: it is not inside a {TRAIN_DIR}/ root"
            )
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def resume_checkpoint(directory: Path) -> Path | None:
    """The slot to resume from, when the run stopped before it finished."""
    path = directory / RESUME_NAME
    return path if path.is_file() else None


def write_train_json(directory: Path, payload: dict) -> Path:
    """Write the record of what ran; this file is what says the run is done."""
    path = directory / TRAIN_JSON
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_run_config(directory: Path, payload: dict) -> Path:
    """Write what this run is, before it trains.

    A resume continues one particular run, so what that run was has to be on
    disk: the config in hand can be a different one, and a run resumed with
    another recipe would produce a checkpoint that is neither run.
    """
    path = directory / CONFIG_NAME
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_run_config(directory: Path) -> dict | None:
    """What the run in this directory was started as; ``None`` if nothing says."""
    path = directory / CONFIG_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def config_differences(recorded: dict, wanted: dict, prefix: str = "") -> list[str]:
    """The keys where the run on disk and the run in hand disagree.

    Nested mappings are walked, so a changed config field reads as
    ``config.updates`` rather than as the whole ``config``.
    """
    differences = []
    for key in sorted(set(recorded) | set(wanted)):
        here = f"{prefix}{key}"
        was, now = recorded.get(key), wanted.get(key)
        if isinstance(was, dict) and isinstance(now, dict):
            differences += config_differences(was, now, f"{here}.")
        elif was != now:
            differences.append(here)
    return differences


def log_line(path: Path, payload: dict) -> None:
    """Append one line of the run's log; appended, so a crash keeps what was seen."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def write_model_config(directory: Path, *, name: str, gpu: str) -> Path:
    """Write the model config that evaluates this run's checkpoint.

    Somebody reading a finished run should not have to write a config to hear
    the model, so the run writes one for them.

    The paths in it are absolute: a config that only works from the directory the
    run was started in would fail the first time somebody reads the run from
    somewhere else.
    """
    checkpoint = (directory / MODEL_NAME).resolve()
    path = (directory / MODEL_CONFIG).resolve()
    path.write_text(
        f"""# Written by 'audiolab train' for the checkpoint of this run:
#   audiolab eval --model {path} --suite configs/suites/fast.yaml
name: {name}
model: {name}
api: transcriptions
send_language: true
server_label: this machine
# No endpoint: 'audiolab eval' starts the server for this checkpoint itself.
serve:
  backend: nemo
  checkpoint: {json.dumps(str(checkpoint))}
  gpu: {json.dumps(gpu)}
""",
        encoding="utf-8",
    )
    return path
