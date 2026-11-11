"""The leakage guard: no clip of an evaluation dataset may be trained on.

Every training run - single dataset or mix - is checked before its model is
loaded, against the evaluation datasets of the suite: the datasets behind the
panels listed in ``configs/suites/full.yaml``. Two things are refused, because
either one would make the run's evaluation score meaningless:

- a training clip whose audio ``sha256`` is the audio of a clip of a prepared
  evaluation dataset (the same recording in both);
- a training clip whose id prefix is an evaluation dataset name (the clip is
  named after a dataset that the panels score).

An evaluation dataset that is not prepared on this machine cannot be checked,
so the run says so - one warning line per missing dataset - and goes on with a
partial check rather than failing.

What the guard does **not** do: it does not compare group ids across
different corpora, because ids collide by accident across corpora. Keeping
evaluation recordings out of a training preparation is the job of the
fetchers, through their ``evaluation_groups`` exclusion; the guard is the
last check, on the clips a run is actually about to train.

The run record keeps what was checked: the datasets, the missing ones and the
number of training clips checked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from audiolab.core.paths import data_dir as default_data_dir
from audiolab.core.progress import say
from audiolab.data.panels import (
    SuiteError,
    load_suite,
    panel_path,
    read_panel,
)
from audiolab.data.store import Clip, read_manifest
from audiolab.train.data import TrainDataError
from audiolab.train.run import TrainRunError

# The suite and the panels it names live in the repository, not in the data
# directory, so the defaults are anchored to the repository root rather than
# to wherever the process happens to run.
_REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_SUITE = _REPO_ROOT / "configs" / "suites" / "full.yaml"
FULL_PANELS = _REPO_ROOT / "configs" / "panels"
MANIFEST = "manifest.jsonl"


@dataclass(frozen=True)
class GuardResult:
    """What the leakage guard checked, for the run's record."""

    evaluation_datasets_checked: tuple[str, ...]
    missing: tuple[str, ...]
    clips_checked: int


def evaluation_datasets(suite_path: str | Path, panels_dir: str | Path) -> list[str]:
    """The datasets behind a suite's panels, in the suite's order, each once.

    The suite is read by the reader the eval command uses, so a suite format
    change reaches the guard and the scorer together.
    """
    path = Path(suite_path)
    try:
        spec = load_suite(path)
    except FileNotFoundError as failure:
        raise TrainRunError(
            f"{path} is missing: the leakage guard needs the suite it checks"
        ) from failure
    except SuiteError as failure:
        raise TrainRunError(str(failure)) from failure
    names: list[str] = []
    for title in spec.panels:
        panel_file = panel_path(title, panels_dir)
        if not panel_file.is_file():
            raise TrainRunError(f"{panel_file} is missing: the suite {path} names it")
        dataset = read_panel(panel_file).dataset
        if dataset not in names:
            names.append(dataset)
    return names


def check_leakage(
    clips: list[Clip],
    *,
    data_dir: str | Path | None = None,
    suite_path: str | Path = FULL_SUITE,
    panels_dir: str | Path = FULL_PANELS,
    print_line: Callable[[str], None] = say,
) -> GuardResult:
    """Refuse a run that would train on what the suite scores.

    ``clips`` are the training clips as drawn for the run. Every prepared
    evaluation dataset contributes the audio ``sha256`` of all its clips; any
    training clip that matches one, or that is named after an evaluation
    dataset, is a refusal. An evaluation dataset that is not prepared here
    names a gap: the check is then partial, and that is printed, not hidden.
    """
    root = Path(data_dir) if data_dir else default_data_dir()
    eval_datasets = evaluation_datasets(suite_path, panels_dir)
    checked: list[str] = []
    missing: list[str] = []
    eval_sha256: set[str] = set()
    eval_clips = 0
    for name in eval_datasets:
        manifest = root / name / MANIFEST
        if not manifest.is_file():
            missing.append(name)
            print_line(
                f"leakage guard: evaluation dataset {name} is not prepared here, "
                "so the check is partial"
            )
            continue
        clips_of_dataset = read_manifest(root / name)
        eval_sha256.update(clip.sha256 for clip in clips_of_dataset)
        eval_clips += len(clips_of_dataset)
        checked.append(name)
    eval_names = set(eval_datasets)
    for clip in clips:
        if clip.sha256 in eval_sha256:
            raise TrainDataError(
                f"leakage guard: training clip {clip.id} is the audio of an evaluation clip "
                f"(sha256 {clip.sha256[:12]}…); the run would train on what the suite scores"
            )
        prefix = clip.id.split(":", 1)[0]
        if prefix in eval_names:
            raise TrainDataError(
                f"leakage guard: training clip {clip.id} is named after the evaluation "
                f"dataset {prefix!r}; the run would train on what the suite scores"
            )
    print_line(
        f"leakage guard: {len(clips)} training clips checked against {eval_clips} "
        f"evaluation clips of {len(checked)} datasets "
        f"(missing: {', '.join(missing) if missing else 'none'})"
    )
    return GuardResult(tuple(checked), tuple(missing), len(clips))
