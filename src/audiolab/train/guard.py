"""The leakage guard: no clip of an evaluation dataset may be trained on.

Every training run - single dataset or mix - is checked before its model is
loaded, against the evaluation datasets of the suite: the datasets behind the
panels listed in ``configs/suites/full.yaml``. Four things are refused, because
any one of them would make the run's evaluation score meaningless:

- a training clip whose audio ``sha256`` is the audio of a clip of a prepared
  evaluation dataset. That hash is of the prepared clip's PCM, so it matches
  prepared audio byte for byte: the same prepared clip, and not the same
  recording prepared another way;
- a training clip whose id prefix is an evaluation dataset name (the clip is
  named after a dataset that the panels score);
- a training clip of a talk or NASA source - or of a dataset derived from one,
  a relabelled set of the source's own clips - whose talk or episode is in the
  duplicate class of a talk or episode of a prepared evaluation set: the
  re-upload, re-release or replay of an evaluation recording, whose prepared
  audio is other bytes (decision 0002; the classes are computed the way the
  preparations compute them, over the source's pinned listing, through
  ``audiolab.data.duplicates``);
- a babble utterance of the noisy evaluation sets, in a run that is not a
  resume: the babble a prepared noisy set mixes is the committed corpus ids of
  ``audiolab/data/babble_utterances.json``, taken from the two babble pools
  (their cards: ``docs/stt/datasets.md``, ``mls-fr-train`` and
  ``librispeech-en-train``), and a training run must not hold an utterance a
  frozen panel mixes under it. Resuming a run is not a new run: a run that
  started before the rule and holds a babble utterance may still be resumed,
  and the babble is the one check a resume does not apply.

An evaluation dataset that is not prepared on this machine cannot be checked,
so the run says so - one warning line per missing dataset - and goes on with a
partial check rather than failing.

What the guard does **not** do: it does not compare group ids across different
corpora, because ids collide by accident across corpora. Nor does it recognize
a recording that is only the *same one* when the class rules cannot say so:
speech re-recorded or re-edited under another title and another length - or
the same audio cut, trimmed, resampled, re-encoded or levelled differently -
hashes to something else and passes; only a duplicate class (the same title
and a close length, or a confirmed pair of the source's committed list) or a
byte-for-byte sha256 reaches such a copy. The classes exist only for the talk
and NASA sources; a clip of any other source is checked by the sha256 and the
name, and by nothing else. A talk or episode the source's pinned listing does
not hold has no class at all, so a clip of it passes the class check without a
word: the guard, like the preparations, only knows the listing. Keeping
evaluation recordings out of a training preparation is the job of the fetchers,
through their ``evaluation_groups`` and duplicate-class exclusions; the guard is
the last check, on the clips a run is actually about to train.

The run record keeps what was checked: the datasets, the missing ones and the
number of training clips checked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from audiolab.core.paths import data_dir as default_data_dir
from audiolab.core.progress import say
from audiolab.data import nasa
from audiolab.data.datasets import base_dataset
from audiolab.data.duplicates import duplicate_classes, peertube_talks
from audiolab.data.noisy import babble_union, load_babble_utterances
from audiolab.data.panels import (
    DEFAULT_PANELS_DIR,
    SuiteError,
    load_suite,
    panel_path,
    read_panel,
)
from audiolab.data.store import Clip, read_manifest
from audiolab.data.talk_listing import read_listing
from audiolab.data.talk_sources import SOURCES
from audiolab.data.talks import PEERTUBE_SOURCES
from audiolab.data.train import evaluation_groups
from audiolab.train.data import TrainDataError
from audiolab.train.run import TrainRunError

# The suite and the panels it names are given the way the eval command gives
# them - relative to the working directory, which is the repository root in the
# commands the documentation shows - and the panels directory is the very
# constant the eval command defaults to, so there is one convention for it.
FULL_SUITE = Path("configs/suites/full.yaml")
FULL_PANELS = DEFAULT_PANELS_DIR
MANIFEST = "manifest.jsonl"

# The evaluation sets a talk or NASA preparation leaves out, and the source
# whose corpus each belongs to: the registry the preparations read from, so a
# source that gains an evaluation set reaches the guard the day the registry
# says so.
EVALUATION_DATASET_SOURCE = {
    dataset: source.name for source in SOURCES.values() for dataset in source.evaluation_datasets
}

# A training dataset prepared from another dataset's own clips under its own
# name: no new recording, only the labels change (a relabelled set, as
# ``src/audiolab/data/relabel.py`` prepares one). Its clips still come from
# the source's corpus, so the duplicate-class check resolves the name to that
# source. One line per derived set, as a source gains one.
DERIVED_TRAINING_DATASETS = {
    "nasa-en-train-qwen": "nasa-en-train",
}


def _clip_source(dataset_name: str) -> str | None:
    """The talk or NASA source a training dataset's clips come from, if any.

    The dataset's own name with a cap suffix off, then one step through the
    derived sets: a relabelled set's clips are the source's, so a clip's talk
    or episode takes its class from the source's corpus.
    """
    name = base_dataset(dataset_name)
    if name in DERIVED_TRAINING_DATASETS:
        name = DERIVED_TRAINING_DATASETS[name]
    return name if name in SOURCES else None


def _class_corpus(source_name: str) -> tuple[list, str]:
    """The talk population and the duplicate domain of one talk source.

    The rule a preparation applies when it excludes the duplicate classes of
    its evaluation sets: the PeerTube sources are one corpus listed three
    times, so their classes span the union of the three pinned listings; NASA's
    confirmed pairs are committed under the domain ``"nasa"``; any other source
    is its own corpus - its pinned listing, its own domain.
    """
    if source_name in PEERTUBE_SOURCES:
        return peertube_talks(), "peertube"
    source = SOURCES[source_name]
    domain = "nasa" if source_name == nasa.NAME else source_name
    return read_listing(source)[1], domain


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
    resuming: bool = False,
    babble_path: str | Path | None = None,
    print_line: Callable[[str], None] = say,
) -> GuardResult:
    """Refuse a run that would train on what the suite scores.

    ``clips`` are the training clips as drawn for the run. Every prepared
    evaluation dataset contributes the audio ``sha256`` of all its clips; any
    training clip that matches one, or that is named after an evaluation
    dataset, is a refusal. A training clip of a talk or NASA source, or of a
    dataset derived from one, whose talk or episode is in the duplicate class
    of a talk or episode of a prepared evaluation set is a refusal too: the
    re-upload or re-release of an evaluation recording, prepared another way.
    And a run that is not a resume
    is refused when any of its clips is a babble utterance of the noisy
    evaluation sets - the committed corpus ids of ``babble_path``, the noisy
    sets' prepared babble - because a frozen panel mixes it under the test
    clips.

    An evaluation dataset that is not prepared here names a gap: the check is
    then partial, and that is printed, not hidden.
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

    # The duplicate classes of the talk and NASA sources involved: the sources
    # of the prepared evaluation sets, and the sources the training clips come
    # from. Only those read a listing and build a class table. A class that
    # holds a group of a prepared evaluation set may not be trained on; the
    # classes are computed over the pinned listings, the very rule the
    # preparations apply.
    sources = {
        EVALUATION_DATASET_SOURCE[name] for name in checked if name in EVALUATION_DATASET_SOURCE
    }
    for clip in clips:
        source = _clip_source(clip.id.partition(":")[0])
        if source is not None:
            sources.add(source)
    domain_of_source: dict[str, str] = {}
    population_by_domain: dict[str, list] = {}
    for source in sorted(sources):
        population, domain = _class_corpus(source)
        domain_of_source[source] = domain
        population_by_domain.setdefault(domain, population)
    classes_by_domain = {
        domain: duplicate_classes(population, domain=domain)
        for domain, population in population_by_domain.items()
    }
    # For each class that holds an evaluation group: the prepared evaluation
    # sets and the talks in it, in the suite's order - what a refusal names.
    eval_class_sets: dict[str, dict[str, list[str]]] = {}
    eval_class_talks: dict[str, dict[str, list[str]]] = {}
    for name in checked:
        source = EVALUATION_DATASET_SOURCE.get(name)
        if source is None:
            continue
        domain = domain_of_source[source]
        classes = classes_by_domain[domain]
        for group in sorted(evaluation_groups(name, root=root)):
            class_id = classes.get(group)
            if class_id is None:
                continue
            if name not in eval_class_sets.setdefault(domain, {}).setdefault(class_id, []):
                eval_class_sets[domain][class_id].append(name)
            if group not in eval_class_talks.setdefault(domain, {}).setdefault(class_id, []):
                eval_class_talks[domain][class_id].append(group)

    # The babble rule is a new-run rule: resuming a run that started before it
    # is not a new run, and a resume is checked for the other things alone.
    babble: set[str] | None = None
    if not resuming:
        babble = set(babble_union(load_babble_utterances(babble_path)))

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
        source = _clip_source(prefix)
        if source is not None:
            domain = domain_of_source[source]
            classes = classes_by_domain[domain]
            talk_id = clip.group.partition(":")[2]
            class_id = classes.get(talk_id)
            if class_id is not None and class_id in eval_class_sets.get(domain, {}):
                sets = eval_class_sets[domain][class_id]
                raise TrainDataError(
                    f"leakage guard: training clip {clip.id} is a duplicate of the "
                    f"evaluation talk {eval_class_talks[domain][class_id][0]} of "
                    f"{', '.join(sets)}; the run would train on what the suite scores"
                )
        if babble is not None and clip.id.partition(":")[2] in babble:
            raise TrainDataError(
                f"leakage guard: training clip {clip.id} is a babble utterance of the noisy "
                "evaluation sets, which a new training run must not hold: the pools' "
                "cards are docs/stt/datasets.md, mls-fr-train and librispeech-en-train"
            )
    print_line(
        f"leakage guard: {len(clips)} training clips checked against {eval_clips} "
        f"evaluation clips of {len(checked)} datasets "
        f"(missing: {', '.join(missing) if missing else 'none'})"
    )
    return GuardResult(tuple(checked), tuple(missing), len(clips))
