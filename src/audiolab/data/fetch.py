"""``audiolab data fetch``: download a dataset and prepare it once.

Preparation is idempotent: a dataset that already has a manifest is verified
and left alone, so a second run costs a hash of every clip and nothing else.
Re-running with ``--force`` deletes and prepares again.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from audiolab.core.paths import cache_dir, data_dir
from audiolab.data.codeswitched import fetch_servicenow_codeswitched
from audiolab.data.csfleurs import fetch_cs_fleurs_fr_en
from audiolab.data.datasets import DATASETS
from audiolab.data.fleurs import fetch_fleurs
from audiolab.data.granary import fetch_granary_fr_train
from audiolab.data.librispeech import fetch_librispeech_en
from audiolab.data.mls import fetch_mls_fr, fetch_mls_fr_train
from audiolab.data.nonspeech import fetch_nonspeech
from audiolab.data.store import read_manifest, verify_clips
from audiolab.data.voxpopuli import fetch_voxpopuli_french_accent
from audiolab.data.voxpopuli_train import fetch_voxpopuli_fr_train

FETCHERS = {
    "fleurs-fr": fetch_fleurs,
    "fleurs-en": fetch_fleurs,
    "granary-fr-train": fetch_granary_fr_train,
    "mls-fr": fetch_mls_fr,
    "mls-fr-train": fetch_mls_fr_train,
    "librispeech-en": fetch_librispeech_en,
    "nonspeech": fetch_nonspeech,
    "voxpopuli-en-french-accent": fetch_voxpopuli_french_accent,
    "servicenow-codeswitched-en-fr": fetch_servicenow_codeswitched,
    "servicenow-codeswitched-en-fr-ca": fetch_servicenow_codeswitched,
    "cs-fleurs-fr-en": fetch_cs_fleurs_fr_en,
    "voxpopuli-fr-train": fetch_voxpopuli_fr_train,
}


def fetch_dataset(name: str, *, force: bool = False, verbose: bool = True) -> Path:
    """Make ``name`` ready under the data directory and return that directory."""
    if name not in FETCHERS:
        known = ", ".join(sorted(DATASETS))
        raise ValueError(f"unknown dataset {name!r}; known datasets: {known}")
    destination = data_dir() / name
    manifest = destination / "manifest.jsonl"

    if manifest.is_file() and not force:
        problems = verify_clips(destination, read_manifest(destination))
        if problems:
            raise RuntimeError(
                f"{name} is prepared but {len(problems)} clips are wrong "
                f"(first: {problems[0]}); run with --force to prepare it again"
            )
        if verbose:
            print(f"{name}: already prepared ({len(read_manifest(destination))} clips), verified")
        return destination

    if force and destination.exists():
        shutil.rmtree(destination)
    if verbose:
        print(f"{name}: preparing (cache: {cache_dir() / name})")
    count = FETCHERS[name](name, cache_dir(), destination, verbose=verbose)
    if verbose:
        print(f"{name}: {count} clips prepared in {destination}")
    return destination
