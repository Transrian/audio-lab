"""``audiolab data fetch`` sizing a training preparation: routing and validation.

The fetcher is a fake that writes a one-clip manifest, so these tests check
where a ``--clips`` preparation lands, what it is recorded under, and which
names and caps are refused, all offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataclasses import replace  # noqa: E402

import pytest  # noqa: E402
from helpers import make_clip, write_tone  # noqa: E402

from audiolab import cli  # noqa: E402
from audiolab.core.audio import pcm_sha256  # noqa: E402
from audiolab.data.datasets import base_dataset  # noqa: E402
from audiolab.data.fetch import FETCHERS, fetch_dataset, sized_dataset_name, validate_clips_cap  # noqa: E402
from audiolab.data.store import read_manifest, write_manifest  # noqa: E402


def test_base_dataset_strips_the_cap_suffix():
    assert base_dataset("granary-fr-train") == "granary-fr-train"
    assert base_dataset("granary-fr-train-40k") == "granary-fr-train"
    assert base_dataset("nasa-en-train-30k") == "nasa-en-train"
    assert base_dataset("fleurs-fr") == "fleurs-fr"
    assert base_dataset("peertube-tech-fr-train-40k") == "peertube-tech-fr-train"


def test_validate_clips_cap_refuses_what_a_larger_preparation_is_not():
    validate_clips_cap(13_000)
    validate_clips_cap(40_000)
    with pytest.raises(ValueError, match="multiple of 1 000"):
        validate_clips_cap(13_500)
    with pytest.raises(ValueError, match="13 000"):
        validate_clips_cap(12_000)
    with pytest.raises(ValueError, match="13 000"):
        validate_clips_cap(5_000)


def test_sized_dataset_name_names_the_directory_after_the_cap():
    assert sized_dataset_name("granary-fr-train", 40_000) == "granary-fr-train-40k"
    assert sized_dataset_name("nasa-en-train", 30_000) == "nasa-en-train-30k"
    with pytest.raises(ValueError, match="-train"):
        sized_dataset_name("fleurs-fr", 40_000)


def fake_fetcher(seen: dict):
    """A fetcher that records what it was asked and prepares one clip under it."""

    def fetch(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool, clips=None) -> int:
        seen["name"] = name
        seen["clips"] = clips
        dataset_dir.mkdir(parents=True, exist_ok=True)
        clip = make_clip(f"{name}:clip-1", group=f"{name}:group-0")
        path = write_tone(dataset_dir / clip.audio, seconds=1.0)
        write_manifest(dataset_dir, [replace(clip, sha256=pcm_sha256(path))])
        return 1

    return fetch


def test_a_sized_preparation_lands_in_its_own_directory_under_its_own_name(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOLAB_DATA", str(tmp_path))
    monkeypatch.setitem(FETCHERS, "granary-fr-train", fake_fetcher(seen := {}))

    destination = fetch_dataset("granary-fr-train", clips=40_000, verbose=False)

    assert destination == tmp_path / "granary-fr-train-40k"
    assert seen == {"name": "granary-fr-train-40k", "clips": 40_000}
    (clip,) = read_manifest(destination)
    assert clip.id == "granary-fr-train-40k:clip-1", "the id carries the directory's name"
    assert clip.group == "granary-fr-train-40k:group-0"


def test_the_usual_preparation_keeps_its_directory_and_its_name(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOLAB_DATA", str(tmp_path))
    monkeypatch.setitem(FETCHERS, "granary-fr-train", fake_fetcher(seen := {}))

    destination = fetch_dataset("granary-fr-train", verbose=False)

    assert destination == tmp_path / "granary-fr-train"
    assert seen == {"name": "granary-fr-train", "clips": None}


def test_a_sized_name_routed_without_clips_verifies_the_prepared_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOLAB_DATA", str(tmp_path))
    monkeypatch.setitem(FETCHERS, "granary-fr-train", fake_fetcher(seen := {}))
    destination = fetch_dataset("granary-fr-train", clips=40_000, verbose=False)

    again = fetch_dataset("granary-fr-train-40k", verbose=False)

    assert again == destination
    assert seen == {"name": "granary-fr-train-40k", "clips": 40_000}, "nothing is re-prepared"


def test_clips_is_refused_for_an_evaluation_set(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOLAB_DATA", str(tmp_path))
    with pytest.raises(ValueError, match="-train"):
        fetch_dataset("fleurs-fr", clips=40_000)


def test_clips_is_refused_for_a_name_that_already_names_its_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOLAB_DATA", str(tmp_path))
    with pytest.raises(ValueError, match="already names its cap"):
        fetch_dataset("granary-fr-train-40k", clips=40_000)


def test_clips_is_refused_for_a_source_whose_cap_is_fixed(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOLAB_DATA", str(tmp_path))

    def fixed(name: str, cache_dir: Path, dataset_dir: Path, *, verbose: bool) -> int:
        raise AssertionError("a source without a clips parameter is not sized")

    monkeypatch.setitem(FETCHERS, "mls-fr-train", fixed)
    with pytest.raises(ValueError, match="not supported"):
        fetch_dataset("mls-fr-train", clips=40_000)


def test_cli_fetch_refuses_clips_for_an_evaluation_set(tmp_path, capsys):
    code = cli.main(["--data-dir", str(tmp_path), "data", "fetch", "fleurs-fr", "--clips", "40000"])
    assert code == 2
    assert "-train" in capsys.readouterr().err


def test_cli_fetch_refuses_a_cap_that_is_not_a_number(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "--data-dir",
                str(tmp_path),
                "data",
                "fetch",
                "granary-fr-train",
                "--clips",
                "40k",
            ]
        )
    assert exit_info.value.code == 2


@pytest.mark.parametrize("value", ["13500", "12000", "5000"])
def test_cli_fetch_refuses_an_invalid_cap(tmp_path, capsys, value):
    code = cli.main(
        ["--data-dir", str(tmp_path), "data", "fetch", "granary-fr-train", "--clips", value]
    )
    assert code == 2, value


def test_cli_fetch_refuses_clips_without_a_name(tmp_path, capsys):
    code = cli.main(["--data-dir", str(tmp_path), "data", "fetch", "--clips", "40000"])
    assert code == 2
    assert "give its name" in capsys.readouterr().err
