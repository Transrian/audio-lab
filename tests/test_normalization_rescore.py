"""Contractions and elision re-scoring, on synthetic CPU-only runs and real scoring code."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import make_clip  # noqa: E402

from audiolab.data.panels import Panel  # noqa: E402
from audiolab.data.store import write_manifest  # noqa: E402

SCRIPT = Path(__file__).resolve().parent.parent / "experiments/normalization-review/rescore.py"
spec = importlib.util.spec_from_file_location("normalization_rescore", SCRIPT)
rescore_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rescore_mod)

EN = "fake-en-full"
FR = "fake-fr-full"
NONSPEECH = "fake-ns-full"
EN_REFERENCE = "it's a test don't worry"
FR_REFERENCE = "j'ai un accord"
N = 12
DR = 200


def rescore_all(runs, *candidates, **kwargs):
    return rescore_mod.rescore(
        runs["base"],
        [runs[name] for name in candidates],
        runs["panels_dir"],
        runs["data_dir"],
        draws=DR,
        seed=20261107,
        **kwargs,
    )


@pytest.fixture
def runs(tmp_path):
    panels_dir, data_dir = tmp_path / "panels", tmp_path / "data"
    panels_dir.mkdir()
    reference = {EN: EN_REFERENCE, FR: FR_REFERENCE, NONSPEECH: ""}
    language = {EN: "en", FR: "fr", NONSPEECH: "none"}
    ids = {}
    for title in (EN, FR, NONSPEECH):
        dataset = title.rsplit("-", 1)[0]
        clips = [
            make_clip(
                f"{dataset}:{index:03d}",
                language=language[title],
                group=f"{dataset}:group-{index // 3}",
                text=reference[title],
            )
            for index in range(N)
        ]
        write_manifest(data_dir / dataset, clips)
        ids[title] = [clip.id for clip in clips]
        panel = Panel(
            name="full",
            dataset=dataset,
            ids=ids[title],
            items=N,
            seed=20261107,
            manifest_sha256="synthetic",
            clips_sha256="synthetic",
            dataset_line=dataset,
            built="2026-11-09",
        )
        (panels_dir / f"{title}.txt").write_text(panel.text(), encoding="utf-8")
    paths = {}
    for model, en_hypothesis, fr_hypothesis in (
        # The base expands English contractions and writes the spaced elision.
        ("base", "it is a test do not worry", "j' ai un accord"),
        # One run writes verbatim English, the other the unspaced elision.
        ("contractions", "it's a test don't worry", "j' ai un accord"),
        ("elision", "it is a test do not worry", "j'ai un accord"),
    ):
        path = tmp_path / model
        path.mkdir()
        (path / "run.json").write_text(
            json.dumps(
                {
                    "model": {"name": model},
                    "normalizer_version": 1,
                    "panels": {title: "hash" for title in (EN, FR, NONSPEECH)},
                }
            ),
            encoding="utf-8",
        )
        records = []
        for title in (EN, FR, NONSPEECH):
            for index, clip_id in enumerate(ids[title]):
                if title == EN:
                    hypothesis = en_hypothesis
                elif title == FR:
                    hypothesis = fr_hypothesis
                else:
                    hypothesis = "yeah" if index % 3 == 0 else ""
                records.append(
                    {"id": clip_id, "panel": title, "hypothesis": hypothesis, "error": ""}
                )
        (path / "records.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        paths[model] = path
    return {**paths, "panels_dir": panels_dir, "data_dir": data_dir, "ids": ids}


def delta(result, panel, candidate, view):
    return result["panels"][panel]["candidates"][candidate][view]


# ---------------------------------------------------------------------------
# The rules themselves.


def test_expansion_table_and_pronoun_s():
    assert rescore_mod.expand_contraction("it's") == "it is"
    assert rescore_mod.expand_contraction("don't") == "do not"
    assert rescore_mod.expand_contraction("i'm") == "i am"
    assert rescore_mod.expand_contraction("we're") == "we are"
    assert rescore_mod.expand_contraction("can't") == "cannot"
    assert rescore_mod.expand_contraction("won't") == "will not"
    assert rescore_mod.expand_contraction("let's") == "let us"
    for pronoun in rescore_mod.S_PRONOUNS:
        assert rescore_mod.expand_contraction(f"{pronoun}'s") == f"{pronoun} is"


def test_ambiguous_forms_are_left_alone():
    # 'd is never expanded, possessives never, and the rest of the text stays.
    assert rescore_mod.expand_contraction("he'd") == "he'd"
    assert rescore_mod.expand_contraction("john's") == "john's"
    assert rescore_mod.expand_contraction("o'clock") == "o'clock"
    assert rescore_mod.expand_contraction("rock'n'roll") == "rock'n'roll"
    # 'll and 've expand on any stem.
    assert rescore_mod.expand_contraction("it'll") == "it will"
    assert rescore_mod.expand_contraction("you've") == "you have"


def test_contractions_apply_on_whole_text():
    assert (
        rescore_mod.apply_contractions("It's what he'd do, don't worry, we're here")
        == "it is what he'd do, do not worry, we are here"
    )
    assert rescore_mod.apply_contractions("john's car is fine") == "john's car is fine"


def test_typographic_apostrophes_are_handled():
    assert rescore_mod.diagnostic_text("it\u2019s fine") == "it is fine"
    assert rescore_mod.diagnostic_text("d\u2019 accord") == "d'accord"


def test_elision_drops_the_space_and_only_the_space():
    assert rescore_mod.apply_elision("j' ai un d' accord") == "j'ai un d'accord"
    assert rescore_mod.apply_elision("qu' il a dit, jusqu' ici") == "qu'il a dit, jusqu'ici"
    assert rescore_mod.apply_elision("l'homme est parti") == "l'homme est parti"
    # A missing apostrophe is a different defect; the rule does not invent one.
    assert rescore_mod.apply_elision("l homme est parti") == "l homme est parti"


def test_diagnostic_text_combines_both_rules():
    assert (
        rescore_mod.diagnostic_text("j' ai dit it's d' accord, he'd say")
        == "j'ai dit it is d'accord, he'd say"
    )


def test_transform_for_selects_the_rules():
    assert rescore_mod.transform_for("both") is rescore_mod.diagnostic_text
    assert rescore_mod.transform_for("none")("it's") == "it's"
    assert rescore_mod.transform_for("contractions")("j' ai") == "j' ai"
    assert rescore_mod.transform_for("elision")("it's") == "it's"
    with pytest.raises(rescore_mod.Problem):
        rescore_mod.transform_for("nope")


# ---------------------------------------------------------------------------
# End to end, on synthetic runs, through the official pairing and intervals.


def test_contraction_run_flips_gain_to_neutral(runs):
    result = rescore_all(runs, "contractions")
    official, diagnostic = (
        delta(result, EN, "contractions", "official"),
        delta(result, EN, "contractions", "diagnostic"),
    )
    # Officially the verbatim run wins on the contracted reference.
    assert official["difference"] < -0.5 and official["ci"][1] < 0
    assert rescore_mod.classify(official) == "gain"
    # Under the diagnostic both sides are expanded, so the win disappears.
    assert diagnostic["difference"] == 0.0 and diagnostic["ci"] == (0.0, 0.0)
    assert rescore_mod.classify(diagnostic) == "neutral"
    assert (EN, "contractions", "gain", "neutral") in result["changes"]


def test_elision_run_flips_gain_to_neutral(runs):
    result = rescore_all(runs, "elision")
    official, diagnostic = (
        delta(result, FR, "elision", "official"),
        delta(result, FR, "elision", "diagnostic"),
    )
    assert official["difference"] < -0.5 and official["ci"][1] < 0
    assert diagnostic["difference"] == 0.0
    assert (FR, "elision", "gain", "neutral") in result["changes"]
    # The English panel of that run was already expanded: the official view
    # and the diagnostic view agree there, and no conclusion changes.
    assert delta(result, EN, "elision", "official")["difference"] == 0.0


def test_nonspeech_is_unchanged_by_the_rules(runs):
    result = rescore_all(runs, "contractions")
    official, diagnostic = (
        delta(result, NONSPEECH, "contractions", "official"),
        delta(result, NONSPEECH, "contractions", "diagnostic"),
    )
    assert official["difference"] == diagnostic["difference"] == 0.0
    assert all(panel != NONSPEECH for panel, _, _, _ in result["changes"])


def test_official_column_matches_compare_runs(runs):
    from audiolab.eval.report import compare_runs

    result = rescore_all(runs, "contractions")
    comparison = compare_runs(
        runs["base"], runs["contractions"], runs["panels_dir"], runs["data_dir"], draws=DR
    )
    for panel in result["panels"]:
        theirs = comparison["panels"][panel]
        mine = result["panels"][panel]["candidates"]["contractions"]["official"]
        assert abs(theirs["difference"] - mine["difference"]) < 1e-12
        assert abs(theirs["ci"]["low"] - mine["ci"][0]) < 1e-12
        assert abs(theirs["ci"]["high"] - mine["ci"][1]) < 1e-12


def test_mixed_normalizer_versions_are_refused(runs):
    path = runs["contractions"] / "run.json"
    identity = json.loads(path.read_text(encoding="utf-8"))
    identity["normalizer_version"] = 2
    path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(rescore_mod.Problem, match="normalizer version"):
        rescore_all(runs, "contractions")


def test_rules_ablation_changes_only_its_own_panel(runs):
    elision_only = rescore_all(runs, "contractions", rules="elision")
    # With only the elision rule, the English contraction gain stays separable.
    official = delta(elision_only, EN, "contractions", "official")
    diagnostic = delta(elision_only, EN, "contractions", "diagnostic")
    assert official == diagnostic
    assert rescore_mod.classify(official) == "gain"


def test_missing_run_is_refused(tmp_path, capsys):
    argv = [
        "--base",
        str(tmp_path / "absent"),
        "--candidate",
        str(tmp_path / "other"),
        "--runs-dir",
        str(tmp_path),
    ]
    assert rescore_mod.main(argv) == 2
    assert "missing" in capsys.readouterr().err


def test_help_needs_no_data_or_environment(capsys, monkeypatch):
    monkeypatch.delenv("AUDIOLAB_RUNS", raising=False)
    with pytest.raises(SystemExit) as result:
        rescore_mod.main(["--help"])
    assert result.value.code == 0
    text = capsys.readouterr().out
    assert "--candidate" in text and "example:" in text
    assert "normalization-review/rescore.py" in text
