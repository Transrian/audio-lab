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
from audiolab.eval.records import RECORDS_NAME, read_records  # noqa: E402

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


def change(result, panel, candidate):
    """The one recorded conclusion change for a panel/candidate pair."""
    found = [
        item for item in result["changes"] if item["panel"] == panel and item["source"] == candidate
    ]
    assert len(found) == 1, found
    return found[0]


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
    # An apostrophe before the letter is an English quoted expression
    # ("rock 'n' roll"), not a French elision: joining it would invent a word.
    assert rescore_mod.apply_elision("rock 'n' roll") == "rock 'n' roll"
    assert rescore_mod.apply_elision("j'ai dit rock 'n' roll") == "j'ai dit rock 'n' roll"


def test_elision_covers_every_kind_of_spacing():
    """The no-break spaces a model writes are the spaces the normalizer reads."""
    assert rescore_mod.apply_elision("j'  ai") == "j'ai"  # two spaces
    assert rescore_mod.apply_elision("d'\u00a0accord") == "d'accord"  # no-break space
    assert rescore_mod.apply_elision("d'\u202faccord") == "d'accord"  # narrow no-break space
    # The transform maps the typographic apostrophes first, as the normalizer
    # does; nothing else is rewritten, not even a dash.
    assert rescore_mod.diagnostic_text("d\u2019accord") == "d'accord"
    assert rescore_mod.diagnostic_text("a\u2013b \u00abc\u00bb") == "a\u2013b \u00abc\u00bb"


def test_only_the_apostrophes_are_rewritten():
    """A dash or a quotation mark is not the script's business, only the apostrophe is."""
    from audiolab.eval.normalize import TYPOGRAPHIC

    for character in TYPOGRAPHIC:
        if character in ("\u2018", "\u2019", "\u02bc"):
            assert rescore_mod.diagnostic_text(f"a{character}b") == "a'b"
        else:
            assert rescore_mod.diagnostic_text(f"a{character}b") == f"a{character}b"


def test_an_elision_quoted_at_the_start_of_a_phrase_stays_unjoined():
    """The lookbehind cannot tell the elision from the quotation mark before it.

    Only the case where the phrase's opening quote carries the elision's
    apostrophe itself: the letter has a quotation mark before it, so the rule
    leaves the spacing alone. That is the chosen behaviour (the alternative would
    join a quoted ``'n'`` too), and no hypothesis of the screening runs, base or
    candidate, contains one.
    """
    assert rescore_mod.apply_elision("il a dit 'j' ai faim'") == "il a dit 'j' ai faim'"
    assert rescore_mod.diagnostic_text("il a dit 'j' ai faim'") == "il a dit 'j' ai faim'"
    # The same words after a space outside a quotation are joined as usual.
    assert rescore_mod.apply_elision("il a dit j' ai faim") == "il a dit j'ai faim"


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
    assert rescore_mod.verdict(official) == "gain"
    # Under the diagnostic both sides are expanded, so the win disappears.
    assert diagnostic["difference"] == 0.0 and diagnostic["ci"] == (0.0, 0.0)
    assert rescore_mod.verdict(diagnostic) == "neutral"
    assert change(result, EN, "contractions")["official"] == "gain"
    assert change(result, EN, "contractions")["diagnostic"] == "neutral"


def test_elision_run_flips_gain_to_neutral(runs):
    result = rescore_all(runs, "elision")
    official, diagnostic = (
        delta(result, FR, "elision", "official"),
        delta(result, FR, "elision", "diagnostic"),
    )
    assert official["difference"] < -0.5 and official["ci"][1] < 0
    assert diagnostic["difference"] == 0.0
    assert change(result, FR, "elision") == {
        "panel": FR,
        "source": "elision",
        "official": "gain",
        "diagnostic": "neutral",
        "position": diagnostic["draws"],
    }
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
    assert all(change["panel"] != NONSPEECH for change in result["changes"])


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


def test_the_scoring_code_version_is_what_is_checked(runs):
    """A run the code no longer scores under is refused, even as the base run."""
    path = runs["base"] / "run.json"
    identity = json.loads(path.read_text(encoding="utf-8"))
    identity["normalizer_version"] = 2  # every run agrees, the code does not
    path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(rescore_mod.Problem, match="scores with the code's version 1"):
        rescore_all(runs, "contractions")


def test_rules_ablation_changes_only_its_own_panel(runs):
    elision_only = rescore_all(runs, "contractions", rules="elision")
    # With only the elision rule, the English contraction gain stays separable.
    official = delta(elision_only, EN, "contractions", "official")
    diagnostic = delta(elision_only, EN, "contractions", "diagnostic")
    assert official == diagnostic
    assert rescore_mod.verdict(official) == "gain"


def test_the_official_column_is_the_code_scoring_the_untransformed_items(runs):
    """With no rule, both columns are one scoring path, so they must agree."""
    result = rescore_all(runs, "contractions", rules="none")
    assert result["changes"] == []
    for section in result["panels"].values():
        for entry in section["candidates"].values():
            assert entry["official"] == entry["diagnostic"]


def test_the_change_prints_the_draws_around_zero(runs):
    """The reported counts come from the same draws as the interval they explain."""
    from audiolab.eval.report import _paired_items, compare_runs

    comparison = compare_runs(
        runs["base"], runs["contractions"], runs["panels_dir"], runs["data_dir"], draws=DR
    )
    items_base, items_cand, _ = _paired_items(
        EN,
        runs["panels_dir"],
        runs["data_dir"],
        read_records(runs["base"] / RECORDS_NAME),
        read_records(runs["contractions"] / RECORDS_NAME),
    )
    # The same seed as compare_runs' own draws (its default, audiolab.eval.stats.SEED),
    # so the two sides are the same draws and the equality below means they agree,
    # not that a difference fell on the boundary.
    delta_ = rescore_mod.paired_delta(
        items_base, items_cand, rescore_mod._identity, draws=DR, seed=rescore_mod.stats.SEED
    )
    assert delta_["ci"] == (
        comparison["panels"][EN]["ci"]["low"],
        comparison["panels"][EN]["ci"]["high"],
    )
    below, on, total = delta_["draws"]["below"], delta_["draws"]["on"], delta_["draws"]["total"]
    assert total == DR
    assert below + on <= DR
    # The reported counts are the interval's own draw distribution, so the side
    # of zero the counts describe is the side the interval ends on.
    assert (delta_["ci"][1] < 0) == (below == DR)


def test_a_hypothesis_that_does_not_score_leaves_the_numbers():
    """The transform path drops what cannot be scored, exactly as the official path does."""

    def item(clip_id, hypothesis):
        return rescore_mod.metrics.Item(
            id=clip_id,
            panel=EN,
            group=f"group-{clip_id}",
            category="",
            language="en",
            reference=EN_REFERENCE,
            hypothesis=hypothesis,
        )

    items = [
        item("good-1", "it's a test don't worry"),
        item("good-2", "it's a test don't worry"),
        item("unreadable", "it's a test don't worry unreadable"),
    ]

    def counts(reference, hypothesis, language, view):
        if "unreadable" in hypothesis:
            raise ValueError("synthetic unreadable hypothesis")
        return original(reference, hypothesis, language, view)

    original = rescore_mod.metrics.counts
    rescore_mod.metrics.counts = counts
    try:
        delta_ = rescore_mod.paired_delta(items, items, rescore_mod._identity, draws=DR, seed=1)
    finally:
        rescore_mod.metrics.counts = original
    # The unreadable answer contributes nothing: not its errors, not its
    # reference words, not its group. The two that scored are the whole score.
    assert delta_["groups"] == 2
    assert delta_["base"] == delta_["candidate"] == 0.0
    assert delta_["difference"] == 0.0
    assert delta_["ci"] == (0.0, 0.0)


def test_a_divergence_in_any_candidate_stops_the_run(runs):
    """Every candidate's official column is checked, not only the first one's."""
    from audiolab.eval.report import compare_runs

    seen: list[Path] = []

    def mostly_official(base, candidate, panels_dir, data_dir, **kwargs):
        real = compare_runs(base, candidate, panels_dir, data_dir, **kwargs)
        seen.append(candidate)
        if len(seen) == 2:  # the second candidate is the one that diverges
            for section in real["panels"].values():
                section["difference"] += 10.0
        return real

    rescore_mod.compare_runs = mostly_official
    try:
        with pytest.raises(rescore_mod.Problem, match="official column diverges"):
            rescore_all(runs, "contractions", "elision")
    finally:
        rescore_mod.compare_runs = compare_runs
    assert len(seen) == 2, seen  # the first candidate passed, the second was reached


def test_a_candidate_label_replaces_the_guessed_name(runs):
    result = rescore_mod.rescore(
        runs["base"],
        [(None, runs["contractions"]), ("My source", runs["elision"])],
        runs["panels_dir"],
        runs["data_dir"],
        draws=DR,
        verify_official=False,
    )
    assert "My source" in result["panels"][EN]["candidates"]
    assert "My source" in rescore_mod.render(result)
    assert rescore_mod.parse_candidate("My source=/tmp/run") == ("My source", Path("/tmp/run"))
    assert rescore_mod.parse_candidate("/tmp/run") == (None, Path("/tmp/run"))


def test_the_report_says_which_version_scored_it(runs):
    result = rescore_all(runs, "elision")
    assert result["normalizer_version"] == 1
    text = rescore_mod.render(result)
    assert "normalizer version used for these numbers: 1" in text
    # Where a verdict sits on zero is printed, not asserted from memory.
    assert "draws on zero, below it" in text
    position = change(result, FR, "elision")["position"]
    assert position["total"] == DR
    assert position["on"] + position["below"] <= DR


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
