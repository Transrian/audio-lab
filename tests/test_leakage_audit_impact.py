"""The rebuild impact report, on a synthetic data root.

No prepared data is needed: the fake root holds a handful of sets whose clips
are crafted so that every rule fires — one clip in an evaluation group, one
pair of within-set re-uploads, one NASA clip in a reserved-episode class, one
SLR57 speaker, one VoxPopuli clip in a shared session, one babble utterance —
and the rule inputs are built with the real committed functions against
crafted ids that satisfy their contracts (a twin made of equal title and
duration, a speaker in the test part's format, a session from the committed
pin, an utterance from the committed babble list).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from audiolab.data import duplicates as dup
from audiolab.data import nasa as nasa_mod
from audiolab.data import noisy, voxpopuli_train
from audiolab.data.african_french import excluded_train_speakers
from audiolab.data.librispeech_pc import babble_exclusions
from audiolab.data.talk_listing import Talk, read_listing, reading_order
from audiolab.data.talks import SEED
from audiolab.data.train import evaluation_groups

SCRIPT = Path(__file__).resolve().parent.parent / "experiments/leakage-audit/impact.py"
spec = importlib.util.spec_from_file_location("leakage_impact", SCRIPT)
impact = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = impact
spec.loader.exec_module(impact)


def row(name: str, id_suffix: str, group: str, kind: str = "recording") -> dict:
    """One manifest line, with every required field of a prepared clip."""
    return {
        "id": f"{name}:{id_suffix}",
        "audio": f"audio/{name}_{id_suffix}.wav",
        "sha256": hashlib.sha256(f"{name}:{id_suffix}".encode()).hexdigest(),
        "duration_s": 5.0,
        "language": "en",
        "text": "a sentence of some length",
        "text_has_casing_punct": True,
        "group": f"{name}:{group}",
        "group_kind": kind,
        "source_split": "listing",
        "license": "CC0-1.0",
    }


def write_dataset(root: Path, name: str, rows: list[dict]) -> None:
    (root / name).mkdir(parents=True)
    (root / name / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# The crafted talks, one record per id. The twins share a normalized title and
# a duration within the tolerance, so the real duplicate_classes merges them.
PEERTUBE_TALKS = [
    Talk(
        id="syn-eval",
        url="",
        title="Synthetic Evaluation",
        duration_s=3600,
        license="CC0",
        author="a",
    ),
    Talk(id="syn-a", url="", title="Synthetic Alpha", duration_s=3600, license="CC0", author="a"),
    Talk(id="syn-b", url="", title="synthetic alpha!", duration_s=3601, license="CC0", author="a"),
    Talk(
        id="syn-clean", url="", title="Synthetic Clean", duration_s=1800, license="CC0", author="a"
    ),
]
DEBCONF_TALKS = [
    Talk(
        id="syn-d1",
        url="",
        title="Synthetic Debconf One",
        duration_s=2400,
        license="CC0",
        author="b",
    ),
    Talk(
        id="syn-d2",
        url="",
        title="synthetic debconf one",
        duration_s=2401,
        license="CC0",
        author="b",
    ),
]


def synthetic_root(root: Path) -> dict[str, str]:
    """One data root where every rule fires; returns the ids it depends on."""
    reserved_slug = min(nasa_mod.RESERVED_TEST_EPISODES)
    real_episodes = read_listing(nasa_mod.SOURCE)[1]
    original = next(e for e in real_episodes if e.id == reserved_slug)
    nasa_talks = [
        Talk(
            id="syn-nasa-twin",
            url="",
            title=original.title,
            duration_s=original.duration_s,
            license=original.license,
            author=original.author,
            channel=original.channel,
        ),
        Talk(
            id="syn-n1",
            url="",
            title="Synthetic First Show",
            duration_s=5000,
            license="CC0",
            author="c",
            channel="show1",
        ),
        Talk(
            id="syn-n2",
            url="",
            title="synthetic first show",
            duration_s=5001,
            license="CC0",
            author="c",
            channel="show1",
        ),
        Talk(
            id="syn-n-clean",
            url="",
            title="Synthetic Clean Show",
            duration_s=7000,
            license="CC0",
            author="c",
            channel="show2",
        ),
    ]
    session = sorted(voxpopuli_train.EVALUATION_SESSIONS)[0]
    assert "20991231-0900-PLENARY-99" not in voxpopuli_train.EVALUATION_SESSIONS
    babble_id = min(uid for ids in noisy.load_babble_utterances().values() for uid in ids)

    write_dataset(
        root, "peertube-edu-captions-fr", [row("peertube-edu-captions-fr", "syn-eval", "syn-eval")]
    )
    write_dataset(
        root,
        "peertube-edu-fr-train",
        [
            row("peertube-edu-fr-train", "syn-eval", "syn-eval"),
            row("peertube-edu-fr-train", "syn-a", "syn-a"),
            row("peertube-edu-fr-train", "syn-b", "syn-b"),
            row("peertube-edu-fr-train", "syn-clean", "syn-clean"),
        ],
    )
    write_dataset(
        root,
        "peertube-edu-fr-train-40k",
        [
            row("peertube-edu-fr-train-40k", "syn-a", "syn-a"),
            row("peertube-edu-fr-train-40k", "syn-b", "syn-b"),
        ],
    )
    write_dataset(
        root,
        "debconf-en-train",
        [
            row("debconf-en-train", "syn-d1", "syn-d1"),
            row("debconf-en-train", "syn-d2", "syn-d2"),
        ],
    )
    write_dataset(root, "nasa-en-test", [row("nasa-en-test", reserved_slug, reserved_slug)])
    write_dataset(
        root,
        "nasa-en-train",
        [
            row("nasa-en-train", "syn-nasa-twin", "syn-nasa-twin"),
            row("nasa-en-train", "syn-n1", "syn-n1"),
            row("nasa-en-train", "syn-n2", "syn-n2"),
            row("nasa-en-train", "syn-n-clean", "syn-n-clean"),
        ],
    )
    write_dataset(
        root,
        "nasa-en-train-30k",
        [
            row("nasa-en-train-30k", "syn-n-clean", "syn-n-clean"),
            # A talk no longer in the pinned listing: the set was prepared from
            # an earlier one. No rule counts it; the report says so.
            row("nasa-en-train-30k", "syn-stale-episode", "syn-stale-episode"),
        ],
    )
    write_dataset(
        root, "nasa-en-train-qwen", [row("nasa-en-train-qwen", "syn-n-clean", "syn-n-clean")]
    )
    write_dataset(
        root,
        "african-accented-fr-test",
        [row("african-accented-fr-test", "cameroon_m_012", "cameroon_m_012", "speaker")],
    )
    write_dataset(
        root,
        "african-accented-fr-train",
        [
            row("african-accented-fr-train", "afc_002", "012", "speaker"),
            row("african-accented-fr-train", "afc_013", "013", "speaker"),
            row("african-accented-fr-train", "ctell1-01", "ctell1-01", "speaker"),
        ],
    )
    write_dataset(
        root,
        "voxpopuli-fr-train",
        [
            row("voxpopuli-fr-train", f"{session}-fr_20090310-11:00:00_0", "100", "speaker"),
            row(
                "voxpopuli-fr-train",
                "20991231-0900-PLENARY-99-fr_20991231-11:00:00_0",
                "101",
                "speaker",
            ),
        ],
    )
    write_dataset(
        root,
        "librispeech-pc-en-train",
        [
            row("librispeech-pc-en-train", babble_id, "123", "speaker"),
            row("librispeech-pc-en-train", "123/1234/123-1234-0001", "123", "speaker"),
        ],
    )
    write_dataset(
        root,
        "librispeech-en-train",
        [
            row("librispeech-en-train", babble_id, "123", "speaker"),
            row("librispeech-en-train", "123/1234/123-1234-0001", "123", "speaker"),
        ],
    )
    return {
        "reserved_slug": reserved_slug,
        "nasa_talks": nasa_talks,
        "real_episodes": real_episodes,
        "session": session,
        "babble_id": babble_id,
    }


def synthetic_inputs(root: Path, ids: dict[str, str]) -> impact.RuleInputs:
    """The rule inputs, with every rule built by its real committed function.

    The classes are computed by the real duplicate_classes over the real
    listings plus the crafted talks; the orders by the real reading_order;
    the reserved NASA episodes are the real committed constant; the SLR57
    mapping by the real excluded_train_speakers on the synthetic test part;
    the sessions and the babble are the real committed pins.
    """
    peertube = dup.duplicate_classes(dup.peertube_talks() + PEERTUBE_TALKS, domain="peertube")
    debconf = dup.duplicate_classes(DEBCONF_TALKS, domain="debconf-en-train")
    nasa = dup.duplicate_classes(ids["real_episodes"] + ids["nasa_talks"], domain="nasa")
    return impact.RuleInputs(
        talk_classes={
            "peertube-edu-fr-train": peertube,
            "debconf-en-train": debconf,
            nasa_mod.NAME: nasa,
        },
        talk_order={
            "peertube-edu-fr-train": tuple(t.id for t in reading_order(PEERTUBE_TALKS, SEED, None)),
            "debconf-en-train": tuple(t.id for t in reading_order(DEBCONF_TALKS, SEED, None)),
            nasa_mod.NAME: tuple(
                t.id
                for t in reading_order(
                    ids["real_episodes"] + ids["nasa_talks"], SEED, nasa_mod.SOURCE.spread_by
                )
            ),
        },
        eval_groups={
            "peertube-edu-fr-train": evaluation_groups(
                "peertube-edu-captions-fr", "peertube-edu-captions-fr-v2", root=root
            ),
            "debconf-en-train": frozenset(),
            nasa_mod.NAME: evaluation_groups("nasa-en-test", "nasa-en-test-v2", root=root),
        },
        nasa_reserved=frozenset(nasa_mod.RESERVED_TEST_EPISODES),
        slr57_excluded=excluded_train_speakers(
            evaluation_groups("african-accented-fr-test", root=root)
        ),
        vox_sessions=voxpopuli_train.EVALUATION_SESSIONS,
        babble=noisy.load_babble_utterances(),
    )


def test_every_rule_fires_and_the_counts_hold(tmp_path):
    root = tmp_path / "data"
    ids = synthetic_root(root)
    inputs = synthetic_inputs(root, ids)
    rows = {row.name: row for row in impact.audit(impact.training_sets(root), root, inputs)}

    def counts(row: impact.SetImpact) -> dict[str, int]:
        return {rule: row.count(rule) for rule in impact.RULES}

    assert counts(rows["peertube-edu-fr-train"]) == {
        "evaluation group or its class": 1,
        "re-upload of a talk already read": 1,
        "nasa reserved-test-episode class": 0,
        "slr57 speaker": 0,
        "voxpopuli shared session": 0,
        "babble utterance": 0,
    }
    assert rows["peertube-edu-fr-train"].clips == 4
    assert rows["peertube-edu-fr-train"].total == 2
    # The sized set: the same within-set pair, no evaluation hit.
    assert rows["peertube-edu-fr-train-40k"].count("re-upload of a talk already read") == 1
    assert rows["peertube-edu-fr-train-40k"].total == 1
    # The 12 000-clip set: only within-set re-uploads.
    assert rows["debconf-en-train"].count("re-upload of a talk already read") == 1
    assert rows["debconf-en-train"].total == 1
    # NASA: the twin of a reserved episode is removed by both the prepared
    # test set's groups and the committed reserved list; the pair is a re-upload.
    assert counts(rows["nasa-en-train"]) == {
        "evaluation group or its class": 1,
        "re-upload of a talk already read": 1,
        "nasa reserved-test-episode class": 1,
        "slr57 speaker": 0,
        "voxpopuli shared session": 0,
        "babble utterance": 0,
    }
    assert rows["nasa-en-train"].total == 2  # the twin and one of the pair; the other is kept
    assert rows["nasa-en-train-30k"].total == 0
    # A talk no longer in the pinned listing is not counted by any rule; the
    # row records it as stale, and the report prints it.
    assert rows["nasa-en-train-30k"].stale == frozenset({"syn-stale-episode"})
    report = impact.render(root, list(rows.values()), inputs, [])
    assert "syn-stale-episode" in report
    assert rows["nasa-en-train-qwen"].total == 0
    # SLR57: the CA16 twin of a test speaker, as the real function maps it.
    assert rows["african-accented-fr-train"].count("slr57 speaker") == 1
    assert rows["african-accented-fr-train"].total == 1
    # VoxPopuli: the clip in a session of the committed pin.
    assert rows["voxpopuli-fr-train"].count("voxpopuli shared session") == 1
    assert rows["voxpopuli-fr-train"].total == 1
    # Babble: the utterance of the committed list.
    assert rows["librispeech-pc-en-train"].count("babble utterance") == 1
    assert rows["librispeech-en-train"].count("babble utterance") == 1
    assert "123/1234/123-1234-0001" not in babble_exclusions(
        ("123/1234/123-1234-0001",), noisy.load_babble_utterances()
    )


def test_the_rebuild_list(tmp_path):
    root = tmp_path / "data"
    ids = synthetic_root(root)
    inputs = synthetic_inputs(root, ids)
    rows = impact.audit(impact.training_sets(root), root, inputs)
    rebuild, not_rebuilt = impact.rebuild_list(rows)
    rebuilt = dict(rebuild)
    assert set(rebuilt) == {
        "african-accented-fr-train",
        "librispeech-pc-en-train",
        "nasa-en-train",
        "nasa-en-train-30k",
        "nasa-en-train-qwen",
        "peertube-edu-fr-train",
        "peertube-edu-fr-train-40k",
        "voxpopuli-fr-train",
    }
    # The derived sets are in the list through their source, not their own counts.
    assert rebuilt["nasa-en-train-30k"] == ("derived from nasa-en-train",)
    assert rebuilt["nasa-en-train-qwen"] == ("derived from nasa-en-train",)
    # The sized set is rebuilt for its own within-set re-uploads.
    assert any("sized set" in why for why in rebuilt["peertube-edu-fr-train-40k"])
    # The 12 000-clip set that only loses re-uploads is not rebuilt, and
    # neither is the babble pool that the babble rule removes clips from.
    assert dict(not_rebuilt) == {
        "debconf-en-train": "only in-set re-uploads in a 12 000-clip set",
        "librispeech-en-train": "babble pool: the noisy sets' committed babble",
    }


def test_the_inputs_report_which_evaluation_sets_are_prepared(tmp_path):
    root = tmp_path / "data"
    synthetic_root(root)
    inputs, status = impact.compute_rule_inputs(root)
    found = {(source, dataset): groups for source, dataset, groups in status}
    assert found[("peertube-edu-fr-train", "peertube-edu-captions-fr")] == 1
    assert found[("peertube-edu-fr-train", "peertube-edu-captions-fr-v2")] is None
    assert found[("nasa-en-train", "nasa-en-test")] == 1
    assert found[("nasa-en-train", "nasa-en-test-v2")] is None
    assert found[("african-accented-fr-train", "african-accented-fr-test")] == 1
    # The rule inputs themselves read the same prepared data.
    assert inputs.eval_groups["peertube-edu-fr-train"] == {"syn-eval"}
    assert inputs.eval_groups["nasa-en-train"] == {min(nasa_mod.RESERVED_TEST_EPISODES)}
    assert inputs.slr57_excluded == {
        "cameroon_m_012": "a test speaker",
        "012": "the CA16 speaker of test speaker cameroon_m_012",
    }


def test_a_malformed_manifest_fails_loudly_and_names_the_set(tmp_path):
    root = tmp_path / "data"
    ids = synthetic_root(root)
    # A line missing a field of a prepared clip.
    line = row("peertube-fr-train", "syn-x", "syn-x")
    del line["text"]
    write_dataset(root, "peertube-fr-train", [line])
    inputs = synthetic_inputs(root, ids)
    with pytest.raises(impact.ImpactError, match=r"peertube-fr-train.*missing fields.*'text'"):
        impact.audit(impact.training_sets(root), root, inputs)
    assert impact.main(["--data-dir", str(root)]) == 1


def test_main_runs_the_real_wiring_and_exits_zero(tmp_path, capsys):
    """main() end-to-end: the real committed wiring over a synthetic data root.

    The set's clips carry real talk ids of the pinned listings, so the report
    computes the classes from the real listings, reads the synthetic evaluation
    set's groups as the reserved ones, and prints the table.
    """
    from audiolab.data import talk_sources

    root = tmp_path / "data"
    _, talks = read_listing(talk_sources.SOURCES["peertube-edu-fr-train"])
    first, second = talks[0].id, talks[1].id
    write_dataset(root, "peertube-edu-captions-fr", [row("peertube-edu-captions-fr", first, first)])
    write_dataset(
        root,
        "peertube-edu-fr-train",
        [
            row("peertube-edu-fr-train", first, first),
            row("peertube-edu-fr-train", second, second),
        ],
    )
    assert impact.main(["--data-dir", str(root)]) == 0
    out = capsys.readouterr().out
    for line in (
        "| peertube-edu-fr-train | 2 | 1 |",
        "peertube-edu-captions-fr-v2 (evaluation set of peertube-edu-fr-train): NOT prepared",
        "- peertube-edu-fr-train — eval class: 1 clip",
    ):
        assert line in out


def test_main_exits_two_without_a_data_root(tmp_path):
    assert impact.main(["--data-dir", str(tmp_path / "nowhere")]) == 2
