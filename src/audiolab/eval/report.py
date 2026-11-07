"""Turn records into ``summary.json`` and ``report.md``.

The JSON holds every number at full precision and is what another program
reads; the Markdown is what a human reads, with the same numbers rounded. Both
are derived only from the records and the panels, so scoring the same run twice
writes the same bytes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from audiolab.eval import metrics, stats
from audiolab.eval.records import (
    IDENTITY_NAME,
    RECORDS_NAME,
    dataset_and_panel,
    latest_records,
    load_panel_items,
    read_records,
    with_records,
)

SUMMARY_NAME = "summary.json"
REPORT_NAME = "report.md"

# 1a measures none of this; saying so beats letting a reader assume it does.
# How many excluded ids the comparison prints before it points at the JSON.
EXCLUDED_IDS_SHOWN = 20

GAPS = (
    "noise and reverberation: not measured in 1a",
    "long-form audio (more than one utterance per request): not measured",
    "term recall, named entities, accents: not measured",
    "semantic score from a language model: not measured",
    "local models on our own GPUs: not measured in 1a",
    "music: the non-speech battery covers silence, noise and everyday sounds only",
    "these panels are small and were chosen by us: absolute numbers use them as the population",
)


def score_run(
    run_dir: str | Path,
    panels_dir: str | Path,
    data_dir: str | Path | None = None,
    *,
    draws: int = stats.DRAWS,
    seed: int = stats.SEED,
) -> dict:
    """Score a run directory and write ``summary.json`` and ``report.md``; returns the summary.

    ``draws`` and ``seed`` are exposed so a test can bound its runtime; the
    defaults are the frozen numbers, which is what a real report uses.
    """
    run_dir, panels_dir = Path(run_dir), Path(panels_dir)
    identity = json.loads((run_dir / IDENTITY_NAME).read_text(encoding="utf-8"))
    records = read_records(run_dir / RECORDS_NAME)
    panels = {
        panel: _panel_items(panel, panels_dir, data_dir, records) for panel in identity["panels"]
    }
    summary = {
        "run": _run_identity(identity, records),
        "panels": {
            panel: panel_summary(panel, items, draws=draws, seed=seed)
            for panel, items in sorted(panels.items())
        },
        "gaps": list(GAPS),
    }
    (run_dir / SUMMARY_NAME).write_text(_clean_json(summary), encoding="utf-8")
    (run_dir / REPORT_NAME).write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _panel_items(
    panel: str, panels_dir: Path, data_dir: Path | None, records: list[dict]
) -> list[metrics.Item]:
    return with_records(load_panel_items(panel, panels_dir, data_dir), records)


def _run_identity(identity: dict, records: list[dict]) -> dict:
    """The run's identity plus what the records themselves say."""
    latest = latest_records(records)
    timestamps = [record["timestamp"] for record in records if record.get("timestamp")]
    return {
        **identity,
        "records": len(records),
        "items": len(latest),
        "failed_requests": sum(1 for record in latest.values() if record.get("error")),
        "first_record": min(timestamps) if timestamps else "",
        "last_record": max(timestamps) if timestamps else "",
    }


def panel_summary(panel: str, items: list[metrics.Item], *, draws: int, seed: int) -> dict:
    """Every number of one panel, with intervals where they mean something."""
    speech = any(item.language != "none" for item in items)
    summary = metrics.speech_summary(items) if speech else metrics.nonspeech_summary(items)
    summary["panel"] = panel
    dataset, name = dataset_and_panel(panel)
    summary["dataset"] = dataset
    summary["panel_name"] = name
    # Intervals of an empty panel are (0, 0): the report still renders, it just
    # has nothing to say about accuracy.
    scored = metrics.score_items(items)
    if speech:
        for view, entry in summary["wer"].items():
            if entry:
                entry["ci"] = _interval(scored, metrics.wer_value(view), draws, seed)
        summary["sentence_error_rate"]["ci"] = _interval(scored, _sentence_value, draws, seed)
    else:
        summary["hallucination_rate"]["ci"] = _interval(
            scored, metrics.hallucination_value, draws, seed
        )
    return summary


def _sentence_value(score: metrics.UtteranceScore) -> tuple[float, float]:
    return (1.0 if score.counts["normalized"].errors else 0.0, 1.0)


def _interval(scored: list[metrics.UtteranceScore], value, draws: int, seed: int) -> dict:
    low, high = stats.ratio_interval(metrics.totals_by_group(scored, value), draws=draws, seed=seed)
    return {"low": low, "high": high}


def compare_runs(
    run_a: str | Path,
    run_b: str | Path,
    panels_dir: str | Path,
    data_dir: str | Path | None = None,
    *,
    draws: int = stats.DRAWS,
    seed: int = stats.SEED,
) -> dict:
    """Paired comparison of two runs over the ids they have in common."""
    run_a, run_b, panels_dir = Path(run_a), Path(run_b), Path(panels_dir)
    identity_a = json.loads((run_a / IDENTITY_NAME).read_text(encoding="utf-8"))
    identity_b = json.loads((run_b / IDENTITY_NAME).read_text(encoding="utf-8"))
    if identity_a["panels"] != identity_b["panels"]:
        raise ValueError(
            "these runs scored different panels; only runs over the same frozen panels compare"
        )
    name_a = identity_a["model"]["name"]
    name_b = identity_b["model"]["name"]
    records_a, records_b = read_records(run_a / RECORDS_NAME), read_records(run_b / RECORDS_NAME)
    comparison = {"a": name_a, "b": name_b, "panels": {}}
    for panel in sorted(identity_a["panels"]):
        items_a, items_b, excluded = _paired_items(
            panel, panels_dir, data_dir, records_a, records_b
        )
        if not items_a:
            raise ValueError(
                f"{panel}: the two runs share no answered ids, so nothing can be compared "
                f"({_exclusion_summary(excluded)})"
            )
        comparison["panels"][panel] = _compare_panel(
            panel, items_a, items_b, name_a, name_b, draws, seed
        )
        comparison["panels"][panel]["excluded"] = excluded
    return comparison


def _paired_items(
    panel: str,
    panels_dir: Path,
    data_dir: Path | None,
    records_a: list[dict],
    records_b: list[dict],
) -> tuple[list[metrics.Item], list[metrics.Item], dict]:
    """The items to compare and the ids left out of it, per run.

    A pair is one clip both runs answered: an id that failed in either run has
    no hypothesis to compare with the other run, so it leaves the comparison —
    and is listed, because a comparison that silently drops the items a model
    could not answer is exactly how a broken tenth of a panel looks like a win.
    """
    latest_a, latest_b = latest_records(records_a), latest_records(records_b)
    answered_a = {key for key, record in latest_a.items() if not record.get("error")}
    answered_b = {key for key, record in latest_b.items() if not record.get("error")}
    shared = answered_a & answered_b
    items = [item for item in load_panel_items(panel, panels_dir, data_dir) if item.id in shared]
    excluded = {
        "not_recorded_in_a": sorted(set(latest_b) - set(latest_a)),
        "not_recorded_in_b": sorted(set(latest_a) - set(latest_b)),
        "failed_in_a": sorted(set(latest_a) - answered_a),
        "failed_in_b": sorted(set(latest_b) - answered_b),
    }
    every = set()
    for ids in excluded.values():
        every.update(ids)
    excluded["ids"] = sorted(every)
    return (
        with_records(items, records_a),
        with_records(items, records_b),
        excluded,
    )


def _exclusion_summary(excluded: dict) -> str:
    """How many ids were left out and why, in one line."""
    return (
        ", ".join(
            f"{len(excluded[key])} {key.replace('_', ' ')}"
            for key in ("not_recorded_in_a", "not_recorded_in_b", "failed_in_a", "failed_in_b")
            if excluded[key]
        )
        or "nothing was left out"
    )


def _compare_panel(
    panel: str,
    items_a: list[metrics.Item],
    items_b: list[metrics.Item],
    name_a: str,
    name_b: str,
    draws: int,
    seed: int,
) -> dict:
    """Paired difference of the headline number, with a verdict."""
    scored_a, scored_b = metrics.score_items(items_a), metrics.score_items(items_b)
    speech = any(item.language != "none" for item in items_a)
    value = metrics.wer_value("normalized") if speech else metrics.hallucination_value
    totals_a = metrics.totals_by_group(scored_a, value)
    totals_b = metrics.totals_by_group(scored_b, value)
    low, high = stats.difference_interval(totals_a, totals_b, draws=draws, seed=seed)
    result = {
        "items": len(items_a),
        "groups": len(metrics.totals_by_group(scored_a, value)),
        "kind": "speech" if speech else "nonspeech",
        "metric": "wer_normalized" if speech else "hallucination_rate",
        "a": stats.ratio(totals_a),
        "b": stats.ratio(totals_b),
        "difference": stats.ratio(totals_b) - stats.ratio(totals_a),
        "ci": {"low": low, "high": high},
        "verdict": stats.verdict(low, high, name_a, name_b),
    }
    return result


def render_comparison(comparison: dict) -> str:
    """The human-readable comparison of two runs."""
    lines = [
        f"# Comparison: {comparison['a']} (A) against {comparison['b']} (B)",
        "",
        "Paired over the ids both runs recorded. A difference is called a difference only",
        "when the 95 % interval of B - A stays away from zero.",
        "",
        "| panel | metric | A | B | B - A | 95 % CI | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for panel, section in comparison["panels"].items():
        interval = f"[{_percent(section['ci']['low'])}, {_percent(section['ci']['high'])}]"
        lines.append(
            f"| {panel} | {section['metric']} | {_percent(section['a'])} "
            f"| {_percent(section['b'])} | {_percent(section['difference'])} "
            f"| {interval} | {section['verdict']} |"
        )
    lines += [
        "",
        f"{sum(section['items'] for section in comparison['panels'].values())} items compared.",
        "",
    ]
    for panel, section in comparison["panels"].items():
        if not section["excluded"]["ids"]:
            continue
        listed = section["excluded"]["ids"][:EXCLUDED_IDS_SHOWN]
        more = len(section["excluded"]["ids"]) - len(listed)
        lines += [
            f"- {panel}: {len(section['excluded']['ids'])} ids left out "
            f"({_exclusion_summary(section['excluded'])}): {', '.join(listed)}"
            + (f", and {more} more" if more else ""),
            "  (the JSON comparison lists every excluded id)",
        ]
    if not any(section["excluded"]["ids"] for section in comparison["panels"].values()):
        lines += ["No id was left out: both runs answered every clip of every panel."]
    return "\n".join(lines) + "\n"


def render_markdown(summary: dict) -> str:
    """The human-readable report: same numbers, rounded, plus what is missing."""
    run = summary["run"]
    lines = [
        f"# Evaluation report: {run['model']['name']} on suite {run['suite']}",
        "",
        f"- model id: `{run['model']['model']}`",
        f"- served by: {run['model']['server_label'] or 'not recorded'}",
        f"- audiolab {run['audiolab_version']}, normalizer version {run['normalizer_version']}",
        f"- git commit: `{run['git_commit'] or 'unknown'}`",
        f"- records: {run['records']} lines for {run['items']} items, "
        f"{run['failed_requests']} failed requests",
        f"- recorded from {run['first_record'] or 'n/a'} to {run['last_record'] or 'n/a'}",
        "",
    ]
    for section in summary["panels"].values():
        lines += _panel_section(section)
    lines += ["## What this does not measure", ""]
    lines += [f"- {gap}" for gap in summary["gaps"]]
    return "\n".join(lines) + "\n"


def _warm_calls(latency: dict) -> str:
    """How many calls the latency covers, naming the cold ones only when there are some."""
    warm = f"{latency['warm_items']} warm calls"
    if not latency["cold_items"]:
        return warm
    return f"{warm}, {latency['cold_items']} cold excluded"


def _panel_section(section: dict) -> list[str]:
    lines = [f"## {section['panel']}", ""]
    if section["kind"] == "speech":
        lines += _speech_section(section)
    else:
        lines += _nonspeech_section(section)
    lines += [
        "",
        f"- latency: median {_number(section['latency']['median_s'])} s, "
        f"p95 {_number(section['latency']['p95_s'])} s over {_warm_calls(section['latency'])}",
        f"- {section['latency']['label']}: median {_number(section['latency']['median_rtf'])}, "
        f"mean {_number(section['latency']['mean_rtf'])}",
        "",
    ]
    return lines


def _speech_section(section: dict) -> list[str]:
    lines = [
        f"- utterances: {section['items']} scored {section['scored_items']}, "
        f"{section['groups']} groups, {section['failed_requests']} failed requests",
        "",
        "| view | WER | 95 % CI | errors | reference words |",
        "| --- | --- | --- | --- | --- |",
    ]
    for view, entry in section["wer"].items():
        if entry is None:
            lines.append(f"| {view} | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {view} | {_percent(entry['value'])} | "
            f"[{_percent(entry['ci']['low'])}, {_percent(entry['ci']['high'])}] | "
            f"{entry['errors']} | {entry['reference_words']} |"
        )
    lines += [
        "",
        f"- sentence error rate: {_percent(section['sentence_error_rate']['value'])} "
        f"(CI [{_percent(section['sentence_error_rate']['ci']['low'])}, "
        f"{_percent(section['sentence_error_rate']['ci']['high'])}])",
        f"- per-utterance WER: p90 {_percent(section['tail']['p90'])}, "
        f"p95 {_percent(section['tail']['p95'])}",
        f"- utterances above 25 % WER: {_percent(section['tail']['share_over_25']['value'])}, "
        f"above 50 %: {_percent(section['tail']['share_over_50']['value'])}",
        f"- loops: {_percent(section['loop_rate']['value'])}, "
        f"truncations: {_percent(section['truncation_rate']['value'])}, "
        f"empty outputs: {_percent(section['empty_output_rate']['value'])}",
    ]
    return lines


def _nonspeech_section(section: dict) -> list[str]:
    lines = [
        f"- clips: {section['items']} scored {section['scored_items']}, "
        f"{section['failed_requests']} failed requests",
        f"- hallucination rate: {_percent(section['hallucination_rate']['value'])} "
        f"(CI [{_percent(section['hallucination_rate']['ci']['low'])}, "
        f"{_percent(section['hallucination_rate']['ci']['high'])}])",
        f"- words per clip: mean {_number(section['words']['mean'])}, "
        f"max {section['words']['max']}, "
        f"{_percent(section['words']['share_at_least_5']['value'])} with 5 words or more",
        "",
    ]
    if section["top_hypotheses"]:
        lines += ["| output | count |", "| --- | --- |"]
        lines += [
            f"| {_cell(entry['text'])} | {entry['count']} |" for entry in section["top_hypotheses"]
        ]
        lines += [""]
    else:
        lines += ["No hypothesis was produced on any clip.", ""]
    lines += _family_table(section)
    return lines


def _family(category: str) -> str:
    """The ESC-50 family of a class label, so the table has a handful of rows."""
    return category.split("/")[0]


def _family_table(section: dict) -> list[str]:
    """The category breakdown merged by family; per class it would be 50 rows."""
    items: dict[str, int] = {}
    invented: dict[str, int] = {}
    words: dict[str, float] = {}
    for name, entry in section["by_category"].items():
        family = _family(name)
        items[family] = items.get(family, 0) + entry["items"]
        invented[family] = invented.get(family, 0) + entry["hallucinated"]
        words[family] = words.get(family, 0.0) + entry["mean_words"] * entry["items"]
    lines = ["| category | clips | hallucination rate | mean words |", "| --- | --- | --- | --- |"]
    lines += [
        f"| {family} | {items[family]} | {_percent(invented[family] / items[family])} | "
        f"{_number(words[family] / items[family])} |"
        for family in sorted(items)
    ]
    return lines


def _cell(text: str, limit: int = 60) -> str:
    """Markdown cells cannot hold a newline or a pipe."""
    collapsed = " ".join(text.split()).replace("|", "\\|")
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def _percent(value: float) -> str:
    return f"{value * 100:.2f} %"


def _number(value: float) -> str:
    return f"{value:.2f}"


def _clean_json(summary: dict) -> str:
    """JSON with sorted keys and no NaN, so the same run scores to the same bytes."""
    return json.dumps(_finite(summary), indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _finite(value):
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
