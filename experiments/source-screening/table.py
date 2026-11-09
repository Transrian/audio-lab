#!/usr/bin/env python3
"""Report retained screening runs using the official comparison, never a new scorer.

Each candidate is paired separately with the baseline: its answered-id subset,
and therefore its baseline value, can differ. Frozen comparison settings apply.
Only run records, summaries, panel files and dataset manifests are read; no model,
audio or network is used. See the adjacent README for the invocation and limits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audiolab.data.panels import read_panel
from audiolab.eval.records import IDENTITY_NAME, RECORDS_NAME, panel_hashes, read_records
from audiolab.eval.report import SUMMARY_NAME, compare_runs, render_comparison


class Problem(ValueError):
    """An input cannot be reported as a complete, comparable screening run."""


def json_of(path: Path) -> dict:
    """Read a JSON object, keeping its filename in input-error messages."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as failure:
        raise Problem(f"{path}: cannot be read as JSON ({failure})") from failure
    if not isinstance(value, dict):
        raise Problem(f"{path}: expected a JSON object")
    return value


class Run:
    """A complete run with the exact frozen panels supplied to this report."""

    def __init__(self, path: Path, panels_dir: Path) -> None:
        self.path = path
        self.identity = json_of(path / IDENTITY_NAME)
        declared = self.identity.get("panels")
        if not isinstance(declared, dict) or not declared:
            raise Problem(f"{path / IDENTITY_NAME}: names no panels")
        if any(panel.endswith("-sealed") for panel in declared):
            raise Problem(f"{path}: sealed final panels are not for exploratory screening")
        if declared != panel_hashes(list(declared), panels_dir):
            raise Problem(f"{path}: a frozen panel has changed since this run scored it")
        wanted = {
            (panel, clip_id)
            for panel in declared
            for clip_id in read_panel(panels_dir / f"{panel}.txt").ids
        }
        if not (path / RECORDS_NAME).is_file():
            raise Problem(f"{path / RECORDS_NAME}: missing")
        try:
            records = read_records(path / RECORDS_NAME)
        except ValueError as failure:
            raise Problem(f"{path / RECORDS_NAME}: invalid JSON ({failure})") from failure
        if any(
            not isinstance(record, dict)
            or not isinstance(record.get("id"), str)
            or not isinstance(record.get("panel"), str)
            for record in records
        ):
            raise Problem(f"{path / RECORDS_NAME}: every record needs an id and panel")
        recorded = {(record["panel"], record["id"]) for record in records}
        if recorded != wanted:
            missing, extra = wanted - recorded, recorded - wanted
            raise Problem(
                f"{path / RECORDS_NAME}: an incomplete run cannot be compared "
                f"({len(missing)} not recorded, {len(extra)} unknown); "
                f"first mismatch: {min(missing or extra)}"
            )
        self.summary = json_of(path / SUMMARY_NAME).get("panels")
        if (
            not isinstance(self.summary, dict)
            or set(self.summary) != set(declared)
            or any(not isinstance(section, dict) for section in self.summary.values())
        ):
            raise Problem(
                f"{path / SUMMARY_NAME}: panels do not match the run; "
                "finish the run and use `audiolab score` with its original scoring rules"
            )


def formatted_table(baseline: Run, candidate: Run) -> str:
    """Stored absolute formatted WER, not a paired estimate or a new verdict."""
    rows = []
    for panel in baseline.identity["panels"]:
        values = []
        for run in (baseline, candidate):
            wer = run.summary[panel].get("wer") or {}
            if not isinstance(wer, dict):
                raise Problem(f"{run.path / SUMMARY_NAME}: invalid WER section for {panel}")
            entry = wer.get("formatted")
            if entry is not None and not isinstance(entry, dict):
                raise Problem(f"{run.path / SUMMARY_NAME}: invalid formatted WER for {panel}")
            values.append(entry.get("value") if entry is not None else None)
        if all(value is not None for value in values):
            rows.append(f"| {panel} | {100 * values[0]:.2f} % | {100 * values[1]:.2f} % |")
    heading = "### Formatted WER (casing and punctuation kept), absolute"
    note = "Stored per-run summaries; their answered subsets can differ. No paired verdict."
    if not rows:
        return f"{heading}\n\nNot available: both summaries need a formatted value.\n"
    return "\n".join(
        [heading, "", note, "", "| panel | base | cand |", "| --- | --- | --- |", *rows]
    )


def render(baseline: Run, candidates: list[tuple[str, Run]], panels_dir: Path, data_dir) -> str:
    """Reuse the official renderer so coverage, exclusions and cautions stay visible."""
    lines = [
        "# Source screening",
        "",
        f"Baseline: {baseline.path}",
        "",
        "Each candidate is paired separately: the baseline value is per row, not shared.",
        "The official comparison uses its frozen settings; no aggregate ranking is made.",
        "",
    ]
    for label, candidate in candidates:
        comparison = compare_runs(baseline.path, candidate.path, panels_dir, data_dir)
        lines += [
            f"## {label}",
            "",
            f"Run: {candidate.path}",
            "",
            render_comparison(comparison).replace("# Comparison:", "### Comparison:", 1),
        ]
        lines += [
            f"- {panel}: {section['items']} ids compared in {section['groups']} groups."
            for panel, section in comparison["panels"].items()
        ]
        lines += ["", formatted_table(baseline, candidate), ""]
    return "\n".join(lines)


def parse_candidate(text: str) -> tuple[str, Path]:
    """Require name=path, without default sources or guessed model names."""
    label, separator, path = text.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError(f"{text!r} is not name=path")
    return label.strip(), Path(path.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report finished source-screening runs with the official paired comparison.",
        epilog=(
            "example:\n"
            "  uv run python experiments/source-screening/table.py \\\n"
            "      --baseline /path/to/base/full \\\n"
            "      --candidate source=/path/to/screen-source/full\n"
            "No model, audio or network request is used."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--baseline", required=True, type=Path, metavar="PATH")
    parser.add_argument(
        "--candidate", required=True, action="append", type=parse_candidate, metavar="NAME=PATH"
    )
    parser.add_argument("--panels-dir", type=Path, default=Path("configs/panels"))
    parser.add_argument("--data-dir", type=Path, help="dataset manifests (default: $AUDIOLAB_DATA)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        labels = [label for label, _ in args.candidate]
        if len(set(labels)) != len(labels):
            raise Problem("candidate labels must be unique")
        baseline = Run(args.baseline, args.panels_dir)
        candidates = [(label, Run(path, args.panels_dir)) for label, path in args.candidate]
        output = render(baseline, candidates, args.panels_dir, args.data_dir)
    except (OSError, ValueError, KeyError, TypeError) as failure:
        print(f"table.py: {failure}", file=sys.stderr)
        return 2
    print(output, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
