"""The ``audiolab`` command.

Each area adds its subcommands here. A subcommand is a function that receives the
parsed arguments and returns an exit code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from audiolab import __version__
from audiolab.core.gpu import list_gpus
from audiolab.core.paths import data_dir, runs_dir
from audiolab.data.datasets import DATASETS
from audiolab.data.fetch import fetch_dataset
from audiolab.data.panels import (
    DEFAULT_PANELS_DIR,
    PANEL_NAMES,
    build_panels,
    check_panel,
    load_dataset_clips,
    panel_path,
)
from audiolab.eval.client import ConfigError, ModelConfig
from audiolab.eval.report import (
    REPORT_NAME,
    SUMMARY_NAME,
    compare_runs,
    render_comparison,
    score_run,
)
from audiolab.eval.runner import RunContext, RunError, load_suite, run_suite


def cmd_gpus(args: argparse.Namespace) -> int:
    gpus = list_gpus()
    if not gpus:
        print("No NVIDIA GPU found.")
        return 1
    print(f"{'index':<6} {'memory':>9}  {'name':<28} uuid")
    for gpu in gpus:
        memory = f"{gpu.memory_mib / 1024:.0f} GiB"
        print(f"{gpu.index:<6} {memory:>9} {gpu.name:<28} {gpu.uuid}")
    return 0


def cmd_data_list(args: argparse.Namespace) -> int:
    """Show the datasets this version can prepare and where they would land."""
    print(f"{'name':<18} {'license':<16} split")
    for name in sorted(DATASETS):
        info = DATASETS[name]
        print(f"{name:<18} {info.license:<16} {info.split}")
    print(f"\ndata directory: {data_dir()}")
    return 0


def cmd_data_fetch(args: argparse.Namespace) -> int:
    """Download and prepare one dataset or every dataset the spec lists."""
    names = args.names or sorted(DATASETS)
    for name in names:
        print(f"fetching {name} into {data_dir() / name}")
        fetch_dataset(name, force=args.force)
    return 0


def cmd_data_panels(args: argparse.Namespace) -> int:
    """Build or check the frozen panels of one dataset."""
    if args.check:
        return _check_panels(args.dataset, args.panels_dir)
    clips = load_dataset_clips(args.dataset)
    built = build_panels(args.dataset, clips, args.panels_dir)
    for panel in built:
        print(f"{panel_path(panel.title, args.panels_dir)}: {len(panel.ids)} ids")
    return 0


def _check_panels(dataset: str, panels_dir: Path) -> int:
    """Verify existing panels against the prepared data; report and exit 1 on problems."""
    clips = load_dataset_clips(dataset)
    failed = False
    for name in PANEL_NAMES:
        path = panel_path(f"{dataset}-{name}", panels_dir)
        if not path.is_file():
            print(f"{path}: missing", file=sys.stderr)
            failed = True
            continue
        problems = check_panel(path, clips, data_dir() / dataset)
        if problems:
            print(f"{path}: {'; '.join(problems)}", file=sys.stderr)
            failed = True
        else:
            print(f"{path}: ok")
    return 1 if failed else 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Run one model over one suite and score it."""
    model = ModelConfig.from_file(args.model)
    suite = load_suite(args.suite)
    context = RunContext(
        model=model,
        suite=suite,
        panels_dir=args.panels_dir,
        data_dir=args.data_dir,
        runs_root=args.runs_dir or runs_dir(),
        base_url=args.base_url or "",
        concurrency=args.concurrency,
        sealed=args.sealed,
        fresh=args.fresh,
    )
    run_dir = run_suite(context)
    print(f"summary: {run_dir / 'summary.json'}")
    print(f"report: {run_dir / REPORT_NAME}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score a finished run again, from its records; the same records give the same files."""
    run_dir = Path(args.run)
    summary = score_run(run_dir, args.panels_dir, args.data_dir)
    total = sum(panel["items"] for panel in summary["panels"].values())
    print(f"scored {total} items over {len(summary['panels'])} panels")
    print(f"summary: {run_dir / SUMMARY_NAME}")
    print(f"report: {run_dir / REPORT_NAME}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two runs over the ids and panels they share."""
    comparison = compare_runs(args.run_a, args.run_b, args.panels_dir, args.data_dir)
    print(render_comparison(comparison))
    if args.json:
        print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def _add_data_commands(commands: argparse._SubParsersAction) -> None:
    data = commands.add_parser("data", help="prepare datasets and freeze panels")
    data_commands = data.add_subparsers(title="data commands", metavar="<command>", required=True)

    listing = data_commands.add_parser("list", help="list the datasets that can be prepared")
    listing.set_defaults(func=cmd_data_list)

    fetch = data_commands.add_parser(
        "fetch", help="download and prepare datasets into $AUDIOLAB_DATA"
    )
    fetch.add_argument("names", nargs="*", help="dataset names; all of them when omitted")
    fetch.add_argument("--force", action="store_true", help="prepare again even if already present")
    fetch.set_defaults(func=cmd_data_fetch)

    panels = data_commands.add_parser("panels", help="build the frozen fast/full/sealed panels")
    panels.add_argument("dataset", help="prepared dataset name, for example fleurs-fr")
    panels.add_argument(
        "--check", action="store_true", help="verify existing panels instead of building"
    )
    panels.add_argument("--panels-dir", default=DEFAULT_PANELS_DIR, help="where panels are written")
    panels.set_defaults(func=cmd_data_panels)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiolab",
        description="A lab for speech experiments.",
    )
    parser.add_argument("--version", action="version", version=f"audiolab {__version__}")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="prepared datasets for this command; overrides $AUDIOLAB_DATA (default ./data)",
    )
    commands = parser.add_subparsers(title="commands", metavar="<command>")

    gpus = commands.add_parser("gpus", help="list the GPUs that --gpu can select")
    gpus.set_defaults(func=cmd_gpus)

    _add_data_commands(commands)
    _add_eval_commands(commands)

    return parser


def _add_eval_commands(commands: argparse._SubParsersAction) -> None:
    """``eval``, ``score`` and ``compare``: one run, its numbers, and two runs side by side."""
    scoring = argparse.ArgumentParser(add_help=False)
    scoring.add_argument(
        "--panels-dir", default=DEFAULT_PANELS_DIR, help="directory holding the frozen panels"
    )
    scoring.add_argument(
        "--runs-dir", default=None, help="root for run outputs (default: $AUDIOLAB_RUNS or ./runs)"
    )

    evaluate = commands.add_parser(
        "eval",
        help="transcribe the panels of a suite with one model and score it",
        parents=[scoring],
    )
    evaluate.add_argument(
        "--model", required=True, help="model config, for example configs/models/x.yaml"
    )
    evaluate.add_argument(
        "--suite", required=True, help="suite config, for example configs/suites/fast.yaml"
    )
    evaluate.add_argument(
        "--concurrency", type=int, default=1, help="requests in flight (default: 1)"
    )
    evaluate.add_argument(
        "--sealed", action="store_true", help="allow the sealed panel (one-time act)"
    )
    evaluate.add_argument("--fresh", action="store_true", help="start a new run directory")
    evaluate.add_argument(
        "--base-url", default="", help="endpoint URL; normally from the config's env var"
    )
    evaluate.set_defaults(func=cmd_eval)

    score = commands.add_parser("score", help="score a finished run again", parents=[scoring])
    score.add_argument("run", help="run directory holding records.jsonl")
    score.set_defaults(func=cmd_score)

    compare = commands.add_parser(
        "compare", help="compare two runs on the panels they share", parents=[scoring]
    )
    compare.add_argument("run_a", help="run directory, the baseline (A)")
    compare.add_argument("run_b", help="run directory, the challenger (B)")
    compare.add_argument("--json", action="store_true", help="also print the comparison as JSON")
    compare.set_defaults(func=cmd_compare)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    if args.data_dir:
        # One place decides where data lives, so every command agrees (paths.data_dir reads this).
        os.environ["AUDIOLAB_DATA"] = args.data_dir
    try:
        return args.func(args)
    except (ConfigError, RunError, ValueError, FileNotFoundError, FileExistsError) as failure:
        print(f"{parser.prog}: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
