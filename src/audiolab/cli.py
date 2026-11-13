"""The ``audiolab`` command.

Each area adds its subcommands here. A subcommand is a function that receives the
parsed arguments and returns an exit code.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from audiolab import __version__
from audiolab.core.gpu import add_gpu_argument, list_gpus
from audiolab.core.paths import cache_dir, data_dir, runs_dir
from audiolab.core.stop import EXIT_STOPPED, Stopped, install_signal_handlers
from audiolab.data.datasets import DATASETS
from audiolab.data.fetch import fetch_dataset, sized_dataset_name, validate_clips_cap
from audiolab.data.panels import (
    DEFAULT_PANELS_DIR,
    PANEL_NAMES,
    build_panels,
    check_panel,
    load_dataset_clips,
    load_suite,
    panel_path,
)
from audiolab.data.talk_listing import Talk, write_listing
from audiolab.data.talk_sources import SOURCES, list_talks, prepare_talk_audio
from audiolab.data.talks import HOURS_PER_CAP, http_client
from audiolab.data.train import TRAIN_CLIP_CAP
from audiolab.eval.client import ConfigError, ModelConfig, NotAnswering, concurrency_from_env
from audiolab.eval.report import (
    REPORT_NAME,
    SUMMARY_NAME,
    compare_runs,
    render_comparison,
    score_run,
)
from audiolab.eval.runner import RunContext, RunError, run_suite
from audiolab.serve.runtime import run_foreground
from audiolab.train.config import TrainConfig
from audiolab.train.runner import run_training


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
    if args.clips is not None and not args.names:
        raise ValueError("--clips sizes one dataset; give its name")
    for name in names:
        target = sized_dataset_name(name, args.clips) if args.clips is not None else name
        print(f"fetching {name} into {data_dir() / target}")
        fetch_dataset(name, force=args.force, clips=args.clips)
    return 0


def cmd_data_list_talks(args: argparse.Namespace) -> int:
    """Read one talk source's catalogue and pin it as its listing."""
    name = args.name
    with http_client() as http:
        talks = list_talks(name, http)
    source = SOURCES[name]
    write_listing(source, listed_on=date.today().isoformat(), talks=talks)
    print(f"{source.listing_path}: {len(talks)} talks, {hours_of(talks):.1f} h")
    return 0


def cmd_data_fetch_audio(args: argparse.Namespace) -> int:
    """Download the audio of the first talks of a talk source, without labelling them."""
    name = args.name
    if args.clips is not None:
        validate_clips_cap(args.clips)
        hours = HOURS_PER_CAP * args.clips / TRAIN_CLIP_CAP
    else:
        hours = args.hours if args.hours is not None else HOURS_PER_CAP
    print(f"preparing the audio of {name} into {cache_dir() / name}")
    prepared = prepare_talk_audio(name, cache_dir(), hours)
    return 0 if prepared.talks else 1


def at_least_one(text: str) -> int:
    """A count of at least one, for a command-line option."""
    if not text.isdecimal() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number of at least 1")
    return int(text)


def clips_cap(text: str) -> int:
    """A ``--clips`` value: a whole number of clips; its cap is checked in fetch."""
    if not text.isdecimal() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number of clips")
    return int(text)


def hours_of(talks: list[Talk]) -> float:
    """How long the talks of a listing last together."""
    return sum(talk.duration_s for talk in talks) / 3600


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
        concurrency=args.concurrency or concurrency_from_env(model),
        sealed=args.sealed,
        fresh=args.fresh,
        gpu=args.gpu,
    )
    run_dir = run_suite(context)
    print(f"summary: {run_dir / 'summary.json'}")
    print(f"report: {run_dir / REPORT_NAME}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Serve one local checkpoint through the same API a remote server speaks."""
    model = ModelConfig.from_file(args.model)
    if model.serve is None:
        raise ConfigError(
            f"{args.model} has no 'serve' block: this command serves a local checkpoint, "
            "and a config that names a server is evaluated with 'audiolab eval --base-url'"
        )
    spec = model.serve.with_gpu(args.gpu)
    print(f"serving {model.model} from {spec.resolved()}")
    return run_foreground(
        spec, model.model, host=args.host, port=args.port, stop_when_stdin_closes=True
    )


def cmd_train(args: argparse.Namespace) -> int:
    """Fine-tune one model on our own data, from one config file."""
    config = TrainConfig.from_file(args.config)
    directory = run_training(
        config,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        runs_root=Path(args.runs_dir) if args.runs_dir else None,
        fresh=args.fresh,
    )
    print(f"checkpoint: {directory / 'model.nemo'}")
    print(f"record: {directory / 'train.json'}")
    print(
        f"eval it with: audiolab eval --model {directory / 'model.yaml'} "
        "--suite configs/suites/fast.yaml"
    )
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


def _command(
    commands: argparse._SubParsersAction,
    name: str,
    summary: str,
    detail: str,
    example: str,
    parents: list[argparse.ArgumentParser] | None = None,
):
    """A subcommand that explains itself: one line in the list, a full text on --help."""
    return commands.add_parser(
        name,
        help=summary,
        description=f"{summary}. {detail}",
        epilog=f"example:\n  {example}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=parents or [],
    )


def _add_data_commands(commands: argparse._SubParsersAction) -> None:
    data = _command(
        commands,
        "data",
        "prepare the datasets and freeze the panels",
        "Nothing is scored here; this only fills $AUDIOLAB_DATA.",
        "audiolab data fetch fleurs-fr",
    )
    data_commands = data.add_subparsers(title="data commands", metavar="<command>", required=True)

    listing = _command(
        data_commands,
        "list",
        "list the datasets this version can prepare",
        "Names are what the other data commands take.",
        "audiolab data list",
    )
    listing.set_defaults(func=cmd_data_list)

    fetch = _command(
        data_commands,
        "fetch",
        "download and prepare datasets into $AUDIOLAB_DATA",
        "Downloads land in $AUDIOLAB_CACHE and are verified against a pinned checksum.",
        "audiolab data fetch fleurs-fr fleurs-en",
    )
    fetch.add_argument("names", nargs="*", help="dataset names; all of them when omitted")
    fetch.add_argument("--force", action="store_true", help="prepare again even if already present")
    fetch.add_argument(
        "--clips",
        type=clips_cap,
        default=None,
        help="prepare a training dataset with a larger cap, a multiple of 1 000 of at "
        "least 13 000: the set lands in a new directory named after the cap "
        "(granary-fr-train --clips 40000 -> granary-fr-train-40k)",
    )
    fetch.set_defaults(func=cmd_data_fetch)

    listing_talks = _command(
        data_commands,
        "list-talks",
        "pin the talks of one talk source from its live catalogue",
        "Asks a PeerTube instance, or a conference's file tree, for the talks the "
        "source may use and writes the pinned listing its preparation reads "
        "(metadata only). Re-run it to refresh a source; a change to a listing is a "
        "change to the dataset and is committed.",
        "audiolab data list-talks debconf-en-train",
    )
    listing_talks.add_argument("name", choices=sorted(SOURCES), help="talk dataset name")
    listing_talks.set_defaults(func=cmd_data_list_talks)

    fetch_audio = _command(
        data_commands,
        "fetch-audio",
        "download the audio of a talk source without labelling it",
        "Downloads the talks a labelling run reads first, into the cache the run "
        "reads them from. No model is asked anything, so this can run while the "
        "speech server is busy with another dataset (D-020).",
        "audiolab data fetch-audio debconf-en-train --hours 65",
    )
    fetch_audio.add_argument("name", choices=sorted(SOURCES), help="talk dataset name")
    hours_or_clips = fetch_audio.add_mutually_exclusive_group()
    hours_or_clips.add_argument(
        "--hours",
        type=float,
        default=None,
        help=f"how much talk to prepare (default {HOURS_PER_CAP:g} h, which is what "
        "the 12 000-clip cap needs at the measured yield)",
    )
    hours_or_clips.add_argument(
        "--clips",
        type=clips_cap,
        default=None,
        help="plan the talk of a larger cap, a multiple of 1 000 of at least 13 000, "
        "at the measured yield of the 12 000-clip cap",
    )
    fetch_audio.set_defaults(func=cmd_data_fetch_audio)

    panels = _command(
        data_commands,
        "panels",
        "build the frozen fast/full/sealed panels",
        "A panel is a list of clip ids with a hash of the manifest it was cut from; "
        "building it again from the same dataset gives the same file.",
        "audiolab data panels fleurs-fr && audiolab data panels fleurs-fr --check",
    )
    panels.add_argument("dataset", help="prepared dataset name, for example fleurs-fr")
    panels.add_argument(
        "--check", action="store_true", help="verify existing panels instead of building"
    )
    panels.add_argument("--panels-dir", default=DEFAULT_PANELS_DIR, help="where panels are written")
    panels.set_defaults(func=cmd_data_panels)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiolab",
        description="Run and score speech-to-text models on frozen panels.",
        epilog=(
            "example:\n"
            "  audiolab eval --model configs/models/example.yaml"
            " --suite configs/suites/fast.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"audiolab {__version__}")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="prepared datasets for this command; overrides $AUDIOLAB_DATA (default ./data)",
    )
    commands = parser.add_subparsers(title="commands", metavar="<command>")

    gpus = _command(
        commands,
        "gpus",
        "list the GPUs a later phase can select",
        "Needs nvidia-smi; without a GPU this prints one line and exits 1.",
        "audiolab gpus",
    )
    gpus.set_defaults(func=cmd_gpus)

    _add_data_commands(commands)
    _add_eval_commands(commands)
    _add_serve_command(commands)
    _add_train_command(commands)

    return parser


def _add_eval_commands(commands: argparse._SubParsersAction) -> None:
    """``eval``, ``score`` and ``compare``: one run, its numbers, and two runs side by side."""
    scoring = argparse.ArgumentParser(add_help=False)
    scoring.add_argument(
        "--panels-dir", default=DEFAULT_PANELS_DIR, help="directory holding the frozen panels"
    )
    _add_eval_command(commands, scoring)
    _add_score_command(commands, scoring)
    _add_compare_command(commands, scoring)


def _add_eval_command(
    commands: argparse._SubParsersAction, scoring: argparse.ArgumentParser
) -> None:
    """``eval``: transcribe a suite with one model, then score it."""
    evaluate = _command(
        commands,
        "eval",
        "transcribe the panels of a suite with one model and score it",
        "Only clips with no cached answer are sent; everything else is re-scored.",
        "audiolab eval --model configs/models/example.yaml --suite configs/suites/fast.yaml",
        parents=[scoring],
    )
    evaluate.add_argument(
        "--runs-dir",
        default=None,
        help="root for run outputs (default: $AUDIOLAB_RUNS or ./runs)",
    )
    evaluate.add_argument(
        "--model", required=True, help="model config, for example configs/models/x.yaml"
    )
    evaluate.add_argument(
        "--suite", required=True, help="suite config, for example configs/suites/fast.yaml"
    )
    evaluate.add_argument(
        "--concurrency",
        type=at_least_one,
        default=None,
        help="requests in flight (default: the model's entry in $AUDIOLAB_CONCURRENCY, else 1)",
    )
    evaluate.add_argument(
        "--sealed", action="store_true", help="allow the sealed panel (one-time act)"
    )
    evaluate.add_argument(
        "--fresh", action="store_true", help="send every clip again, even those already answered"
    )
    evaluate.add_argument(
        "--base-url", default="", help="endpoint URL; normally from the config's env var"
    )
    # For a model config with a 'serve' block, which GPU serves it; the config's
    # own choice stands unless this is given.
    add_gpu_argument(evaluate, default=None)
    evaluate.set_defaults(func=cmd_eval)


def _add_score_command(
    commands: argparse._SubParsersAction, scoring: argparse.ArgumentParser
) -> None:
    """``score``: recompute a finished run's numbers without sending anything."""
    score = _command(
        commands,
        "score",
        "score a finished run again",
        "Reads the run's records and writes the same files; use it after changing a metric.",
        "audiolab score $AUDIOLAB_RUNS/eval/example-model/fast",
        parents=[scoring],
    )
    score.add_argument("run", help="run directory holding records.jsonl")
    score.set_defaults(func=cmd_score)


def _add_compare_command(
    commands: argparse._SubParsersAction, scoring: argparse.ArgumentParser
) -> None:
    """``compare``: pair two runs over the ids they both answered."""
    compare = _command(
        commands,
        "compare",
        "compare two runs on the panels they share",
        "Pairs the ids both runs answered and reports the ones left out.",
        "audiolab compare RUN_A RUN_B --json",
        parents=[scoring],
    )
    compare.add_argument("run_a", help="run directory, the baseline (A)")
    compare.add_argument("run_b", help="run directory, the challenger (B)")
    compare.add_argument("--json", action="store_true", help="also print the comparison as JSON")
    compare.set_defaults(func=cmd_compare)


def _add_serve_command(commands: argparse._SubParsersAction) -> None:
    """``serve``: expose a local checkpoint, for evaluation or for another program."""
    serve = _command(
        commands,
        "serve",
        "serve one local checkpoint through an OpenAI-compatible API",
        "Reads the model config's 'serve' block, loads the checkpoint it names and "
        "answers requests itself until it is stopped. This is the server 'audiolab "
        "eval' starts for such a config; run it by hand to use the model from "
        "another program.",
        "audiolab serve --model configs/models/parakeet-tdt-0.6b-v3-nemo.yaml",
    )
    serve.add_argument("--model", required=True, help="model config with a 'serve' block")
    serve.add_argument("--host", default="127.0.0.1", help="address to listen on")
    serve.add_argument("--port", type=int, default=8000, help="port to listen on")
    add_gpu_argument(serve, default=None)
    serve.set_defaults(func=cmd_serve)


def _add_train_command(commands: argparse._SubParsersAction) -> None:
    """``train``: fine-tune one model on the clips a config asks for."""
    train = _command(
        commands,
        "train",
        "fine-tune one model on our own data",
        "Trains the checkpoint the config names on clips drawn from a prepared "
        "dataset, for the number of updates the config asks for, and writes the "
        "result into $AUDIOLAB_RUNS/train/<config name>/. A run that stopped is "
        "continued, not restarted; a finished run needs --fresh.",
        "audiolab train --config configs/train/example.yaml",
    )
    train.add_argument(
        "--config", required=True, help="training config, for example configs/train/example.yaml"
    )
    train.add_argument(
        "--runs-dir", default=None, help="root for run outputs (default: $AUDIOLAB_RUNS or ./runs)"
    )
    train.add_argument(
        "--fresh", action="store_true", help="start over, discarding a previous run of this config"
    )
    train.set_defaults(func=cmd_train)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    if args.data_dir:
        # One place decides where data lives, so every command agrees (paths.data_dir reads this).
        os.environ["AUDIOLAB_DATA"] = args.data_dir
    install_signal_handlers()
    try:
        return args.func(args)
    except Stopped as stopped:
        print(f"{parser.prog}: {stopped}", file=sys.stderr)
        return EXIT_STOPPED
    except (
        ConfigError,
        NotAnswering,
        RunError,
        ValueError,
        FileNotFoundError,
        FileExistsError,
    ) as failure:
        print(f"{parser.prog}: {failure}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
