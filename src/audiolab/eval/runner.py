"""Run a suite against one model: ask what is not cached, then score.

Answers live in the transcript cache (``eval.cache``), one file per way of
asking a model, shared by every suite. A run sends only the clips the cache has
no good answer for, appending each answer as it arrives, so an interrupted run
resumes where it stopped and a new dataset costs only its own clips. The run
directory of a suite is then rebuilt from the cache: its records, its identity
and its scores always describe the current panels, never a mix.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from audiolab import __version__
from audiolab.core.parallel import map_requests
from audiolab.core.paths import runs_dir
from audiolab.core.stop import raise_if_stopped
from audiolab.data.panels import DEFAULT_PANELS_DIR
from audiolab.eval.cache import (
    answer_key,
    cache_file,
    cached_answers,
    open_cache,
    transcript_key,
)
from audiolab.eval.client import (
    ModelConfig,
    TranscriptionClient,
    TranscriptionError,
    api_key_from_env,
    base_url_from_env,
    server_root,
)
from audiolab.eval.metrics import Item
from audiolab.eval.normalize import NORMALIZER_VERSION
from audiolab.eval.records import (
    IDENTITY_KEY_NAME,
    IDENTITY_NAME,
    RECORDS_NAME,
    append_record,
    clip_paths,
    dataset_and_panel,
    git_commit,
    load_panel_items,
    panel_hashes,
    read_records,
    write_records,
)
from audiolab.eval.report import score_run
from audiolab.serve.config import ServeSpec
from audiolab.serve.launcher import reported_identity, serving

DEFAULT_CONCURRENCY = 1
SEALED_WARNING = (
    "scoring the sealed panel is a one-time act reserved to the owner: "
    "once it is seen, it can no longer measure anything"
)


class RunError(RuntimeError):
    """A run that cannot start or continue as asked."""


@dataclass(frozen=True)
class SuiteSpec:
    """The panels a suite scores."""

    name: str
    panels: list[str]


def load_suite(path: str | Path) -> SuiteSpec:
    """Read ``configs/suites/<name>.yaml``."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("panels"), list) or not raw["panels"]:
        raise RunError(f"{path} must hold a non-empty 'panels' list")
    name = str(raw.get("name") or path.stem)
    return SuiteSpec(name=name, panels=[str(panel) for panel in raw["panels"]])


@dataclass
class RunContext:
    """Everything one run needs, gathered in one place."""

    model: ModelConfig
    suite: SuiteSpec
    panels_dir: str | Path = DEFAULT_PANELS_DIR
    data_dir: str | Path | None = None
    runs_root: str | Path = field(default_factory=runs_dir)
    base_url: str = ""
    api_key: str = ""
    concurrency: int = DEFAULT_CONCURRENCY
    sealed: bool = False
    fresh: bool = False
    # The GPU a locally served model uses; ``None`` keeps the config's choice.
    gpu: str | None = None
    verbose: bool = True

    def __post_init__(self) -> None:
        """The command line passes directories as strings; paths are used here."""
        self.panels_dir = Path(self.panels_dir)
        self.data_dir = Path(self.data_dir) if self.data_dir else None
        self.runs_root = Path(self.runs_root)

    def serving_spec(self) -> ServeSpec | None:
        """The serve block to run here, or ``None`` when a server already answers.

        ``base_url`` wins: pointing the command at a server that is already
        running (the one ``audiolab serve`` shows) must not start a second one.
        """
        if self.model.serve is None or self.base_url:
            return None
        return self.model.serve.with_gpu(self.gpu)


def run_suite(context: RunContext, *, client_factory=None) -> Path:
    """Answer and score every panel of the suite; returns the run directory.

    A model config with a ``serve`` block makes this start the server itself and
    stop it afterwards, so evaluating a local checkpoint is one command
    (D-016).
    """
    _refuse_sealed_without_flag(context)
    spec = context.serving_spec()
    if spec is None:
        # An endpoint the run was pointed at, not a checkpoint it serves: what its
        # answers belong to is the address, and the config's ``serve`` block, if it
        # has one, describes a server this run never started.
        return _answer_and_score(
            context, context.base_url or base_url_from_env(context.model), client_factory
        )
    with serving(spec, context.model.model) as base_url:
        _refuse_foreign_server(base_url, spec)
        return _answer_and_score(context, base_url, client_factory, served=spec)


def _refuse_foreign_server(base_url: str, spec: ServeSpec) -> None:
    """Stop before an answer is recorded if the server is not the checkpoint asked for.

    The server is asked what it holds rather than believed. A stale port, a child
    that loaded other weights, or a server that says nothing would all fill the
    cache with answers attributed to the wrong checkpoint, and the cache is what
    every later run trusts (D-017).
    """
    wanted = spec.identity()
    reported = reported_identity(base_url)
    if reported != wanted:
        raise RunError(
            f"the server started at {base_url} reports {_identity_words(reported)}, not "
            f"{_identity_words(wanted)} as this config asks; stopping instead of recording "
            "answers as another checkpoint's"
        )


def _identity_words(identity: dict | None) -> str:
    """One identity, in a message: what it is, or that there is none."""
    return "no identity" if identity is None else json.dumps(identity, sort_keys=True)


def _answer_and_score(
    context: RunContext, base_url: str, client_factory=None, *, served: ServeSpec | None = None
) -> Path:
    """Send what is not cached, then rebuild and score the run directory."""
    items_by_panel = {
        panel: load_panel_items(panel, context.panels_dir, context.data_dir)
        for panel in context.suite.panels
    }
    key_bytes = identity_key(Path(context.runs_root))
    fingerprint = identity_fingerprint(_endpoint_identity(served, base_url), key_bytes)
    key = transcript_key(context.model, fingerprint)
    cache_path = cache_file(Path(context.runs_root), key)
    open_cache(cache_path, key)
    pending = _pending_items(items_by_panel, cached_answers(cache_path), context.fresh)

    if pending:
        factory = client_factory or _default_client
        _transcribe_pending(context, pending, cache_path, base_url, factory)
    run_dir = _write_run(context, items_by_panel, cache_path, served)
    failed = _announce(context, run_dir, cache_path, sent=len(pending))
    score_run(run_dir, context.panels_dir, context.data_dir)
    if failed:
        # The report exists and counts the failures; the non-zero exit is what
        # tells a caller that the numbers are not a full picture.
        raise RunError(
            f"{failed} panel items have no answer (their request failed); "
            f"they are sent again on the next run (see {run_dir / RECORDS_NAME})"
        )
    return run_dir


def _endpoint_identity(served: ServeSpec | None, base_url: str) -> str:
    """What the answers of a run belong to.

    For a server the run was pointed at, it is that server's root. For a
    checkpoint the run served itself it is the checkpoint - its content hash, or
    its Hub id - with the backend and the decoding settings, because the
    temporary server listens on a port that changes on every run, and a key that
    followed the port would ask the same checkpoint again for every clip of every
    run (D-017). ``served`` says which of the two this run is: a config may name
    a ``serve`` block and still be pointed somewhere else for one run.
    """
    if served is not None:
        return json.dumps(served.identity(), sort_keys=True)
    return server_root(base_url)


def _default_client(context: RunContext, base_url: str):
    return TranscriptionClient(
        context.model, base_url, context.api_key or api_key_from_env(context.model)
    )


def _refuse_sealed_without_flag(context: RunContext) -> None:
    """The sealed panel is a one-time measurement; it needs an explicit flag."""
    sealed_panels = [
        panel for panel in context.suite.panels if dataset_and_panel(panel)[1] == "sealed"
    ]
    if sealed_panels and not context.sealed:
        raise RunError(
            f"{', '.join(sealed_panels)} is the sealed panel; pass --sealed to score it. "
            f"{SEALED_WARNING}"
        )
    if sealed_panels:
        print(f"warning: {SEALED_WARNING}")


def _identity(
    context: RunContext,
    panels: list[str],
    cache_path: Path,
    served: ServeSpec | None = None,
) -> dict:
    """What a run directory describes: the model, the panels, the code, the cache.

    It is rewritten on every run, so it always matches the records next to it.
    The endpoint is not here: it is part of the cache key, under a keyed
    fingerprint that stays in the cache directory, because a run directory is
    meant to be shared and a URL names a host.
    """
    return {
        "model": context.model.identity(include_serve=served is not None),
        "panels": panel_hashes(panels, context.panels_dir),
        "suite": context.suite.name,
        "audiolab_version": __version__,
        "normalizer_version": NORMALIZER_VERSION,
        "git_commit": git_commit(),
        "transcripts": cache_path.name,
    }


def identity_fingerprint(identity: str, key: bytes) -> str:
    """A keyed fingerprint of what a run's answers belong to.

    A plain hash of an identity is a guess away from being reversed: a private
    address is one of a handful of shapes, so the fingerprint would leak what it
    hides. The key never leaves the machine and the fingerprint is only ever
    compared, never printed, which makes it useless to whoever reads a shared
    run.
    """
    return hmac.new(key, identity.encode("utf-8"), hashlib.sha256).hexdigest()


def identity_key(runs_root: Path) -> bytes:
    """The per-machine key of the identity fingerprint, created on first use.

    It is written with mode 0600 and never leaves the runs directory: it is not
    a secret the project manages, only something that makes the fingerprints in
    one machine's cache key files meaningless anywhere else.
    """
    path = runs_root / IDENTITY_KEY_NAME
    if path.is_file():
        return bytes.fromhex(path.read_text(encoding="utf-8").strip())
    key = secrets.token_bytes(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:  # another process created it between the two calls
        return bytes.fromhex(path.read_text(encoding="utf-8").strip())
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(key.hex() + "\n")
    return key


def _write_run(
    context: RunContext,
    items_by_panel: dict[str, list[Item]],
    cache_path: Path,
    served: ServeSpec | None = None,
) -> Path:
    """Rebuild the suite's run directory from the cache: identity and records.

    One record per panel item, the cached answer with the panel it is scored
    in. Rebuilding rather than appending is what keeps a directory from mixing
    two versions of a panel, or two servers.
    """
    run_dir = Path(context.runs_root) / "eval" / context.model.name / context.suite.name
    run_dir.mkdir(parents=True, exist_ok=True)
    answers = cached_answers(cache_path)
    records = [
        {**answers[_answer_key(item)], "panel": panel}
        for panel, items in items_by_panel.items()
        for item in items
    ]
    identity = _identity(context, list(items_by_panel), cache_path, served)
    _warn_if_replacing_another_run(context, run_dir, identity)
    (run_dir / IDENTITY_NAME).write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_records(run_dir / RECORDS_NAME, records)
    return run_dir


def _warn_if_replacing_another_run(context: RunContext, run_dir: Path, identity: dict) -> None:
    """Say so when the directory held answers from another cache file.

    That happens when the same config name now asks differently (another served
    id, request option or server). The answers are not lost, they stay in their
    cache file; only this view of them is replaced.
    """
    previous_path = run_dir / IDENTITY_NAME
    if not previous_path.is_file():
        return
    previous = json.loads(previous_path.read_text(encoding="utf-8")).get("transcripts")
    if previous and previous != identity["transcripts"] and context.verbose:
        print(
            f"note: {run_dir} showed the answers of {previous}; it now shows "
            f"{identity['transcripts']} (copy the directory first to keep both views)"
        )


def _answer_key(item: Item) -> tuple[str, str, str]:
    """The cache key of an item; no language is asked for non-speech clips."""
    return answer_key(item.id, item.audio_sha256, _language_asked(item))


def _language_asked(item: Item) -> str:
    return "" if item.language == "none" else item.language


def _pending_items(
    items_by_panel: dict[str, list[Item]], answers: dict[tuple[str, str, str], dict], fresh: bool
) -> list[Item]:
    """The clips to send: those with no cached answer, plus those that failed.

    A failed request is not a measurement: it stays in the cache file (the
    summary counts failed requests) and the clip is sent again, the newest
    answer winning. ``fresh`` sends every clip again. A clip in two panels of
    the suite is sent once.
    """
    pending: dict[tuple[str, str, str], Item] = {}
    for items in items_by_panel.values():
        for item in items:
            key = _answer_key(item)
            cached = answers.get(key)
            if fresh or cached is None or cached.get("error"):
                pending.setdefault(key, item)
    return list(pending.values())


def _transcribe_pending(
    context: RunContext, pending: list[Item], cache_path: Path, base_url: str, factory
) -> None:
    """Send every pending item, appending each answer to the cache as it arrives.

    A warmup waits until the model answers, then the items are sent
    ``context.concurrency`` at a time (D-022). A stop (D-021) lets the items in
    flight finish, sends no other, and raises :class:`~audiolab.core.stop.Stopped`
    before anything is written or scored. One client serves the whole run:
    it holds the connection pool, and creating one per item would open a
    connection per request and never close it.
    """
    lock = threading.Lock()
    paths = {item.panel: clip_paths(item.panel, context.data_dir) for item in pending}
    refused = False

    def work(index_and_item: tuple[int, Item]) -> None:
        nonlocal refused
        index, item = index_and_item
        record = _transcribe_one(client, item, paths[item.panel][item.id], cold=index == 0)
        with lock:
            append_record(cache_path, record)
            if client.language_refused and not refused:
                refused = True
                if context.verbose:
                    print(
                        "warning: the server refused the language field; "
                        "it is not sent again for the rest of this run"
                    )
            if context.verbose:
                _report_progress(index + 1, len(pending), record)

    with closing(factory(context, base_url)) as client:
        client.warm_up()
        map_requests(work, enumerate(pending), context.concurrency)
    # A stopped run sent only some of its items: it is not written as complete,
    # and the next one sends the rest, the cache holding what this one received.
    raise_if_stopped("evaluation stopped")


def _transcribe_one(client, item: Item, audio_file: Path, *, cold: bool) -> dict:
    """One request, one record: a failure is an error record, never empty text."""
    language = _language_asked(item)
    try:
        result = client.transcribe(audio_file.read_bytes(), language)
    except TranscriptionError as failure:
        return {
            "id": item.id,
            "audio_sha256": item.audio_sha256,
            "language": language,
            "panel": item.panel,
            "hypothesis": "",
            "error": str(failure),
            "latency_s": 0.0,
            "attempts": getattr(failure, "attempts", 0),
            "cold": cold,
            "language_sent": False,
            "timestamp": _now(),
        }
    return {
        "id": item.id,
        "audio_sha256": item.audio_sha256,
        "language": language,
        "panel": item.panel,
        "hypothesis": result.text,
        "error": "",
        "latency_s": round(result.latency_s, 6),
        "attempts": result.attempts,
        "cold": cold,
        "language_sent": result.language_sent,
        "timestamp": _now(),
    }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _report_progress(done: int, total: int, record: dict) -> None:
    state = "error" if record["error"] else "ok"
    print(f"[{done}/{total}] {record['id']} {state} ({record['latency_s']:.2f}s)", flush=True)


def _announce(context: RunContext, run_dir: Path, cache_path: Path, *, sent: int) -> int:
    """Say where the run lives and how it went; returns how many items have no answer."""
    records = read_records(run_dir / RECORDS_NAME)
    failed = [record for record in records if record["error"]]
    if context.verbose:
        print(
            f"records: {run_dir / RECORDS_NAME} ({len(records)} panel items, {len(failed)} failed)"
        )
        print(f"requests sent now: {sent}; transcript cache: {cache_path}")
        if failed:
            print(f"first failure: {failed[0]['id']}: {failed[0]['error']}")
    return len(failed)
