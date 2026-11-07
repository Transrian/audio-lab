"""Run a suite against one model and keep every answer on disk.

Two rules shape this file. A run is resumable: results are appended and flushed
as they arrive, and rerunning skips what is already recorded. A run has an
identity: if the model config, the endpoint, the panels or the normalizer
changed, the old records no longer belong to the new run, so the runner refuses
to add to them instead of mixing two experiments in one directory. The endpoint
enters the identity as a keyed fingerprint, so the run directory never holds the
address of the server it talked to.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from audiolab import __version__
from audiolab.core.paths import runs_dir
from audiolab.data.panels import DEFAULT_PANELS_DIR
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
    ENDPOINT_FINGERPRINT_KEY,
    IDENTITY_KEY_NAME,
    IDENTITY_NAME,
    RECORDS_NAME,
    append_record,
    clip_paths,
    dataset_and_panel,
    git_commit,
    latest_records,
    load_panel_items,
    panel_hashes,
    read_records,
)
from audiolab.eval.report import score_run

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
    verbose: bool = True

    def __post_init__(self) -> None:
        """The command line passes directories as strings; paths are used here."""
        self.panels_dir = Path(self.panels_dir)
        self.data_dir = Path(self.data_dir) if self.data_dir else None
        self.runs_root = Path(self.runs_root)


def run_suite(context: RunContext, *, client_factory=None) -> Path:
    """Score every panel of the suite, then write the summary; returns the run directory."""
    _refuse_sealed_without_flag(context)
    base_url = context.base_url or base_url_from_env(context.model)
    items_by_panel = {
        panel: load_panel_items(panel, context.panels_dir, context.data_dir)
        for panel in context.suite.panels
    }
    identity = _identity(context, list(items_by_panel), base_url)
    run_dir = _prepare_run_dir(context, identity)
    records_path = run_dir / RECORDS_NAME
    pending = _pending_items(items_by_panel, read_records(records_path))

    if pending:
        factory = client_factory or _default_client
        _transcribe_pending(context, pending, records_path, base_url, factory)
    failed = _announce(context, records_path)
    score_run(run_dir, context.panels_dir, context.data_dir)
    if failed:
        # The report exists and counts the failures; the non-zero exit is what
        # tells a caller that the numbers are not a full picture.
        raise RunError(
            f"{failed} of {len(read_records(records_path))} requests failed; see {records_path}"
        )
    return run_dir


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


def _identity(context: RunContext, panels: list[str], base_url: str) -> dict:
    """What a resumable run is: the model, the endpoint, the panels, the code.

    The endpoint is part of the identity, because the same model served from two
    places is two experiments; it is recorded as a keyed fingerprint, because a
    run directory can be committed or shared and a URL names a host. The
    ``server_label`` of the model config says in words which server this was.
    """
    return {
        "model": context.model.identity(),
        "panels": panel_hashes(panels, context.panels_dir),
        "suite": context.suite.name,
        "audiolab_version": __version__,
        "normalizer_version": NORMALIZER_VERSION,
        "git_commit": git_commit(),
        ENDPOINT_FINGERPRINT_KEY: endpoint_fingerprint(
            base_url, identity_key(Path(context.runs_root))
        ),
    }


def endpoint_fingerprint(base_url: str, key: bytes) -> str:
    """A keyed fingerprint of the server a run talked to.

    A plain hash of a URL is a guess away from being reversed: a private address
    is one of a handful of shapes, so the fingerprint would leak what it hides.
    The key never leaves the machine and the fingerprint is only ever compared,
    never printed, which makes it useless to whoever reads a shared run.
    """
    return hmac.new(key, server_root(base_url).encode("utf-8"), hashlib.sha256).hexdigest()


def identity_key(runs_root: Path) -> bytes:
    """The per-machine key of the identity fingerprint, created on first use.

    It is written with mode 0600 and never leaves the runs directory: it is not
    a secret the project manages, only something that makes the fingerprints in
    one machine's run directories meaningless anywhere else.
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


def _prepare_run_dir(context: RunContext, identity: dict) -> Path:
    """Find the run directory: resume the matching one, refuse a mismatching one."""
    root = Path(context.runs_root) / "eval" / context.model.name / context.suite.name
    identity_path = root / IDENTITY_NAME
    if not context.fresh and identity_path.is_file():
        recorded = json.loads(identity_path.read_text(encoding="utf-8"))
        if recorded == identity:
            return root
        raise RunError(
            f"{root} holds a different run: {_identity_differences(recorded, identity)}. "
            "Pass --fresh to start a new directory, or use a different --runs dir."
        )
    run_dir = _timestamped(root) if context.fresh and root.exists() else root
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / IDENTITY_NAME).write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dir


def _identity_differences(recorded: dict, current: dict) -> str:
    """Which parts of the identity changed, so the message says what to fix.

    Nested parts are named by their path: "model changed ({...})" would leave the
    reader to diff two JSON blobs, while "model.model changed (served-id ->
    other-id)" is the answer. Values are shown except for the keyed fingerprint,
    which is only ever compared.
    """
    before, after = _flatten_identity(recorded), _flatten_identity(current)
    changed = []
    for path in sorted(set(before) | set(after)):
        if before.get(path) == after.get(path):
            continue
        if path == ENDPOINT_FINGERPRINT_KEY:
            changed.append(f"{path} changed (the server URL is not the one recorded)")
            continue
        was, now = _short_value(before.get(path)), _short_value(after.get(path))
        changed.append(f"{path} changed ({was} -> {now})")
    return "; ".join(changed) if changed else "the recorded identity is not the same one"


def _flatten_identity(identity: dict, prefix: str = "") -> dict:
    """One level per dotted path, so a difference names the field that moved."""
    flat: dict = {}
    for key, value in identity.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten_identity(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def _short_value(value, limit: int = 40) -> str:
    """A value as the error message shows it: short, and never a long JSON blob."""
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _timestamped(root: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root.parent / f"{root.name}-{stamp}"


def _pending_items(items_by_panel: dict[str, list[Item]], records: list[dict]) -> list[Item]:
    """The items to send: those with no record yet, plus those that failed.

    A failed request is not a measurement: its record keeps the failure visible
    (the summary counts failed requests separately) and the item is sent again
    on the next run, the newest record winning.
    """
    last = latest_records(records)
    return [
        item
        for items in items_by_panel.values()
        for item in items
        if item.id not in last or last[item.id].get("error")
    ]


def _transcribe_pending(
    context: RunContext, pending: list[Item], records_path: Path, base_url: str, factory
) -> None:
    """Send every pending item, appending each answer as soon as it arrives.

    One client serves the whole run: it holds the connection pool, and creating
    one per item would open a connection per request and never close it.
    """
    lock = threading.Lock()
    paths = {item.panel: clip_paths(item.panel, context.data_dir) for item in pending}
    refused = False

    def work(index_and_item: tuple[int, Item]) -> None:
        nonlocal refused
        index, item = index_and_item
        record = _transcribe_one(client, item, paths[item.panel][item.id], cold=index == 0)
        with lock:
            append_record(records_path, record)
            if client.language_refused and not refused:
                refused = True
                if context.verbose:
                    print(
                        "warning: the server refused the language field; "
                        "it is not sent again for the rest of this run"
                    )
            if context.verbose:
                _report_progress(index + 1, len(pending), record)

    work_items = list(enumerate(pending))
    with closing(factory(context, base_url)) as client:
        if context.concurrency > 1:
            with ThreadPoolExecutor(max_workers=context.concurrency) as pool:
                list(pool.map(work, work_items))
        else:
            for entry in work_items:
                work(entry)


def _transcribe_one(client, item: Item, audio_file: Path, *, cold: bool) -> dict:
    """One request, one record: a failure is an error record, never empty text."""
    language = "" if item.language == "none" else item.language
    try:
        result = client.transcribe(audio_file.read_bytes(), language)
    except TranscriptionError as failure:
        return {
            "id": item.id,
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


def _announce(context: RunContext, records_path: Path) -> int:
    """Say where the run lives and how it went; returns how many requests failed."""
    records = read_records(records_path)
    latest = latest_records(records)
    failed = [record for record in latest.values() if record["error"]]
    if context.verbose:
        retried = len(records) - len(latest)
        extra = f", {retried} earlier records kept from retries" if retried else ""
        print(f"records: {records_path} ({len(latest)} items, {len(failed)} failed{extra})")
        if failed:
            print(f"first failure: {failed[0]['id']}: {failed[0]['error']}")
    return len(failed)
