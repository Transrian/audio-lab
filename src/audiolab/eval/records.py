"""What a run is made of: panels in, records out.

A panel holds ids; the reference text and the audio path come from the manifest
of the dataset the panel belongs to. Keeping the two apart is what lets the
repository hold the panels without holding any dataset content (decision 0009).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from audiolab.core.paths import data_dir
from audiolab.data.panels import PANEL_NAMES, read_panel
from audiolab.data.store import manifest_path, read_manifest
from audiolab.eval.metrics import Item

RECORDS_NAME = "records.jsonl"
IDENTITY_NAME = "run.json"
# The run identity records the endpoint under a keyed fingerprint: a run
# directory can be shared, and the address of the server must not travel with
# it, while resuming against a different server must still be refused. The
# fingerprint is written to ``run.json`` only, never to the summary or the
# report, and the key that makes it unguessable stays next to the runs.
ENDPOINT_FINGERPRINT_KEY = "endpoint_fingerprint"
# The key that makes that fingerprint unforgeable and unguessable, next to the
# runs it protects; created with mode 0600 on first use.
IDENTITY_KEY_NAME = ".identity-key"

# Fields of one record; anything else is corruption rather than a new feature.
RECORD_FIELDS = (
    "id",
    "audio_sha256",
    "panel",
    "hypothesis",
    "error",
    "latency_s",
    "attempts",
    "cold",
    "language_sent",
    "timestamp",
)


def dataset_and_panel(panel_title: str) -> tuple[str, str]:
    """Split ``fleurs-fr-fast`` into the dataset ``fleurs-fr`` and the panel ``fast``."""
    dataset, _, name = panel_title.rpartition("-")
    if not dataset or name not in PANEL_NAMES:
        raise ValueError(f"{panel_title!r} is not <dataset>-<{'|'.join(PANEL_NAMES)}>")
    return dataset, name


def load_panel_items(
    panel_title: str, panels_dir: str | Path, data_dir_: str | Path | None = None
) -> list[Item]:
    """Every item of a panel, without any hypothesis yet."""
    dataset, _ = dataset_and_panel(panel_title)
    panel = read_panel(Path(panels_dir) / f"{panel_title}.txt")
    root = Path(data_dir_) if data_dir_ else data_dir()
    clips = {clip.id: clip for clip in read_manifest(root / dataset)}
    missing = [clip_id for clip_id in panel.ids if clip_id not in clips]
    if missing:
        raise FileNotFoundError(
            f"{panel_title}: {len(missing)} panel ids are not in {manifest_path(root / dataset)} "
            f"(first: {missing[0]}); re-fetch {dataset} or rebuild the panel"
        )
    return [
        Item(
            id=clip_id,
            panel=panel_title,
            group=clips[clip_id].group,
            category=clips[clip_id].category,
            language=clips[clip_id].language,
            reference=clips[clip_id].text,
            audio_sha256=clips[clip_id].sha256,
            duration_s=clips[clip_id].duration_s,
            text_has_casing_punct=clips[clip_id].text_has_casing_punct,
        )
        for clip_id in panel.ids
    ]


def clip_paths(panel_title: str, data_dir_: str | Path | None = None) -> dict[str, Path]:
    """Where the audio of each clip of a panel lives."""
    dataset, _ = dataset_and_panel(panel_title)
    root = Path(data_dir_) if data_dir_ else data_dir()
    return {clip.id: root / dataset / clip.audio for clip in read_manifest(root / dataset)}


def panel_hashes(panel_titles: list[str], panels_dir: str | Path) -> dict[str, str]:
    """sha256 of each panel file, so a resume can tell whether the ids changed."""
    panels_dir = Path(panels_dir)
    return {
        title: hashlib.sha256((panels_dir / f"{title}.txt").read_bytes()).hexdigest()
        for title in panel_titles
    }


def append_record(path: Path, record: dict) -> None:
    """Append one finished item and flush, so a crash keeps what was done."""
    unknown = set(record) - set(RECORD_FIELDS)
    if unknown:
        raise ValueError(f"record has unknown fields: {sorted(unknown)}")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def write_records(path: Path, records: list[dict]) -> None:
    """Replace a records file in one step, so a reader never sees half of it."""
    for record in records:
        unknown = set(record) - set(RECORD_FIELDS)
        if unknown:
            raise ValueError(f"record has unknown fields: {sorted(unknown)}")
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    partial = path.with_suffix(".partial")
    partial.write_text(text, encoding="utf-8")
    partial.replace(path)


def read_records(path: Path) -> list[dict]:
    """Read every finished item, in the order it was written."""
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def latest_records(records: list[dict]) -> dict[str, dict]:
    """The newest record of each item.

    An item can have several: a failed request is retried on the next run and
    the new record is appended, so the error stays visible in the file while the
    item itself is measured. Everything that reports on items uses this view.
    """
    latest: dict[str, dict] = {}
    for record in records:
        latest[record["id"]] = record
    return latest


def with_records(items: list[Item], records: list[dict]) -> list[Item]:
    """Put the recorded results back onto the items, in panel order."""
    by_id = latest_records(records)
    merged = []
    for item in items:
        record = by_id.get(item.id)
        if record is None:
            merged.append(item)
            continue
        merged.append(
            Item(
                id=item.id,
                panel=item.panel,
                group=item.group,
                category=item.category,
                language=item.language,
                reference=item.reference,
                audio_sha256=item.audio_sha256,
                hypothesis=record.get("hypothesis", ""),
                error=record.get("error", ""),
                latency_s=float(record.get("latency_s", 0.0)),
                attempts=int(record.get("attempts", 1)),
                duration_s=item.duration_s,
                cold=bool(record.get("cold", False)),
                text_has_casing_punct=item.text_has_casing_punct,
            )
        )
    return merged


def git_commit() -> str:
    """The commit a run was made from, or an empty string outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""
