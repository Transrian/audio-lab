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

# Fields of one record; anything else is corruption rather than a new feature.
RECORD_FIELDS = (
    "id",
    "panel",
    "hypothesis",
    "error",
    "latency_s",
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
    panel_title: str, panels_dir: Path, data_dir_: Path | None = None
) -> list[Item]:
    """Every item of a panel, without any hypothesis yet."""
    dataset, _ = dataset_and_panel(panel_title)
    panel = read_panel(panels_dir / f"{panel_title}.txt")
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
            duration_s=clips[clip_id].duration_s,
            text_has_casing_punct=clips[clip_id].text_has_casing_punct,
        )
        for clip_id in panel.ids
    ]


def clip_paths(panel_title: str, data_dir_: Path | None = None) -> dict[str, Path]:
    """Where the audio of each clip of a panel lives."""
    dataset, _ = dataset_and_panel(panel_title)
    root = Path(data_dir_) if data_dir_ else data_dir()
    return {clip.id: root / dataset / clip.audio for clip in read_manifest(root / dataset)}


def panel_hashes(panel_titles: list[str], panels_dir: Path) -> dict[str, str]:
    """sha256 of each panel file, so a resume can tell whether the ids changed."""
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


def read_records(path: Path) -> list[dict]:
    """Read every finished item, in the order it was written."""
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def with_records(items: list[Item], records: list[dict]) -> list[Item]:
    """Put the recorded results back onto the items, in panel order."""
    by_id = {record["id"]: record for record in records}
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
                hypothesis=record.get("hypothesis", ""),
                error=record.get("error", ""),
                latency_s=float(record.get("latency_s", 0.0)),
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
