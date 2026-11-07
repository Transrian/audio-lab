"""ESC-50: environmental sounds, used only as non-speech prompts.

The corpus (2000 five-second clips, 50 classes) is fetched from its official
GitHub repository. Its license is CC BY-NC 3.0, so it is only used to evaluate
models, never to train one (decision 0009), and the repository stores neither
its audio nor its CSV.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from audiolab.data.download import DownloadError, download

CSV_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
AUDIO_URL = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio"
LICENSE = "CC-BY-NC-3.0"
CLIPS_PER_CLASS = 4
CLASSES = 50


def selected_clips(csv_path: Path, per_class: int = CLIPS_PER_CLASS) -> list[dict[str, str]]:
    """Pick ``per_class`` clips per class, spread over the class, in a fixed order.

    The choice depends only on the file list, so every machine prepares the
    same battery without a random seed to remember. Clips of one class are
    taken at a regular stride to avoid a bias towards the start of the class.
    """
    rows = _read_rows(csv_path)
    by_class: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_class.setdefault(row["category"], []).append(row)
    if len(by_class) != CLASSES:
        raise DownloadError(f"esc50.csv lists {len(by_class)} classes, expected {CLASSES}")
    chosen = []
    for category in sorted(by_class):
        clips = sorted(by_class[category], key=lambda row: row["filename"])
        stride = max(1, len(clips) // per_class)
        chosen.extend(clips[index * stride] for index in range(per_class))
    return sorted(chosen, key=lambda row: row["filename"])


def fetch_esc50_clips(audio_dir: Path, chosen: list[dict[str, str]]) -> list[tuple[str, Path]]:
    """Download the chosen clips in parallel and return ``(class, path)`` pairs."""

    def one(row: dict[str, str]) -> Path:
        return download(f"{AUDIO_URL}/{row['filename']}", audio_dir / row["filename"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(one, chosen))
    return [(row["category"], path) for row, path in zip(chosen, paths, strict=True)]


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "filename" not in rows[0] or "category" not in rows[0]:
        raise DownloadError("esc50.csv does not have the expected columns")
    return rows
