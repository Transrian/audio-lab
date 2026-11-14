"""Per-clip model answers of the label audit: what one model heard in one clip.

The label audit measures the error of the talk pipeline's model labels against
human labels, on clips that carry both: 3 000 clips of ``nasa-en-train``
(human transcripts) and 1 000 of ``peertube-edu-captions-fr`` (human captions).
Two served models — Whisper large-v3 and Qwen3-ASR, the two of the talk
pipeline's agreement rule — are asked for the transcript of each clip's own
audio, and the agreement rule decides which clips it would have kept and how
its labels score against the human text. The derived set
``nasa-en-train-qwen`` (plan 06, Part B) relabels the whole source with the
Qwen answer alone, and reads the same answers this module caches, so it never
asks for what the audit already asked.

An answer is one JSON file per clip and model, under
``$AUDIOLAB_CACHE/label-audit/<model id>/<clip id>.json``. The file holds
everything that can change the answer — the model id, the language sent, the
extras of the request, and the ``sha256`` of the clip's audio — and a reuse is
trusted only when every one of them matches what the current caller would send
(D-017): another model id, a clip prepared again with other audio, or a caller
sending another language are all asked again. The raw text is kept, not what
was read from it, so a changed reading never reuses an old one.

The audit's clip sample is fixed the same way: ``random.Random(seed).sample``
over the manifest's ids sorted, written to ``<dataset>-sample.json`` next to
the answers, with the seed and the whole id set as its key, so a preparation
that changes under the sample is not served the old one.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path

# The models the audit asks: the two of the agreement rule, by their served
# ids, so a server that serves other ids under these names asks for what it
# really holds.
AUDIT_MODELS = ("stt-v-whisper-large-v3", "stt-v-qwen3-asr-1.7B")
# The seed the plan pins the sample to: the same sample on every machine and
# every run, as long as the dataset's ids do not change.
SAMPLE_SEED = 20261110


class AuditCacheError(RuntimeError):
    """A cached answer that does not describe what the caller would ask."""


def audit_dir(cache: Path) -> Path:
    """Where the audit keeps its answers and its sample."""
    return cache / "label-audit"


def answer_path(cache: Path, model: str, clip_id: str) -> Path:
    """One model's answer for one clip, by the clip's full id."""
    return audit_dir(cache) / model / f"{clip_id}.json"


def read_answer(
    path: Path, *, model: str, language: str, audio_sha256: str, request: dict
) -> str | None:
    """The cached text, or None when the file is absent or not for this request.

    The check is on the file's own words — model, language, request, audio
    hash — not on the path, so a moved or renamed file cannot be reused for
    something it does not describe.
    """
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(row, dict) or row.get("model") != model:
        return None
    if row.get("language") != language or row.get("request") != request:
        return None
    if row.get("audio_sha256") != audio_sha256:
        return None
    text = row.get("text")
    return text if isinstance(text, str) else None


def write_answer(
    path: Path, *, model: str, language: str, audio_sha256: str, request: dict, text: str
) -> None:
    """One answer, written whole: a torn file is read as absent, not as an answer."""
    row = {
        "model": model,
        "language": language,
        "request": request,
        "audio_sha256": audio_sha256,
        "text": text,
    }
    _write_json(path, row)


def _write_json(path: Path, row: dict) -> None:
    """A JSON object written atomically under its own name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".partial")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(row, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def sample_path(cache: Path, dataset: str) -> Path:
    """The fixed clip sample of one dataset, beside the audit's answers."""
    return audit_dir(cache) / f"{dataset}-sample.json"


def read_sample(
    path: Path, *, dataset: str, count: int, seed: int, manifest_ids: list[str]
) -> list[str] | None:
    """The fixed sample, or None when the file does not describe this dataset.

    The key is the dataset name, the seed, the count, and the whole id set of
    the prepared manifest, held beside the draw: a re-preparation that holds
    other clips — even one that still holds every sampled id — is not served
    the old draw, and a caller that asks for another count draws again.
    """
    if not path.is_file():
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(row, dict):
        return None
    if row.get("dataset") != dataset or row.get("seed") != seed or row.get("count") != count:
        return None
    ids = row.get("ids")
    if not isinstance(ids, list) or len(ids) != count or not set(ids) <= set(manifest_ids):
        return None
    if row.get("manifest") != sorted(manifest_ids):
        return None
    return ids


def write_sample(
    path: Path, *, dataset: str, count: int, seed: int, manifest_ids: list[str]
) -> list[str]:
    """Draw the fixed sample over the ids sorted, and write it where it is read back."""
    if count > len(manifest_ids):
        raise AuditCacheError(
            f"{dataset}: the sample asks for {count} clips "
            f"but the manifest holds {len(manifest_ids)}"
        )
    ids = random.Random(seed).sample(sorted(manifest_ids), count)
    _write_json(
        path,
        {
            "dataset": dataset,
            "seed": seed,
            "count": count,
            "ids": ids,
            "manifest": sorted(manifest_ids),
        },
    )
    return ids
