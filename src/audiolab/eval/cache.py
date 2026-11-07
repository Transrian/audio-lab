"""The transcript cache: what a model answered, kept once per clip.

Sending audio to a model is the slow part of an evaluation; scoring takes
seconds. So answers are stored apart from any suite, under a key made of what
decides them and nothing else: the model as it is asked (served id, request
options, whether a language is sent), the server that answered, and the audio
of the clip. A new dataset, a new suite, a rebuilt panel or a new commit then
sends only the clips never asked before, and a normalizer change sends nothing.

A cache file is append-only: every answer is added and flushed as it arrives,
and when a clip was asked more than once the newest answer is the one used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from audiolab.eval.client import ModelConfig
from audiolab.eval.records import ENDPOINT_FINGERPRINT_KEY, read_records

# Bump when a change to the client could change what a server returns (how the
# audio is encoded, which fields a request carries): every clip is asked again.
CACHE_FORMAT = 1
TRANSCRIPTS_DIR = "transcripts"


def transcript_key(model: ModelConfig, endpoint_fingerprint: str) -> dict:
    """Everything that decides an answer, apart from the clip itself.

    The config's ``name``, ``server_label`` and environment variable names are
    left out on purpose: renaming a config or relabelling a server changes no
    answer, so it must not throw the answers away.
    """
    return {
        "cache_format": CACHE_FORMAT,
        "api": model.api,
        "model": model.model,
        "send_language": model.send_language,
        "request": model.request,
        ENDPOINT_FINGERPRINT_KEY: endpoint_fingerprint,
    }


def cache_file(runs_root: Path, model: ModelConfig, key: dict) -> Path:
    """``<runs>/transcripts/<config name>/<key digest>.jsonl``.

    The config name is only there so a person finds the file; the digest is what
    separates two ways of asking the same model.
    """
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return Path(runs_root) / TRANSCRIPTS_DIR / model.name / f"{digest}.jsonl"


def open_cache(path: Path, key: dict) -> None:
    """Create the cache directory and write, once, the key the file stands for.

    The key file holds the endpoint fingerprint, which is meaningless outside
    this machine (see ``runner.identity_key``); it never enters a summary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    key_path = path.with_suffix(".key.json")
    if not key_path.is_file():
        key_path.write_text(json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cached_answers(path: Path) -> dict[tuple[str, str], dict]:
    """The newest answer per clip, keyed by (clip id, audio sha256).

    The audio hash is part of the key so that a clip whose audio changed (a
    re-fetch, a fixed preparation step) is asked again rather than scored with
    an answer to other audio.
    """
    answers: dict[tuple[str, str], dict] = {}
    for record in read_records(path):
        answers[(record["id"], record.get("audio_sha256", ""))] = record
    return answers
