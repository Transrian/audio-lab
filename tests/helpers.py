"""Helpers shared by the tests. Everything here works offline and without datasets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import soundfile as sf

from audiolab.core.audio import SAMPLE_RATE
from audiolab.data.panels import Panel
from audiolab.data.store import Clip
from audiolab.train import runner
from audiolab.train.guard import check_leakage
from audiolab.train.run import LOG_NAME, MODEL_NAME
from audiolab.train.runner import Training


def write_tone(path, seconds: float = 1.0, rate: int = SAMPLE_RATE, frequency: int = 440):
    """Write a short readable tone where a test needs real audio on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(int(seconds * rate)) / rate
    samples = (0.3 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    sf.write(str(path), samples, rate, subtype="PCM_16")
    return path


def write_parquet(path, columns: dict) -> object:
    """A Parquet file shaped like the ones the Hub serves, with the audio in it.

    Each key of ``columns`` is a column name and its value is the list of that
    column's values. An ``audio`` column takes ``{"bytes": ..., "path": ...}``
    dictionaries, the shape the Hub gives an audio feature inside Parquet.
    """
    audio_type = pyarrow.struct([("bytes", pyarrow.binary()), ("path", pyarrow.string())])
    arrays = {
        name: pyarrow.array(values, type=audio_type if name == "audio" else None)
        for name, values in columns.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pyarrow.table(arrays), str(path))
    return path


def audio_cell(path, name: str = "clip.wav") -> dict:
    """The ``audio`` cell of a Parquet row holding the file at ``path``."""
    return {"bytes": path.read_bytes(), "path": name}


def make_clip(clip_id: str, language: str = "fr", group: str = "", **fields) -> Clip:
    """A manifest clip with plausible defaults, so a test only states what it checks."""
    is_speech = language != "none"
    values = {
        "id": clip_id,
        "audio": f"audio/{clip_id.replace(':', '_')}.wav",
        # A distinct hash per clip, as real audio has: the transcript cache keys
        # answers on it, so identical fake hashes would hide a wrong reuse.
        "sha256": hashlib.sha256(clip_id.encode("utf-8")).hexdigest(),
        "duration_s": 3.0,
        "language": language,
        "text": "bonjour tout le monde" if is_speech else "",
        "text_has_casing_punct": is_speech,
        "group": group or clip_id,
        "group_kind": "speaker" if is_speech else "clip",
        "source_split": "test",
        "license": "CC-BY-4.0",
        "category": "",
    }
    return Clip(**(values | fields))


def speech_clips(prefix: str, speakers: int = 40, per_speaker: int = 15) -> list[Clip]:
    """A fake speech dataset: several utterances per speaker."""
    return [
        make_clip(f"{prefix}:{speaker:03d}-{index:03d}", group=f"{prefix}:spk{speaker:03d}")
        for speaker in range(speakers)
        for index in range(per_speaker)
    ]


def start_transcription_server(respond):
    """Start a local OpenAI-compatible endpoint for a test.

    ``respond`` receives the request headers and returns ``(status, body,
    content_type)``. The server answers a run's warmup itself, so ``respond``
    sees only the requests of the clips. The server listens on localhost and a
    random port, so tests never touch the network and never use a fixed address.
    """
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - the name is fixed by http.server
            length = int(self.headers.get("Content-Length", 0))
            if is_warmup(self.rfile.read(length)):
                status, body, content_type = 200, '{"text": ""}', "application/json"
            else:
                status, body, content_type = respond(self.headers)
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            """Keep the test output clean."""

    server = http.server.ThreadingHTTPServer(("localhost", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://localhost:{server.server_port}"


def is_warmup(content: bytes) -> bool:
    """Whether a request body is the warmup a run sends before its first request."""
    from audiolab.eval.client import silence_wav

    return silence_wav() in content


# --- what the training tests share ----------------------------------------------
#
# The two tests of `audiolab train` (test_train.py and test_train_mix.py) drive
# the same sequence with the same fakes; what they share lives here, so a change
# to the heavy step's call or to the config keys is made once.


# The keys every config must set, at values small enough for a test. What a run
# trains on, how long, in what batches and at what rates are choices, not
# defaults: a run would otherwise train something the file does not say.
MINIMAL = {
    "base_model": "/models/base.nemo",
    "dataset": "fleurs-fr",
    "clips": 4,
    "updates": 2,
    "batch_size": 2,
    "lr_encoder": 1.0e-5,
    "lr_heads": 3.0e-5,
}
# The same recipe in the mix form: a list of datasets instead of dataset+clips.
MIX_MINIMAL = {
    "base_model": "/models/base.nemo",
    "datasets": [{"name": "ds-a", "clips": 4}, {"name": "ds-b", "clips": 4}],
    "updates": 4,
    "batch_size": 2,
    "lr_encoder": 1.0e-5,
    "lr_heads": 3.0e-5,
}


@dataclass
class FakeResult:
    """What the heavy step returns, with the fields a run's record reads."""

    final_loss: float = 1.5
    updates: int = 64
    wall_seconds: float = 32.0
    peak_vram_gib: float = 7.25
    device: str = "Fake GPU"
    stopped: bool = False


class FakeStep:
    """The heavy step, replaced: it writes the two files a real run writes."""

    def __init__(self) -> None:
        self.calls: list[Training] = []

    def __call__(self, training: Training, *, print_line=print):
        self.calls.append(training)
        (training.directory / MODEL_NAME).write_bytes(b"fine-tuned weights")
        (training.directory / LOG_NAME).write_text(
            '{"update": 64, "loss": 1.5}\n', encoding="utf-8"
        )
        return FakeResult()


def guard_suite(root: Path, names: list[str]) -> tuple[Path, Path]:
    """A suite with one full panel per evaluation dataset, for the guard to check.

    Returns the suite file and the panels directory it names. The panels are
    written through the same ``Panel`` class as the frozen ones, so the guard
    reads a file shaped like the real thing.
    """
    panels_dir = root / "panels"
    panels_dir.mkdir(exist_ok=True)
    for name in names:
        (panels_dir / f"{name}-full.txt").write_text(
            Panel(
                name="full",
                dataset=name,
                ids=[f"{name}:0000"],
                items=1,
                seed=1,
                manifest_sha256="0" * 64,
                clips_sha256="0" * 64,
                dataset_line=f"{name} (1 clip in 1 group; test 1)",
                built="2026-01-01",
            ).text(),
            encoding="utf-8",
        )
    suite = root / "suite.yaml"
    suite.write_text(
        "name: guard\npanels:\n" + "\n".join(f"  - {name}-full" for name in names) + "\n",
        encoding="utf-8",
    )
    return suite, panels_dir


def point_the_guard_at_a_test_suite(monkeypatch, root: Path) -> None:
    """Make the guard of the next training run check a suite the tests own.

    The real guard reads the repository's full suite, whose evaluation datasets
    (fleurs-fr, mls-fr, …) a test would have to dodge by name. Against a suite of
    one dataset the tests never prepare, the guard warns instead of refusing, and
    its block in train.json is still written and checkable.
    """
    suite, panels_dir = guard_suite(root, ["guard-eval"])
    monkeypatch.setattr(
        runner, "check_leakage", partial(check_leakage, suite_path=suite, panels_dir=panels_dir)
    )
