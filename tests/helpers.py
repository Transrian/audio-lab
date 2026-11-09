"""Helpers shared by the tests. Everything here works offline and without datasets."""

from __future__ import annotations

import hashlib

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import soundfile as sf

from audiolab.core.audio import SAMPLE_RATE
from audiolab.data.store import Clip


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
