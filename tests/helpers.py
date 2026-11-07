"""Helpers shared by the tests. Everything here works offline and without datasets."""

from __future__ import annotations

import hashlib

import numpy as np
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
    content_type)``. The server listens on localhost and a random port, so tests
    never touch the network and never use a fixed address.
    """
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - the name is fixed by http.server
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
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
