"""The HTTP layer of the temporary server: two endpoints, no framework (D-016).

Only what the evaluation client speaks is implemented: ``GET /v1/models`` and
``POST /v1/audio/transcriptions`` with a multipart body. The server holds one
backend and answers one request at a time, which is all a clip-by-clip
evaluation of a local model needs. The standard library is enough here, and a
framework would add a dependency to every install for two routes.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"
MODELS_PATH = "/v1/models"
# The OpenAI form carries more fields than this (prompt, temperature, model,
# ...). They are ignored on purpose: a local checkpoint decodes greedily, and
# pretending to honour an option would make a run's identity a lie.
RESPONSE_FORMATS = ("json", "text")


class RequestError(ValueError):
    """A request that cannot be answered as written; the server replies 400."""


def parse_multipart(body: bytes, content_type: str) -> dict[str, str | bytes]:
    """The form fields of a multipart body: text as ``str``, the file as ``bytes``.

    Parsing with ``email`` rather than by hand keeps quoted field names,
    encodings and the trailing CRLF rules out of this file.
    """
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body
    )
    fields: dict[str, str | bytes] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        if part.get_filename() is not None:
            fields[name] = payload
        else:
            charset = part.get_content_charset() or "utf-8"
            fields[name] = payload.decode(charset, "replace")
    return fields


class TranscriptionServer:
    """One checkpoint behind an OpenAI-compatible transcription API.

    ``transcribe`` receives the bytes of one audio file and the language the
    request asked for (empty when it asked for none) and returns the transcript.
    Everything the server knows about the model comes from that function, so the
    HTTP layer is testable without a model, and a new backend changes nothing
    here.

    ``identity`` is what ``GET /v1/models`` reports the server holds: a run that
    started this server compares it with the checkpoint it meant to serve before
    it records an answer, which is what makes a stale port or a child holding
    other weights impossible to mistake for the model a run asked for.
    """

    def __init__(
        self,
        transcribe: Callable[[bytes, str], str],
        model_id: str,
        host: str = "127.0.0.1",
        port: int = 0,
        identity: dict | None = None,
    ):
        self.transcribe = transcribe
        self.model_id = model_id
        self.identity = identity
        self.host = host
        # A model holds a GPU and is not reentrant, so requests are serialised
        # here rather than inside every backend. The socket is bound now, in the
        # constructor, so a caller that gets a server can use it immediately.
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer((host, port), self._handler())
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self._httpd.server_address[1])

    @property
    def base_url(self) -> str:
        """The server root, the form every model config and cache key uses."""
        return f"http://{self.host}:{self.port}"

    def start(self) -> TranscriptionServer:
        """Serve in a background thread; returns self so it can be started inline."""
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and release the port; safe on a server never started, or twice."""
        if self._thread is not None:
            self._httpd.shutdown()
            self._thread.join(timeout=10)
            self._thread = None
        self._httpd.server_close()

    def __enter__(self) -> TranscriptionServer:
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        """The request handler, which needs the server's fields and nothing else."""
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 - the name is fixed by http.server
                if self.path.rstrip("/") == MODELS_PATH:
                    self._send_json({"object": "list", "data": [server._model_entry()]})
                else:
                    self.send_error(404, "not found")

            def do_POST(self) -> None:  # noqa: N802 - the name is fixed by http.server
                if self.path.rstrip("/") != TRANSCRIPTIONS_PATH:
                    self.send_error(404, "not found")
                    return
                try:
                    body = self._read_body()
                    fields = parse_multipart(body, self.headers.get("Content-Type", ""))
                    audio = fields.get("file")
                    if not isinstance(audio, bytes) or not audio:
                        raise RequestError("the request needs a non-empty 'file' field")
                    # Checked before the model runs: an answer nobody can read is
                    # not worth decoding for.
                    response_format = str(fields.get("response_format") or "json")
                    if response_format not in RESPONSE_FORMATS:
                        raise RequestError(
                            f"response_format {response_format!r} is not supported; "
                            f"use one of {', '.join(RESPONSE_FORMATS)}"
                        )
                    language = fields.get("language")
                    text = server._transcribe(audio, language if isinstance(language, str) else "")
                    if response_format == "text":
                        self._send(text.encode("utf-8"), "text/plain; charset=utf-8")
                    else:
                        self._send_json({"text": text})
                except RequestError as failure:
                    self._send_json({"error": {"message": str(failure)}}, status=400)
                except Exception as failure:  # a backend failure is not a client error
                    self._send_json({"error": {"message": f"transcription failed: {failure}"}}, 500)

            def _read_body(self) -> bytes:
                length = int(self.headers.get("Content-Length") or 0)
                if length <= 0:
                    raise RequestError("the request has no body")
                # Read first, then judge: a body left in the socket would be read
                # as the start of the next request on the same connection.
                body = self.rfile.read(length)
                if "multipart/form-data" not in self.headers.get("Content-Type", ""):
                    raise RequestError("the request must be multipart/form-data")
                return body

            def _send_json(self, payload: dict, status: int = 200) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self._send(body, "application/json", status)

            def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, message: str, *args) -> None:
                """Quiet by default: a run prints its own progress, one line per clip."""

        return Handler

    def _model_entry(self) -> dict:
        entry = {"id": self.model_id, "object": "model", "owned_by": "audiolab"}
        if self.identity is not None:
            entry["identity"] = self.identity
        return entry

    def _transcribe(self, audio: bytes, language: str) -> str:
        with self._lock:
            answer = self.transcribe(audio, language)
        if not isinstance(answer, str):
            # The contract is a transcript, and a backend that breaks it is worth
            # an error: the answer would otherwise be recorded as a result.
            raise TypeError(f"the backend returned {type(answer).__name__}, not a transcript")
        return answer
