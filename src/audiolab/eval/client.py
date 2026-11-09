"""Talk to any OpenAI-compatible transcription endpoint (D-015).

The URL and the key come from the environment, never from a config file, and
this module never writes either of them anywhere. Failures are raised, not
swallowed: the runner records them as errors so a report never mistakes a
broken request for an empty transcript.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

from audiolab.serve.config import ServeSpec

# Transient failures are retried; anything else is reported as it is.
RETRY_ATTEMPTS = 3
# What replaces the API key in a message that goes into a record: a secret that
# reaches a file people read, copy and attach to issues has leaked.
REDACTED_API_KEY = "<redacted api key>"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
BACKOFF_SECONDS = (1.0, 2.0)
REQUEST_TIMEOUT_S = 120.0
TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"
# How many requests each served model takes at once; see concurrency_from_env.
CONCURRENCY_ENV = "AUDIOLAB_CONCURRENCY"
# Servers document their root URL, but they are also handed out as ".../v1";
# both are accepted, and only one "/v1" ever ends up in a request path.
V1_SUFFIX = "/v1"


def server_root(url: str) -> str:
    """The server root behind a URL: no trailing slash, no trailing ``/v1``."""
    root = url.strip().rstrip("/")
    if root.endswith(V1_SUFFIX):
        root = root[: -len(V1_SUFFIX)].rstrip("/")
    if url.strip() and not root:
        raise ConfigError(f"{url!r} is not a server URL: it must name the server, not only /v1")
    return root


class TranscriptionError(RuntimeError):
    """A request that failed after every retry, and how many requests that was."""

    def __init__(self, message: str, attempts: int = 0) -> None:
        super().__init__(message)
        self.attempts = attempts


class ConfigError(ValueError):
    """A model config file that cannot be used as written."""


@dataclass(frozen=True)
class ModelConfig:
    """One model, as described by ``configs/models/<name>.yaml``.

    A config describes either a server to call (``base_url_env``) or a local
    checkpoint to serve (``serve``); see
    [evaluation.md](../../../docs/stt/evaluation.md). The two are evaluated the
    same way, so the rest of the code sees no difference.
    """

    name: str
    model: str
    api: str = "transcriptions"
    base_url_env: str = "STT_BASE_URL"
    api_key_env: str = ""
    send_language: bool = True
    # What the model is served by, in the words of whoever wrote the config
    # ("local llama.cpp", "vendor API"): recorded in the run identity, and never
    # a URL, since a run record must not name a private endpoint.
    server_label: str = ""
    request: dict = field(default_factory=dict)
    # Present when this machine serves the model itself (D-016). Its checkpoint
    # is then what the answers belong to, not a port number.
    serve: ServeSpec | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> ModelConfig:
        """Read a model config, refusing anything that is clearly wrong."""
        path = Path(path)
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as failure:
            raise ConfigError(f"cannot read model config {path}: {failure}") from failure
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} must contain a mapping")
        known = {field_name for field_name in cls.__dataclass_fields__}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ConfigError(f"{path} has unknown keys: {', '.join(unknown)}")
        for required in ("name", "model"):
            if not raw.get(required):
                raise ConfigError(f"{path} needs a non-empty {required!r}")
        if raw.get("api", "transcriptions") != "transcriptions":
            raise ConfigError(f"{path}: only api 'transcriptions' is supported")
        serve = ServeSpec.from_mapping(raw["serve"], path) if raw.get("serve") else None
        return cls(
            name=str(raw["name"]),
            model=str(raw["model"]),
            api="transcriptions",
            base_url_env=str(raw.get("base_url_env", "STT_BASE_URL")),
            api_key_env=str(raw.get("api_key_env", "")),
            send_language=bool(raw.get("send_language", True)),
            server_label=str(raw.get("server_label", "")),
            request=dict(raw.get("request") or {}),
            serve=serve,
        )

    def identity(self, *, include_serve: bool = True) -> dict:
        """What makes a run this model: the config, without any secret value.

        Only the *names* of the environment variables are recorded, so a run
        identity can be committed or shared safely. A served checkpoint appears
        as its content, never as the path it was read from.

        ``include_serve`` is for a run that was pointed at a running server
        instead of being served its own: it must not describe a checkpoint it
        never loaded, and must not need that file to exist to be recorded.
        """
        return {
            "name": self.name,
            "model": self.model,
            "api": self.api,
            "base_url_env": self.base_url_env,
            "api_key_env": self.api_key_env,
            "send_language": self.send_language,
            "server_label": self.server_label,
            "request": self.request,
            "serve": self.serve.identity() if (self.serve and include_serve) else None,
        }


@dataclass(frozen=True)
class Transcription:
    """What one call produced.

    ``latency_s`` is the time the successful request took and nothing else: a
    retry after a refusal, or a backoff sleep, is not part of how long the model
    took to answer. ``attempts`` counts the requests sent for this item.
    ``answer`` is the whole JSON answer, for a caller that asked the server for
    more than the text (``response_format: verbose_json`` adds timestamps).
    """

    text: str
    language_sent: bool
    latency_s: float
    attempts: int = 1
    answer: dict = field(default_factory=dict, compare=False, repr=False)


class TranscriptionClient:
    """Calls one endpoint; one instance per run, closed at the end of it."""

    def __init__(
        self,
        config: ModelConfig,
        base_url: str,
        api_key: str = "",
        *,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
        timeout_s: float = REQUEST_TIMEOUT_S,
    ):
        self.config = config
        self.base_url = server_root(base_url)
        self.url = self.base_url + TRANSCRIPTIONS_PATH
        self.api_key = api_key
        self.sleep = sleep
        self.client = httpx.Client(transport=transport, timeout=timeout_s)
        # A server that refuses the language field once refuses it always, so it
        # is not asked again for the rest of the run.
        self.language_refused = False

    def __enter__(self) -> TranscriptionClient:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Release the connection pool; a run must not leave sockets behind."""
        self.client.close()

    def transcribe(self, audio: bytes, language: str = "") -> Transcription:
        """Send one WAV and return its transcript.

        ``language`` is empty for non-speech items and when the model config
        says the server does not accept a language field.
        """
        sends_language = bool(language) and self.config.send_language and not self.language_refused
        sent_language = language if sends_language else ""
        answer, refused, latency_s, attempts = self._attempt(audio, sent_language)
        if refused:
            # The server rejected the language field: send the same audio once
            # more without it, and remember that this is what happened.
            self.language_refused = True
            sent_language = ""
            answer, _, latency_s, more = self._attempt(audio, "")
            attempts += more
        return Transcription(
            text=str(answer["text"]),
            language_sent=bool(sent_language),
            latency_s=latency_s,
            attempts=attempts,
            answer=answer,
        )

    def _attempt(self, audio: bytes, language: str) -> tuple[dict, bool, float, int]:
        """One call with its retries; the time and the attempts of the answer it returns."""
        attempt = 0
        while True:
            started = time.monotonic()
            try:
                response = self._post(audio, language)
            except httpx.TransportError as failure:
                attempt += 1
                if attempt >= RETRY_ATTEMPTS:
                    raise TranscriptionError(
                        self._redact(f"connection failed: {failure}"), attempts=attempt
                    ) from failure
                self.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])
                continue
            took = time.monotonic() - started
            attempt += 1
            if response.status_code in RETRY_STATUSES:
                if attempt >= RETRY_ATTEMPTS:
                    raise TranscriptionError(
                        f"HTTP {response.status_code} after {attempt} attempts", attempts=attempt
                    )
                self.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])
                continue
            if response.status_code == 400 and language and "language" in response.text.lower():
                return ({}, True, 0.0, attempt)
            if response.status_code >= 400:
                raise TranscriptionError(
                    self._redact(f"HTTP {response.status_code}: {_short(response.text)}"),
                    attempts=attempt,
                )
            return (_read_answer(response), False, took, attempt)

    def _redact(self, message: str) -> str:
        """A message without the API key, whatever the server put in it.

        An endpoint that answers an error page containing the request headers,
        as several do, would otherwise write the key into ``records.jsonl``.
        """
        if not self.api_key:
            return message
        return message.replace(self.api_key, REDACTED_API_KEY)

    def _post(self, audio: bytes, language: str) -> httpx.Response:
        """The multipart request: 16 kHz mono WAV, plus the configured extras."""
        data = {"model": self.config.model}
        data.update({key: str(value) for key, value in self.config.request.items()})
        if language:
            data["language"] = language
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return self.client.post(
            self.url,
            files={"file": ("audio.wav", audio, "audio/wav")},
            data=data,
            headers=headers,
        )


def _read_answer(response: httpx.Response) -> dict:
    """The answer as JSON, with its transcript under ``text`` even when it is plain text."""
    if "json" in response.headers.get("content-type", ""):
        payload = response.json()
        if not isinstance(payload, dict) or "text" not in payload:
            raise TranscriptionError("JSON answer without a 'text' field")
        return payload
    return {"text": response.text}


def _short(text: str, limit: int = 200) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit] + "..."


def base_url_from_env(config: ModelConfig) -> str:
    """The server root, read from the variable the config names.

    ``STT_BASE_URL`` may hold the server root (``https://server``) or the API
    version root (``https://server/v1``); see :func:`server_root`.
    """
    url = os.environ.get(config.base_url_env, "").strip()
    if not url:
        raise ConfigError(f"${config.base_url_env} is not set; it must hold the server URL")
    return server_root(url)


def api_key_from_env(config: ModelConfig) -> str:
    """The bearer token, read from the variable the config names, if it names one."""
    if not config.api_key_env:
        return ""
    return os.environ.get(config.api_key_env, "").strip()


def concurrency_from_env(config: ModelConfig) -> int:
    """How many requests the served model takes at once: one, unless the environment says more.

    ``$AUDIOLAB_CONCURRENCY`` holds ``<served model id>=<N>`` entries separated
    by commas, for example ``stt-v-whisper-large-v3=10``. What an engine
    supports varies, so a model it does not name is sent one request at a time
    (D-022). Like the URL, it is a fact of the server, and stays out of the
    config files.
    """
    counts: dict[str, int] = {}
    for entry in os.environ.get(CONCURRENCY_ENV, "").split(","):
        if not entry.strip():
            continue
        model, _, count = entry.strip().rpartition("=")
        if not model or not count.isdigit() or int(count) < 1:
            raise ConfigError(
                f"${CONCURRENCY_ENV} holds {entry.strip()!r}; "
                "each entry must be <served model id>=<N>, with N at least 1"
            )
        counts[model] = int(count)
    return counts.get(config.model, 1)


def map_requests[T, R](
    ask: Callable[[T], R],
    items: Iterable[T],
    concurrency: int = 1,
    stop: threading.Event | None = None,
) -> list[R]:
    """``ask`` on every item, ``concurrency`` at a time; the results in the items' order.

    This is how every caller sends requests N at a time (D-022). It runs the
    requests of one model only: a caller that asks several models through a
    hot-swapping server sends every request of one model, through one call,
    before the next model's (D-020), so concurrency never mixes two models.

    The items are taken in order, one at a time, as a worker becomes free, so
    an iterator that is still producing them (a talk still downloading) is read
    as the requests go. No item starts once ``stop`` is set: the items in flight
    finish, and the results returned are those of the items started, the first
    ones. An ``ask`` that fails, or an interrupted caller, sets ``stop`` itself,
    so that an ``ask`` watching it ends early, and the failure is raised once
    the items in flight are done.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, not {concurrency}")
    stop = threading.Event() if stop is None else stop
    numbered = enumerate(items)
    taking = threading.Lock()
    results: dict[int, R] = {}

    def work() -> None:
        try:
            while True:
                with taking:
                    entry = None if stop.is_set() else next(numbered, None)
                if entry is None:
                    return
                index, item = entry
                results[index] = ask(item)
        except BaseException:
            stop.set()
            raise

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        workers = [pool.submit(work) for _ in range(concurrency)]
        try:
            for worker in workers:
                worker.result()
        except BaseException:
            stop.set()
            raise
    return [results[index] for index in range(len(results))]
