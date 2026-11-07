"""Talk to any OpenAI-compatible transcription endpoint (D-015).

The URL and the key come from the environment, never from a config file, and
this module never writes either of them anywhere. Failures are raised, not
swallowed: the runner records them as errors so a report never mistakes a
broken request for an empty transcript.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml

# Transient failures are retried; anything else is reported as it is.
RETRY_ATTEMPTS = 3
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
BACKOFF_SECONDS = (1.0, 2.0)
REQUEST_TIMEOUT_S = 120.0
TRANSCRIPTIONS_PATH = "/v1/audio/transcriptions"
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
    """One model, as described by ``configs/models/<name>.yaml``."""

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
        return cls(
            name=str(raw["name"]),
            model=str(raw["model"]),
            api="transcriptions",
            base_url_env=str(raw.get("base_url_env", "STT_BASE_URL")),
            api_key_env=str(raw.get("api_key_env", "")),
            send_language=bool(raw.get("send_language", True)),
            server_label=str(raw.get("server_label", "")),
            request=dict(raw.get("request") or {}),
        )

    def identity(self) -> dict:
        """What makes a run this model: the config, without any secret value.

        Only the *names* of the environment variables are recorded, so a run
        identity can be committed or shared safely.
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
        }


@dataclass(frozen=True)
class Transcription:
    """What one call produced.

    ``latency_s`` is the time the successful request took and nothing else: a
    retry after a refusal, or a backoff sleep, is not part of how long the model
    took to answer. ``attempts`` counts the requests sent for this item.
    """

    text: str
    language_sent: bool
    latency_s: float
    attempts: int = 1


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
        text, refused, latency_s, attempts = self._attempt(audio, sent_language)
        if refused:
            # The server rejected the language field: send the same audio once
            # more without it, and remember that this is what happened.
            self.language_refused = True
            sent_language = ""
            text, _, latency_s, more = self._attempt(audio, "")
            attempts += more
        return Transcription(
            text=text,
            language_sent=bool(sent_language),
            latency_s=latency_s,
            attempts=attempts,
        )

    def _attempt(self, audio: bytes, language: str) -> tuple[str, bool, float, int]:
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
                        f"connection failed: {failure}", attempts=attempt
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
                return ("", True, 0.0, attempt)
            if response.status_code >= 400:
                raise TranscriptionError(
                    f"HTTP {response.status_code}: {_short(response.text)}", attempts=attempt
                )
            return (_read_text(response), False, took, attempt)

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


def _read_text(response: httpx.Response) -> str:
    """The transcript, from JSON when the server answers JSON and plainly otherwise."""
    if "json" in response.headers.get("content-type", ""):
        payload = response.json()
        if not isinstance(payload, dict) or "text" not in payload:
            raise TranscriptionError("JSON answer without a 'text' field")
        return str(payload["text"])
    return response.text


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
