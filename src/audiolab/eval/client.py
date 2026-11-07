"""Talk to any OpenAI-compatible transcription endpoint (D-015).

The URL and the key come from the environment, never from a config file, and
this module never writes either of them anywhere. Failures are raised, not
swallowed: the runner records them as errors so a report never mistakes a
broken request for an empty transcript.
"""

from __future__ import annotations

import hashlib
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


class TranscriptionError(RuntimeError):
    """A request that failed after every retry."""


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
            "request": self.request,
        }


@dataclass(frozen=True)
class Transcription:
    """What one call produced."""

    text: str
    language_sent: bool
    latency_s: float


class TranscriptionClient:
    """Calls one endpoint; one instance per run."""

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
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.sleep = sleep
        self.client = httpx.Client(transport=transport, timeout=timeout_s)

    def transcribe(self, audio: bytes, language: str = "") -> Transcription:
        """Send one WAV and return its transcript.

        ``language`` is empty for non-speech items and when the model config
        says the server does not accept a language field.
        """
        sent_language = language if (language and self.config.send_language) else ""
        started = time.monotonic()
        text, failed_with_language = self._attempt(audio, sent_language)
        if failed_with_language:
            # The server rejected the language field: send the same audio once
            # more without it, and remember that this is what happened.
            text, _ = self._attempt(audio, "")
            sent_language = ""
        return Transcription(
            text=text, language_sent=bool(sent_language), latency_s=time.monotonic() - started
        )

    def _attempt(self, audio: bytes, language: str) -> tuple[str, bool]:
        """One call with its retries; returns the text and whether language was refused."""
        attempt = 0
        while True:
            try:
                response = self._post(audio, language)
            except httpx.TransportError as failure:
                attempt += 1
                if attempt >= RETRY_ATTEMPTS:
                    raise TranscriptionError(f"connection failed: {failure}") from failure
                self.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])
                continue
            if response.status_code in RETRY_STATUSES:
                attempt += 1
                if attempt >= RETRY_ATTEMPTS:
                    raise TranscriptionError(
                        f"HTTP {response.status_code} after {attempt} attempts"
                    )
                self.sleep(BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)])
                continue
            if response.status_code == 400 and language and "language" in response.text.lower():
                return ("", True)
            if response.status_code >= 400:
                raise TranscriptionError(f"HTTP {response.status_code}: {_short(response.text)}")
            return (_read_text(response), False)

    def _post(self, audio: bytes, language: str) -> httpx.Response:
        """The multipart request: 16 kHz mono WAV, plus the configured extras."""
        data = {"model": self.config.model}
        data.update({key: str(value) for key, value in self.config.request.items()})
        if language:
            data["language"] = language
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return self.client.post(
            self.base_url + TRANSCRIPTIONS_PATH,
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
    """The endpoint URL, read from the variable the config names."""
    url = os.environ.get(config.base_url_env, "").strip()
    if not url:
        raise ConfigError(f"${config.base_url_env} is not set; it must hold the server URL")
    return url


def api_key_from_env(config: ModelConfig) -> str:
    """The bearer token, read from the variable the config names, if it names one."""
    if not config.api_key_env:
        return ""
    return os.environ.get(config.api_key_env, "").strip()


def host_hash(url: str) -> str:
    """A stable fingerprint of the endpoint host, safe to write into run.json."""
    return hashlib.sha256(url.rstrip("/").encode("utf-8")).hexdigest()[:12]
