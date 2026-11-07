"""The transcription client: retries, the language fallback, and honest errors."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx  # noqa: E402
import pytest  # noqa: E402

from audiolab.eval.client import (  # noqa: E402
    ConfigError,
    ModelConfig,
    TranscriptionClient,
    TranscriptionError,
    host_hash,
)


class Recorder:
    """An HTTP transport that answers from a script and remembers the requests."""

    def __init__(self, answers: list[tuple[int, str, str]]):
        self.answers = answers
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body, content_type = self.answers[
            min(len(self.requests) - 1, len(self.answers) - 1)
        ]
        return httpx.Response(status, text=body, headers={"content-type": content_type})


def make_config(**overrides) -> ModelConfig:
    values = {"name": "test-model", "model": "served-id", "api_key_env": "STT_API_KEY"}
    values.update(overrides)
    return ModelConfig(**values)


def make_client(answers, config: ModelConfig | None = None, sleeps: list[float] | None = None):
    recorder = Recorder(answers)
    slept = sleeps if sleeps is not None else []
    client = TranscriptionClient(
        config or make_config(),
        "http://localhost:9",
        transport=httpx.MockTransport(recorder),
        sleep=slept.append,
    )
    return client, recorder, slept


def answer(text: str, status: int = 200) -> tuple[int, str, str]:
    return (status, json.dumps({"text": text}), "application/json")


def test_json_answer_is_read():
    client, recorder, _ = make_client([answer("bonjour")])
    result = client.transcribe(b"wav", "fr")
    assert result.text == "bonjour"
    assert result.language_sent is True
    assert result.latency_s >= 0
    assert recorder.requests[0].url.path == "/v1/audio/transcriptions"


def test_plain_text_answer_is_accepted_verbatim():
    """The client never "fixes" an answer: normalizing happens in scoring."""
    client, _, _ = make_client([(200, "bonjour\n", "text/plain")])
    assert client.transcribe(b"wav", "fr").text == "bonjour\n"


def test_json_without_text_is_an_error():
    client, _, _ = make_client([(200, json.dumps({"result": "x"}), "application/json")])
    with pytest.raises(TranscriptionError):
        client.transcribe(b"wav", "fr")


def test_transient_status_is_retried_then_succeeds():
    client, recorder, slept = make_client([(503, "busy", "text/plain"), answer("bonjour")])
    assert client.transcribe(b"wav", "fr").text == "bonjour"
    assert len(recorder.requests) == 2
    assert slept == [1.0]


def test_retries_are_given_up_and_reported():
    client, recorder, _ = make_client([(500, "broken", "text/plain")])
    with pytest.raises(TranscriptionError) as failure:
        client.transcribe(b"wav", "fr")
    assert "500" in str(failure.value)
    assert len(recorder.requests) == 3


def test_rate_limit_is_retried():
    client, recorder, _ = make_client([(429, "slow down", "text/plain"), answer("bonjour")])
    assert client.transcribe(b"wav", "fr").text == "bonjour"
    assert len(recorder.requests) == 2


def test_language_refusal_is_retried_without_the_field():
    refusal = (400, json.dumps({"detail": "unsupported language"}), "application/json")
    client, recorder, _ = make_client([refusal, answer("bonjour")])
    result = client.transcribe(b"wav", "fr")
    assert result.text == "bonjour"
    assert result.language_sent is False
    sent = [request.content.decode("utf-8", "replace") for request in recorder.requests]
    assert "language" not in sent[1] or "fr" not in sent[1]


def test_bad_request_that_is_not_about_language_is_an_error():
    client, recorder, _ = make_client(
        [(400, json.dumps({"detail": "bad audio"}), "application/json")]
    )
    with pytest.raises(TranscriptionError):
        client.transcribe(b"wav", "fr")
    assert len(recorder.requests) == 1


def test_non_speech_sends_no_language():
    client, recorder, _ = make_client([answer("")])
    result = client.transcribe(b"wav", "")
    assert result.language_sent is False
    assert b"language" not in recorder.requests[0].content


def test_config_can_forbid_the_language_field():
    client, recorder, _ = make_client([answer("hello")], config=make_config(send_language=False))
    assert client.transcribe(b"wav", "en").language_sent is False
    assert b"language" not in recorder.requests[0].content


def test_request_extras_and_model_are_sent():
    client, recorder, _ = make_client(
        [answer("hi")], config=make_config(request={"temperature": 0})
    )
    client.transcribe(b"wav", "en")
    body = recorder.requests[0].content.decode("utf-8", "replace")
    assert "served-id" in body and "temperature" in body


def test_api_key_is_sent_as_a_bearer_token():
    client, recorder, _ = make_client([answer("hi")])
    client.api_key = "secret-token"
    client.transcribe(b"wav", "fr")
    assert recorder.requests[0].headers["authorization"] == "Bearer secret-token"


def test_transport_failure_is_retried_then_reported():
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    client = TranscriptionClient(
        make_config(),
        "http://localhost:9",
        transport=httpx.MockTransport(explode),
        sleep=lambda _: None,
    )
    with pytest.raises(TranscriptionError) as failure:
        client.transcribe(b"wav", "fr")
    assert "connection failed" in str(failure.value)


def test_model_config_never_holds_a_url_or_a_key(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        "name: x\nmodel: served\nbase_url_env: MY_URL\napi_key_env: MY_KEY\n"
        "send_language: true\nrequest: {}\n",
        encoding="utf-8",
    )
    config = ModelConfig.from_file(path)
    identity = json.dumps(config.identity())
    assert "http" not in identity
    assert "MY_URL" in identity and "MY_KEY" in identity
    assert "secret" not in identity


def test_model_config_refuses_unknown_keys(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("name: x\nmodel: y\nbase_url: http://localhost\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        ModelConfig.from_file(path)


def test_model_config_needs_a_name_and_a_model(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("name: x\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        ModelConfig.from_file(path)


def test_model_config_refuses_another_api(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text("name: x\nmodel: y\napi: chat\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        ModelConfig.from_file(path)


def test_host_hash_hides_the_url():
    digest = host_hash("http://localhost:9000")
    assert len(digest) == 12
    assert "localhost" not in digest
    assert digest == host_hash("http://localhost:9000/")
