"""The transcription client: retries, the language fallback, and honest errors."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx  # noqa: E402
import pytest  # noqa: E402

from audiolab.eval.client import (  # noqa: E402
    ConfigError,
    ModelConfig,
    TranscriptionClient,
    TranscriptionError,
    concurrency_from_env,
    map_requests,
    server_root,
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


def test_the_whole_json_answer_is_kept_for_a_caller_that_asked_for_more():
    verbose = {"text": "bonjour", "segments": [{"start": 0.0, "end": 1.2, "text": "bonjour"}]}
    config = make_config(request={"response_format": "verbose_json"})
    client, recorder, _ = make_client([(200, json.dumps(verbose), "application/json")], config)
    result = client.transcribe(b"wav", "fr")
    assert result.text == "bonjour"
    assert result.answer == verbose
    assert b"verbose_json" in recorder.requests[0].content


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


def test_the_server_label_is_free_text_and_carries_no_url():
    """A run identity records what serves the model, in the config author's words."""
    config = ModelConfig(name="fake-model", model="served-id")
    assert config.identity()["server_label"] == ""
    labelled = ModelConfig(name="fake-model", model="served-id", server_label="local llama.cpp")
    assert labelled.identity()["server_label"] == "local llama.cpp"


def test_a_base_url_may_be_the_server_root_or_its_v1():
    """Servers are handed out as "https://server" and as "https://server/v1"."""
    config = ModelConfig(name="fake-model", model="served-id")
    paths: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={"text": "bonjour"})

    for url in (
        "https://server.example",
        "https://server.example/",
        "https://server.example/v1",
        "https://server.example/v1/",
    ):
        with TranscriptionClient(config, url, transport=httpx.MockTransport(respond)) as client:
            assert client.transcribe(b"wav").text == "bonjour"
    assert paths == ["/v1/audio/transcriptions"] * 4


def test_a_base_url_of_only_v1_is_refused():
    """ "/v1" alone names no server, and the message says so."""
    config = ModelConfig(name="fake-model", model="served-id")
    with pytest.raises(ConfigError, match="not a server URL"):
        TranscriptionClient(config, "/v1")
    with pytest.raises(ConfigError, match="not a server URL"):
        server_root("/v1/")
    assert server_root("https://server.example/v1/") == "https://server.example"


# Sending N requests at a time (D-022).


def test_requests_are_sent_n_at_a_time_and_answered_in_the_items_order():
    together = threading.Barrier(3, timeout=10)  # three in flight, or none is answered
    in_flight, most = [0], [0]
    lock = threading.Lock()

    def ask(number: int) -> int:
        with lock:
            in_flight[0] += 1
            most[0] = max(most[0], in_flight[0])
        if number < 3:
            together.wait()
        time.sleep(0.01 * (5 - number))  # the later items answer first
        with lock:
            in_flight[0] -= 1
        return number * 10

    assert map_requests(ask, range(6), concurrency=3) == [0, 10, 20, 30, 40, 50]
    assert most[0] == 3, "never more than N requests at once"


def test_one_at_a_time_reads_the_items_as_the_requests_go():
    log: list[str] = []

    def items():
        for number in range(3):
            log.append(f"take {number}")
            yield number

    def ask(number: int) -> int:
        log.append(f"ask {number}")
        return number

    assert map_requests(ask, items()) == [0, 1, 2]
    assert log == ["take 0", "ask 0", "take 1", "ask 1", "take 2", "ask 2"]


def test_a_failed_request_stops_the_others_and_is_raised_once_they_are_done():
    failing = threading.Event()
    started: list[int] = []
    finished: list[int] = []

    def ask(number: int) -> int:
        started.append(number)
        if number == 1:
            raise TranscriptionError("HTTP 500")
        assert failing.wait(timeout=10), "the failure sets the event the asks watch"
        finished.append(number)
        return number

    with pytest.raises(TranscriptionError, match="HTTP 500"):
        map_requests(ask, range(10), concurrency=2, failing=failing)

    assert sorted(started) == [0, 1], "no item starts once a request failed"
    assert finished == [0], "the request in flight finishes"


def test_a_stopped_run_starts_no_new_request_and_returns_those_it_started():
    stop = threading.Event()

    def ask(number: int) -> int:
        if number == 2:
            stop.set()  # Ctrl-C while the third request is answered
        return number

    assert map_requests(ask, range(10), should_stop=stop.is_set) == [0, 1, 2]
    assert map_requests(ask, range(10), concurrency=3, should_stop=stop.is_set) == []


def test_a_concurrency_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        map_requests(str, [1], concurrency=0)


def test_the_concurrency_of_a_model_comes_from_the_environment_and_defaults_to_one(monkeypatch):
    whisper, other = make_config(model="stt-v-whisper-large-v3"), make_config(model="other")
    monkeypatch.delenv("AUDIOLAB_CONCURRENCY", raising=False)
    assert concurrency_from_env(whisper) == 1

    monkeypatch.setenv("AUDIOLAB_CONCURRENCY", " stt-v-whisper-large-v3=10 , qwen=4,")
    assert (concurrency_from_env(whisper), concurrency_from_env(other)) == (10, 1)

    for wrong in ("stt-v-whisper-large-v3", "stt-v-whisper-large-v3=0", "=3", "a=many"):
        monkeypatch.setenv("AUDIOLAB_CONCURRENCY", wrong)
        with pytest.raises(ConfigError, match="<served model id>=<N>"):
            concurrency_from_env(whisper)
