"""The temporary server: the HTTP contract, and the identity of its checkpoint.

Nothing here imports NeMo or torch. The server takes the transcribing function
as an argument, so a fake backend drives the whole path - that is the same seam
the evaluation runner uses.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx  # noqa: E402
import pytest  # noqa: E402

from audiolab.core.checkpoint import file_sha256  # noqa: E402
from audiolab.eval.client import ModelConfig, TranscriptionClient  # noqa: E402
from audiolab.serve.config import ServeConfigError, ServeSpec  # noqa: E402
from audiolab.serve.http import TranscriptionServer, parse_multipart  # noqa: E402
from audiolab.serve.launcher import serve_command, serving, start_server  # noqa: E402

FAKE_WAV = b"RIFF\x00\x00\x00\x00WAVEfmt "


def fake_backend(text: str = "bonjour tout le monde", calls: list | None = None):
    """A transcribing function that records what it was asked."""

    def transcribe(audio: bytes, language: str = "") -> str:
        if calls is not None:
            calls.append((audio, language))
        return text

    return transcribe


def model_config(path: Path, serve_block: str = "") -> ModelConfig:
    path.write_text(
        "name: local-model\nmodel: parakeet-tdt-0.6b-v3\n"
        "base_url_env: UNUSED_URL\nserver_label: this machine\n"
        f"{serve_block}",
        encoding="utf-8",
    )
    return ModelConfig.from_file(path)


def checkpoint_file(tmp_path: Path, name: str = "model.nemo", content: bytes = b"weights") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_the_server_answers_the_form_the_evaluation_client_sends(tmp_path):
    """The contract is the client's, checked through the client itself."""
    calls: list = []
    with TranscriptionServer(fake_backend(calls=calls), "parakeet-tdt-0.6b-v3") as server:
        config = ModelConfig(name="local", model="parakeet-tdt-0.6b-v3")
        with TranscriptionClient(config, server.base_url) as client:
            result = client.transcribe(FAKE_WAV, "fr")
    assert result.text == "bonjour tout le monde"
    assert result.language_sent
    assert calls == [(FAKE_WAV, "fr")]


def test_models_lists_the_served_checkpoint(tmp_path):
    with TranscriptionServer(fake_backend(), "my-checkpoint") as server:
        answer = httpx.get(f"{server.base_url}/v1/models")
        assert answer.status_code == 200
        assert [entry["id"] for entry in answer.json()["data"]] == ["my-checkpoint"]


def test_the_server_root_and_the_version_root_answer_the_same():
    """A client may be pointed at either; the server accepts both."""
    with TranscriptionServer(fake_backend("bonjour"), "m") as server:
        for url in (server.base_url, f"{server.base_url}/v1"):
            config = ModelConfig(name="local", model="m")
            with TranscriptionClient(config, url) as client:
                assert client.transcribe(FAKE_WAV, "").text == "bonjour"


def test_a_text_response_format_answers_plain_text():
    """The form field decides the answer's shape, as OpenAI's API does."""
    with TranscriptionServer(fake_backend("bonjour tout le monde"), "m") as server:
        answer = httpx.post(
            f"{server.base_url}/v1/audio/transcriptions",
            files={"file": ("audio.wav", FAKE_WAV, "audio/wav")},
            data={"model": "m", "response_format": "text"},
        )
    assert answer.status_code == 200
    assert answer.headers["content-type"].startswith("text/plain")
    assert answer.text == "bonjour tout le monde"


def test_an_unknown_response_format_is_refused_without_running_the_model():
    calls: list = []
    with TranscriptionServer(fake_backend(calls=calls), "m") as server:
        answer = httpx.post(
            f"{server.base_url}/v1/audio/transcriptions",
            files={"file": ("audio.wav", FAKE_WAV, "audio/wav")},
            data={"response_format": "srt"},
        )
    assert answer.status_code == 400
    assert "srt" in answer.json()["error"]["message"]
    assert calls == []  # nothing was decoded for an answer nobody could read


def test_a_request_without_audio_is_refused():
    with TranscriptionServer(fake_backend(), "m") as server:
        missing = httpx.post(
            f"{server.base_url}/v1/audio/transcriptions",
            files={"file": ("", b"", "audio/wav")},
            data={"model": "m"},
        )
        wrong_type = httpx.post(
            f"{server.base_url}/v1/audio/transcriptions",
            content=b"not multipart",
            headers={"Content-Type": "application/json"},
        )
        unknown_path = httpx.post(f"{server.base_url}/v1/other", content=b"")
    assert missing.status_code == 400
    assert "file" in missing.json()["error"]["message"]
    # A refusal must not mention the language field: the client reads that word
    # in a 400 as "this server does not take a language", and would stop
    # sending one for the rest of the run.
    assert "language" not in missing.json()["error"]["message"].lower()
    assert wrong_type.status_code == 400
    assert unknown_path.status_code == 404


def test_a_backend_failure_is_an_error_not_an_empty_transcript():
    """An exception in the model must never be cached as an empty answer."""

    def broken(audio: bytes, language: str = "") -> str:
        raise RuntimeError("out of memory")

    with TranscriptionServer(broken, "m") as server:
        answer = httpx.post(
            f"{server.base_url}/v1/audio/transcriptions",
            files={"file": ("audio.wav", FAKE_WAV, "audio/wav")},
            data={"model": "m"},
        )
    assert answer.status_code == 500
    assert "out of memory" in answer.json()["error"]["message"]


def test_multipart_fields_are_read_with_their_own_encoding():
    """Field names and values come from a real multipart body, not from a regex."""
    body = (
        b"--BOUND\r\n"
        b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n"
        b"RIFFdata\r\n"
        b"--BOUND\r\n"
        b'Content-Disposition: form-data; name="language"\r\n\r\n'
        b"fr\r\n"
        b"--BOUND--\r\n"
    )
    fields = parse_multipart(body, "multipart/form-data; boundary=BOUND")
    assert fields["file"] == b"RIFFdata"
    assert fields["language"] == "fr"


FAKE_SERVER = Path(__file__).resolve().parent / "fake_serve.py"


def fake_command(spec=None, model_id: str = "") -> list[str]:
    """The command a test serves with: the real child protocol, no model."""
    return [sys.executable, str(FAKE_SERVER), "0", "--stop-when-stdin-closes"]


def test_a_server_child_announces_the_address_it_serves():
    """The port is chosen by the child and read from its output, never guessed."""
    process = start_server(
        ServeSpec(checkpoint="x.nemo"), "fake-model", command_factory=fake_command
    )
    try:
        assert process.base_url.startswith("http://127.0.0.1:")
        answer = httpx.get(f"{process.base_url}/v1/models", timeout=5)
        assert answer.json()["data"][0]["id"] == "fake-model"
    finally:
        process.stop()
    with pytest.raises(httpx.TransportError):
        httpx.get(f"{process.base_url}/v1/models", timeout=2)


def test_two_servers_of_the_same_checkpoint_do_not_need_the_same_port():
    """Every run gets a free port; nothing may depend on which one it is."""
    first = start_server(ServeSpec(checkpoint="x.nemo"), "m", command_factory=fake_command)
    second = start_server(ServeSpec(checkpoint="x.nemo"), "m", command_factory=fake_command)
    try:
        assert first.base_url != second.base_url
        assert httpx.get(f"{second.base_url}/v1/models", timeout=5).status_code == 200
    finally:
        first.stop()
        second.stop()


def test_the_server_is_stopped_when_the_serving_block_ends():
    with serving(ServeSpec(checkpoint="x.nemo"), "m", command_factory=fake_command) as base_url:
        assert httpx.get(f"{base_url}/v1/models", timeout=5).status_code == 200
    with pytest.raises(httpx.TransportError):
        httpx.get(f"{base_url}/v1/models", timeout=2)


def test_a_child_that_cannot_start_says_what_it_printed():
    """An interpreter without NeMo must fail with its own message, not a timeout."""

    def broken_command(spec=None, model_id: str = "") -> list[str]:
        return [sys.executable, "-c", "raise SystemExit('no module named nemo')"]

    with pytest.raises(ServeConfigError, match="no module named nemo"):
        start_server(ServeSpec(checkpoint="x.nemo"), "m", command_factory=broken_command)


def test_server_commands_name_the_checkpoint_and_the_interpreter(tmp_path, monkeypatch):
    """The child is told exactly what to load, in the interpreter that can load it."""
    checkpoint = checkpoint_file(tmp_path, "serve-me.nemo", b"weights")
    monkeypatch.setenv("AUDIOLAB_NEMO_PYTHON", "/opt/nemo/python")
    spec = ServeSpec(checkpoint=str(checkpoint), gpu="GPU-abc", max_symbols=7)
    command = serve_command(spec, "my-id")
    assert command[0] == "/opt/nemo/python"
    assert command[1:3] == ["-m", "audiolab.serve"]
    assert command[command.index("--checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--gpu") + 1] == "GPU-abc"
    assert command[command.index("--max-symbols") + 1] == "7"
    assert command[command.index("--model-id") + 1] == "my-id"
    assert "--stop-when-stdin-closes" in command

    monkeypatch.delenv("AUDIOLAB_NEMO_PYTHON")
    assert serve_command(spec, "my-id")[0] == sys.executable  # this environment, by default


def test_serve_binds_the_port_before_returning(tmp_path):
    """The in-process server (what `audiolab serve` runs) listens when it returns."""
    server = TranscriptionServer(fake_backend(), "m", port=0).start()
    try:
        assert server.port > 0
        assert server.base_url == f"http://127.0.0.1:{server.port}"
        assert httpx.get(f"{server.base_url}/v1/models", timeout=5).status_code == 200
    finally:
        server.stop()


def test_a_serve_block_is_read_and_its_unknown_keys_refused(tmp_path):
    config = model_config(
        tmp_path / "local.yaml",
        "send_language: true\nserve:\n  backend: nemo\n  checkpoint: /tmp/some.nemo\n  gpu: '1'\n",
    )
    assert config.serve is not None
    assert config.serve.backend == "nemo"
    assert config.serve.checkpoint == "/tmp/some.nemo"
    assert config.serve.gpu == "1"

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: x\nmodel: y\nserve:\n  backend: nemo\n  checkpoint: /tmp/a.nemo\n  beam: 4\n",
        encoding="utf-8",
    )
    with pytest.raises(ServeConfigError, match="unknown serve keys: beam"):
        ModelConfig.from_file(bad)


def test_a_serve_block_needs_a_checkpoint():
    with pytest.raises(ServeConfigError, match="needs a 'checkpoint'"):
        ServeSpec.from_mapping({"backend": "nemo"})
    with pytest.raises(ServeConfigError, match="unknown serve backend"):
        ServeSpec.from_mapping({"backend": "vllm", "checkpoint": "/tmp/a.nemo"})


def test_the_gpu_override_keeps_the_rest_of_the_block():
    spec = ServeSpec(checkpoint="/tmp/a.nemo", gpu="0", max_symbols=4)
    assert spec.with_gpu("GPU-abc").gpu == "GPU-abc"
    assert spec.with_gpu("GPU-abc").max_symbols == 4
    assert spec.with_gpu(None) is spec


def test_the_identity_of_a_checkpoint_is_its_content_not_its_path(tmp_path):
    """A cache key must survive a move and a copy, and notice a rewrite."""
    first = checkpoint_file(tmp_path, "a.nemo", b"weights-one")
    moved = tmp_path / "elsewhere" / "b.nemo"
    moved.parent.mkdir()
    moved.write_bytes(first.read_bytes())

    spec = ServeSpec(checkpoint=str(first))
    assert spec.identity() == ServeSpec(checkpoint=str(moved)).identity()
    assert spec.identity()["checkpoint"] == f"sha256={file_sha256(first)}"
    assert str(tmp_path) not in json.dumps(spec.identity())

    first.write_bytes(b"weights-two")
    assert spec.identity() != ServeSpec(checkpoint=str(moved)).identity()


def test_the_identity_follows_the_decoding_settings(tmp_path):
    """Greedy with another symbol budget can answer differently, so it is a new key."""
    checkpoint = checkpoint_file(tmp_path, "same.nemo", b"same-weights")
    spec = ServeSpec(checkpoint=str(checkpoint), max_symbols=10)
    other = ServeSpec(checkpoint=str(checkpoint), max_symbols=20)
    assert spec.identity()["checkpoint"] == other.identity()["checkpoint"]
    assert spec.identity()["decoding"] != other.identity()["decoding"]
    assert spec.identity()["decoding"]["strategy"] == "greedy_batch"


def test_a_hugging_face_id_is_an_identity_of_its_own():
    spec = ServeSpec(checkpoint="nvidia/parakeet-tdt-0.6b-v3")
    assert spec.identity()["checkpoint"] == "hf=nvidia/parakeet-tdt-0.6b-v3"
    assert spec.resolved().hf_id == "nvidia/parakeet-tdt-0.6b-v3"
    assert spec.resolved().path == ""


def test_the_checkpoint_variable_overrides_the_committed_path(tmp_path, monkeypatch):
    """A committed config names a public model; this machine serves a local copy."""
    local = checkpoint_file(tmp_path, "local.nemo", b"local-weights")
    spec = ServeSpec(checkpoint="/committed/path.nemo", checkpoint_env="AUDIOLAB_TEST_NEMO")
    monkeypatch.setenv("AUDIOLAB_TEST_NEMO", str(local))
    assert spec.resolved().path == str(local)
    assert spec.identity()["checkpoint"] == f"sha256={file_sha256(local)}"

    monkeypatch.delenv("AUDIOLAB_TEST_NEMO")
    assert spec.resolved().path == "/committed/path.nemo"


def test_a_missing_checkpoint_says_so_before_anything_runs():
    spec = ServeSpec(checkpoint="/does/not/exist.nemo")
    with pytest.raises(ServeConfigError, match="does not exist"):
        spec.identity()


def test_a_run_identity_carries_the_checkpoint_content_and_no_path(tmp_path):
    checkpoint = checkpoint_file(tmp_path, "run.nemo", b"trained-weights")
    config = model_config(
        tmp_path / "m.yaml", f"serve:\n  backend: nemo\n  checkpoint: {checkpoint}\n"
    )
    identity = json.dumps(config.identity(), sort_keys=True)
    assert file_sha256(checkpoint) in identity
    assert str(tmp_path) not in identity
