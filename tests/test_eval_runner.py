"""A run end to end: a local fake endpoint, a tiny dataset, and the files it leaves."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx  # noqa: E402
import pytest  # noqa: E402
from helpers import make_clip, start_transcription_server, write_tone  # noqa: E402

from audiolab.data.panels import build_panels, read_panel  # noqa: E402
from audiolab.data.store import pcm_sha256_of, write_manifest  # noqa: E402
from audiolab.eval.client import ModelConfig, TranscriptionClient  # noqa: E402
from audiolab.eval.records import (  # noqa: E402
    IDENTITY_NAME,
    RECORDS_NAME,
    latest_records,
    read_records,
)
from audiolab.eval.report import REPORT_NAME, SUMMARY_NAME, compare_runs, score_run  # noqa: E402
from audiolab.eval.runner import RunContext, RunError, SuiteSpec, load_suite, run_suite  # noqa: E402

PANEL = "fake-fr-fast"


@pytest.fixture
def workspace(tmp_path) -> dict:
    """A prepared dataset, its panels, a model config and a suite, all in tmp_path."""
    data_dir = tmp_path / "datasets"
    clips = []
    for speaker in range(6):
        for index in range(3):
            clip_id = f"fake:{speaker:03d}-{index:03d}"
            clip = make_clip(clip_id, group=f"fake:spk{speaker:03d}", text="bonjour tout le monde")
            write_tone(data_dir / "fake-fr" / clip.audio, seconds=1.0)
            clips.append(clip)
    write_manifest(data_dir / "fake-fr", clips)
    panels_dir = tmp_path / "panels"
    build_panels("fake-fr", clips, panels_dir, built="2026-11-07")

    model_path = tmp_path / "model.yaml"
    model_path.write_text(
        "name: fake-model\nmodel: served-id\nbase_url_env: FAKE_URL\napi_key_env: FAKE_KEY\n"
        "send_language: true\nrequest: {}\n",
        encoding="utf-8",
    )
    suite_path = tmp_path / "fast.yaml"
    suite_path.write_text(f"name: fast\npanels: [{PANEL}]\n", encoding="utf-8")
    return {
        "data_dir": data_dir,
        "panels_dir": panels_dir,
        "runs_dir": tmp_path / "runs",
        "model": ModelConfig.from_file(model_path),
        "suite": load_suite(suite_path),
    }


def context(workspace: dict, **overrides) -> RunContext:
    values = {
        "model": workspace["model"],
        "suite": workspace["suite"],
        "panels_dir": workspace["panels_dir"],
        "data_dir": workspace["data_dir"],
        "runs_root": workspace["runs_dir"],
        "base_url": "http://localhost:9",
        "verbose": False,
    }
    values.update(overrides)
    return RunContext(**values)


def panel_size(workspace: dict, panel: str = PANEL) -> int:
    return len(read_panel(workspace["panels_dir"] / f"{panel}.txt").ids)


def server_answering(text: str = "bonjour tout le monde", counter: list | None = None):
    """A fake endpoint that answers with one text and counts the calls."""

    def respond(headers):
        if counter is not None:
            counter.append(headers.get("Authorization", ""))
        return (200, json.dumps({"text": text}), "application/json")

    return start_transcription_server(respond)


def test_a_run_records_every_item_and_writes_a_summary(workspace):
    calls: list = []
    server, url = server_answering(counter=calls)
    try:
        run_dir = run_suite(context(workspace, base_url=url))
    finally:
        server.shutdown()
    records = read_records(run_dir / RECORDS_NAME)
    assert len(records) == panel_size(workspace)
    assert all(record["error"] == "" for record in records)
    assert all(record["hypothesis"] == "bonjour tout le monde" for record in records)
    assert sum(record["cold"] for record in records) == 1
    assert all(record["language_sent"] for record in records)
    identity = json.loads((run_dir / IDENTITY_NAME).read_text(encoding="utf-8"))
    assert url not in json.dumps(identity)  # a run record never names the endpoint
    assert identity["model"]["server_label"] == ""
    assert identity["normalizer_version"] == 1
    summary = json.loads((run_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["panels"][PANEL]["wer"]["normalized"]["value"] == 0.0
    assert summary["run"]["records"] == panel_size(workspace)
    assert summary["panels"][PANEL]["wer"]["formatted"]["value"] == 0.0
    report = (run_dir / REPORT_NAME).read_text(encoding="utf-8")
    assert "## fake-fr-fast" in report
    assert "normalized" in report and "95 % CI" in report
    assert "What this does not measure" in report


def test_a_second_run_resumes_without_asking_again(workspace):
    calls: list = []
    server, url = server_answering(counter=calls)
    try:
        first = run_suite(context(workspace, base_url=url))
        second = run_suite(context(workspace, base_url=url))
    finally:
        server.shutdown()
    assert first == second
    assert len(calls) == panel_size(workspace)
    assert len(read_records(second / RECORDS_NAME)) == panel_size(workspace)


def test_a_changed_panel_refuses_to_resume(workspace):
    server, url = server_answering()
    try:
        run_suite(context(workspace, base_url=url))
        changed = context(workspace, base_url=url)
        changed.suite = SuiteSpec(name="fast", panels=[PANEL])
        (workspace["panels_dir"] / f"{PANEL}.txt").write_text(
            (workspace["panels_dir"] / f"{PANEL}.txt")
            .read_text(encoding="utf-8")
            .replace("fake:000-000", "fake:000-001"),
            encoding="utf-8",
        )
        with pytest.raises(RunError, match="different run"):
            run_suite(changed)
    finally:
        server.shutdown()


def test_fresh_starts_a_new_directory(workspace):
    server, url = server_answering()
    try:
        first = run_suite(context(workspace, base_url=url))
        second = run_suite(context(workspace, base_url=url, fresh=True))
    finally:
        server.shutdown()
    assert second != first
    assert second.name.startswith("fast-")
    assert (second / IDENTITY_NAME).is_file()


def test_a_failed_request_is_recorded_as_an_error(workspace):
    def refuse(headers):
        return (400, json.dumps({"detail": "no audio"}), "application/json")

    server, url = start_transcription_server(refuse)
    try:
        with pytest.raises(RunError, match="failed"):
            run_suite(context(workspace, base_url=url))
    finally:
        server.shutdown()
    run_dir = workspace["runs_dir"] / "eval" / "fake-model" / "fast"
    records = read_records(run_dir / RECORDS_NAME)
    assert records and all(record["error"] and not record["hypothesis"] for record in records)
    summary = json.loads((run_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["panels"][PANEL]["failed_requests"] == panel_size(workspace)
    assert summary["panels"][PANEL]["scored_items"] == 0


def test_the_url_comes_from_the_environment_variable(workspace, monkeypatch):
    calls: list = []
    server, url = server_answering(counter=calls)
    monkeypatch.setenv("FAKE_URL", url)
    try:
        run_dir = run_suite(context(workspace, base_url=""))
    finally:
        server.shutdown()
    assert len(calls) == panel_size(workspace)
    assert run_dir.is_dir()


def test_a_missing_url_variable_is_refused(workspace, monkeypatch):
    monkeypatch.delenv("FAKE_URL", raising=False)
    with pytest.raises(Exception, match="FAKE_URL"):
        run_suite(context(workspace, base_url=""))


def test_one_client_serves_the_whole_run_and_is_closed(workspace):
    """A run opens one connection pool and closes it, however many items it sends."""
    server, url = server_answering()
    made: list = []

    def factory(context_, base_url):
        client = TranscriptionClient(context_.model, base_url)
        made.append(client)
        return client

    try:
        run_suite(context(workspace, base_url=url), client_factory=factory)
    finally:
        server.shutdown()
    assert len(made) == 1
    assert made[0].client.is_closed


def test_a_refused_language_field_is_not_sent_again(workspace):
    """A server that refuses the language once is asked for it once, not per item."""
    sent_language: list[bool] = []

    def respond(request):
        asked = b'name="language"' in request.content
        sent_language.append(asked)
        if asked:
            return httpx.Response(400, json={"detail": "unknown field: language"})
        return httpx.Response(200, json={"text": "bonjour tout le monde"})

    def factory(context_, base_url):
        return TranscriptionClient(context_.model, base_url, transport=httpx.MockTransport(respond))

    run_dir = run_suite(context(workspace, base_url="http://server.test"), client_factory=factory)
    assert sent_language.count(True) == 1
    assert sent_language.count(False) == panel_size(workspace)
    records = read_records(run_dir / RECORDS_NAME)
    assert not any(record["language_sent"] for record in records)


def test_a_retry_is_recorded_with_the_attempt_count(workspace):
    """Latency is the successful request only, and the attempts are visible."""
    attempts: list = []

    def respond(headers):
        attempts.append(1)
        if len(attempts) == 1:
            return (500, "boom", "text/plain")
        return (200, json.dumps({"text": "bonjour tout le monde"}), "application/json")

    server, url = start_transcription_server(respond)
    client = TranscriptionClient(workspace["model"], url, sleep=lambda _: None, transport=None)
    try:
        result = client.transcribe(b"wav", "fr")
    finally:
        client.close()
        server.shutdown()
    assert result.text == "bonjour tout le monde"
    assert result.attempts == 2
    assert result.latency_s < 1.0  # the backoff sleep is not part of the answer


def test_a_failed_item_is_sent_again_on_the_next_run(workspace):
    """An error record is not a measurement, so the item gets another chance."""
    state = {"fail": True}

    def respond(request):
        if state["fail"]:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"text": "bonjour tout le monde"})

    def factory(context_, base_url):
        return TranscriptionClient(
            context_.model,
            base_url,
            transport=httpx.MockTransport(respond),
            sleep=lambda _: None,
        )

    with pytest.raises(RunError, match="failed"):
        run_suite(context(workspace, base_url="http://server.test"), client_factory=factory)
    state["fail"] = False
    run_dir = run_suite(context(workspace, base_url="http://server.test"), client_factory=factory)
    records = read_records(run_dir / RECORDS_NAME)
    assert len(records) > panel_size(workspace)  # the failures stay visible
    latest = latest_records(records)
    assert len(latest) == panel_size(workspace)
    assert not [record for record in latest.values() if record["error"]]


def test_a_changed_identity_names_the_field_that_changed(workspace):
    server, url = server_answering()
    try:
        run_suite(context(workspace, base_url=url))
        changed = context(workspace, base_url=url)
        changed.model = ModelConfig(name="fake-model", model="another-id")
        with pytest.raises(RunError, match="panels|model"):
            run_suite(changed)
    finally:
        server.shutdown()
    message = None
    try:
        run_suite(changed)
    except RunError as failure:
        message = str(failure)
    assert message is not None
    assert "model" in message


def test_scoring_twice_writes_the_same_bytes(workspace):
    server, url = server_answering()
    try:
        run_dir = run_suite(context(workspace, base_url=url))
    finally:
        server.shutdown()
    first = (run_dir / SUMMARY_NAME).read_bytes()
    score_run(run_dir, workspace["panels_dir"], workspace["data_dir"], draws=200)
    assert (run_dir / SUMMARY_NAME).read_bytes() == first


def test_the_sealed_panel_needs_the_flag(workspace):
    workspace["suite"] = SuiteSpec(name="fast", panels=["fake-fr-sealed"])
    server, url = server_answering()
    try:
        with pytest.raises(RunError, match="sealed"):
            run_suite(context(workspace, base_url=url))
        run_dir = run_suite(context(workspace, base_url=url, sealed=True))
    finally:
        server.shutdown()
    assert read_records(run_dir / RECORDS_NAME)


def _run_model(workspace, url: str, name: str) -> Path:
    """Run the same suite under a second model name, for the comparison tests."""
    changed = context(workspace, base_url=url)
    changed.model = ModelConfig(name=name, model="served-id")
    return run_suite(changed)


def test_compare_two_models_pairs_the_same_ids(workspace):
    server, url = server_answering(text="bonjour tout le monde")
    try:
        good = _run_model(workspace, url, "good-model")
    finally:
        server.shutdown()
    server, url = server_answering(text="bonjour tout le monde et aussi")
    try:
        bad = _run_model(workspace, url, "bad-model")
    finally:
        server.shutdown()
    comparison = compare_runs(good, bad, workspace["panels_dir"], workspace["data_dir"], draws=200)
    section = comparison["panels"][PANEL]
    assert section["items"] == panel_size(workspace)
    assert section["kind"] == "speech"
    assert section["b"] > section["a"]
    assert section["verdict"] == "good-model better"


def test_compare_refuses_runs_over_different_panels(workspace):
    server, url = server_answering()
    try:
        one_panel = _run_model(workspace, url, "one-model")
        workspace["suite"] = SuiteSpec(name="other", panels=["fake-fr-full", "fake-fr-fast"])
        two_panels = _run_model(workspace, url, "two-model")
        with pytest.raises(ValueError, match="different panels"):
            compare_runs(
                one_panel, two_panels, workspace["panels_dir"], workspace["data_dir"], draws=50
            )
    finally:
        server.shutdown()


def test_compare_refuses_a_run_with_no_shared_ids(workspace, tmp_path):
    server, url = server_answering()
    try:
        first = _run_model(workspace, url, "good-model")
    finally:
        server.shutdown()
    empty = tmp_path / "empty-run"
    empty.mkdir()
    (empty / IDENTITY_NAME).write_text(
        (first / IDENTITY_NAME).read_text(encoding="utf-8"), encoding="utf-8"
    )
    (empty / RECORDS_NAME).write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="share no ids"):
        compare_runs(first, empty, workspace["panels_dir"], workspace["data_dir"], draws=50)


def test_a_non_speech_panel_is_scored_for_hallucinations(tmp_path):
    """The battery panel goes through the same path, and only counts invented words."""
    data_dir = tmp_path / "datasets"
    clips = []
    for index in range(4):
        clip = make_clip(f"ns:silence-{index:03d}", language="none", category="silence")
        write_tone(data_dir / "nonspeech" / clip.audio, seconds=1.0)
        clips.append(clip)
    write_manifest(data_dir / "nonspeech", clips)
    panels_dir = tmp_path / "panels"
    (panels_dir).mkdir(parents=True, exist_ok=True)
    panel_text = "\n".join(
        [
            "# panel: nonspeech-full",
            "# dataset: nonspeech (4 clips in 4 groups; generated 4)",
            "# items: 4  (clip ids below, one per line)",
            f"# clips_sha256: {pcm_sha256_of(clips)}",
            "# manifest_sha256: x",
            "# seed: 1",
            "# built: 2026-11-07",
            *[clip.id for clip in clips],
            "",
        ]
    )
    (panels_dir / "nonspeech-full.txt").write_text(panel_text, encoding="utf-8")
    server, url = server_answering(text="merci d'avoir regardé")
    try:
        run_dir = run_suite(
            RunContext(
                model=ModelConfig(name="fake-model", model="served-id"),
                suite=SuiteSpec(name="fast", panels=["nonspeech-full"]),
                panels_dir=panels_dir,
                data_dir=data_dir,
                runs_root=tmp_path / "runs",
                base_url=url,
                verbose=False,
            )
        )
    finally:
        server.shutdown()
    summary = json.loads((run_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
    section = summary["panels"]["nonspeech-full"]
    assert section["kind"] == "nonspeech"
    assert section["hallucination_rate"]["value"] == 1.0
    assert section["hallucination_rate"]["ci"]["low"] == 1.0
    assert section["top_hypotheses"][0]["count"] == 4
    assert section["by_category"]["silence"]["items"] == 4
    assert "| silence | 4 | 100.00 %" in (run_dir / REPORT_NAME).read_text(encoding="utf-8")
    records = read_records(run_dir / RECORDS_NAME)
    assert all(record["language_sent"] is False for record in records)
