"""A run end to end: a local fake endpoint, a tiny dataset, and the files it leaves."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import httpx  # noqa: E402
import pytest  # noqa: E402
from helpers import is_warmup, make_clip, start_transcription_server, write_tone  # noqa: E402

from audiolab.core.audio import pcm_sha256  # noqa: E402
from audiolab.core.stop import Stopped, request_stop  # noqa: E402
from audiolab.data.panels import SuiteSpec, build_panels, load_suite, read_panel  # noqa: E402
from audiolab.data.store import pcm_sha256_of, write_manifest  # noqa: E402
from audiolab.eval import metrics, runner, stats  # noqa: E402
from audiolab.eval.client import (  # noqa: E402
    ModelConfig,
    TranscriptionClient,
    server_root,  # noqa: E402
)
from audiolab.eval.records import (  # noqa: E402
    IDENTITY_NAME,
    RECORDS_NAME,
    read_records,
)
from audiolab.eval.report import (  # noqa: E402
    REPORT_NAME,
    SUMMARY_NAME,
    compare_runs,
    render_comparison,
    score_run,
)
from audiolab.eval.runner import (  # noqa: E402
    RunContext,
    RunError,
    identity_fingerprint,
    run_suite,
)
from audiolab.serve.config import ServeSpec  # noqa: E402
from audiolab.serve.http import TranscriptionServer  # noqa: E402

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
            # A different tone per clip, so the audio of one clip is recognised
            # in a request (the multipart boundary changes on every request).
            write_tone(
                data_dir / "fake-fr" / clip.audio, seconds=1.0, frequency=200 + speaker * 3 + index
            )
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


def cache_lines(workspace: dict) -> list[dict]:
    """Every answer the transcript cache of the fake model holds, oldest first."""
    files = sorted((workspace["runs_dir"] / "transcripts").glob("*.jsonl"))
    return [record for path in files for record in read_records(path)]


def test_a_new_panel_only_sends_the_clips_never_asked(workspace):
    """Adding a panel to a suite costs its new clips, not the whole suite again."""
    calls: list = []
    server, url = server_answering(counter=calls)
    full = "fake-fr-full"
    try:
        run_suite(context(workspace, base_url=url))
        bigger = context(workspace, base_url=url)
        bigger.suite = SuiteSpec(name="fast", panels=[PANEL, full])
        run_dir = run_suite(bigger)
    finally:
        server.shutdown()
    fast_ids = set(read_panel(workspace["panels_dir"] / f"{PANEL}.txt").ids)
    full_ids = set(read_panel(workspace["panels_dir"] / f"{full}.txt").ids)
    assert len(calls) == len(fast_ids | full_ids)
    records = read_records(run_dir / RECORDS_NAME)
    assert {record["panel"] for record in records} == {PANEL, full}
    assert len(records) == len(fast_ids) + len(full_ids)


def test_what_changes_no_answer_sends_nothing_again(workspace, monkeypatch):
    """A new commit, a new normalizer or a renamed server label keeps the answers."""
    calls: list = []
    server, url = server_answering(counter=calls)
    try:
        run_suite(context(workspace, base_url=url))
        monkeypatch.setattr(runner, "git_commit", lambda: "another-commit")
        monkeypatch.setattr(runner, "NORMALIZER_VERSION", 99)
        relabelled = context(workspace, base_url=url)
        relabelled.model = ModelConfig(
            name="renamed-config",
            model="served-id",
            base_url_env="FAKE_URL",
            api_key_env="FAKE_KEY",
            server_label="the same server, renamed",
        )
        run_dir = run_suite(relabelled)
    finally:
        server.shutdown()
    assert len(calls) == panel_size(workspace)
    identity = json.loads((run_dir / IDENTITY_NAME).read_text(encoding="utf-8"))
    assert identity["git_commit"] == "another-commit"  # the run says what scored it
    assert identity["normalizer_version"] == 99


def test_what_changes_an_answer_asks_again(workspace):
    """Another served model, or new audio under the same id, is a new question."""
    calls: list = []
    server, url = server_answering(counter=calls)
    try:
        run_suite(context(workspace, base_url=url))
        other = context(workspace, base_url=url)
        other.model = ModelConfig(name="fake-model", model="another-id")
        run_suite(other)
        assert len(calls) == 2 * panel_size(workspace)

        # New audio for one clip: only that clip is sent again.
        first = read_panel(workspace["panels_dir"] / f"{PANEL}.txt").ids[0]
        manifest_path = workspace["data_dir"] / "fake-fr" / "manifest.jsonl"
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
        row = next(row for row in rows if row["id"] == first)
        audio = workspace["data_dir"] / "fake-fr" / row["audio"]
        write_tone(audio, seconds=1.0, frequency=999)
        row["sha256"] = pcm_sha256(audio)
        manifest_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        run_suite(context(workspace, base_url=url))
    finally:
        server.shutdown()
    assert len(calls) == 2 * panel_size(workspace) + 1


def test_a_changed_language_asks_again(workspace):
    """The language is part of the request, so a relabelled clip is a new question."""
    calls: list = []
    server, url = server_answering(counter=calls)
    try:
        run_suite(context(workspace, base_url=url))
        manifest_path = workspace["data_dir"] / "fake-fr" / "manifest.jsonl"
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
        manifest_path.write_text(
            "".join(json.dumps({**row, "language": "en"}, sort_keys=True) + "\n" for row in rows)
        )
        run_suite(context(workspace, base_url=url))
    finally:
        server.shutdown()
    assert len(calls) == 2 * panel_size(workspace)


def test_a_torn_last_line_costs_one_clip_not_the_cache(workspace, capsys):
    """A crash mid-write leaves half a line; the next run asks that clip again."""
    calls: list = []
    server, url = server_answering(counter=calls)
    try:
        run_suite(context(workspace, base_url=url))
        (cache_path,) = (workspace["runs_dir"] / "transcripts").glob("*.jsonl")
        data = cache_path.read_bytes()
        cache_path.write_bytes(data[: len(data) - 40])
        run_dir = run_suite(context(workspace, base_url=url, verbose=True))
    finally:
        server.shutdown()
    assert len(calls) == panel_size(workspace) + 1
    assert "is not a complete answer" in capsys.readouterr().out
    assert len(read_records(run_dir / RECORDS_NAME)) == panel_size(workspace)
    # The new answer was not glued onto the torn line: every other line parses.
    lines = cache_path.read_text(encoding="utf-8").splitlines()
    assert sum(1 for line in lines if not line.endswith("}")) == 1


def test_answers_given_without_a_language_are_counted(workspace):
    def respond(request):
        if b'name="language"' in request.content:
            return httpx.Response(400, json={"detail": "unknown field: language"})
        return httpx.Response(200, json={"text": "bonjour tout le monde"})

    def factory(context_, base_url):
        return TranscriptionClient(context_.model, base_url, transport=httpx.MockTransport(respond))

    run_dir = run_suite(context(workspace, base_url="http://server.test"), client_factory=factory)
    summary = json.loads((run_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["panels"][PANEL]["answered_without_language"] == panel_size(workspace)
    assert "without a language hint" in (run_dir / REPORT_NAME).read_text(encoding="utf-8")


def test_a_run_sends_its_requests_n_at_a_time(workspace):
    together = threading.Barrier(3, timeout=10)  # three in flight, or none is answered
    sent: list[int] = []
    lock = threading.Lock()

    def respond(request):
        if is_warmup(request.content):
            return httpx.Response(200, json={"text": ""})
        with lock:
            sent.append(len(sent))
            first_three = len(sent) <= 3
        if first_three:
            together.wait()
        return httpx.Response(200, json={"text": "bonjour tout le monde"})

    def factory(context_, base_url):
        return TranscriptionClient(context_.model, base_url, transport=httpx.MockTransport(respond))

    run_dir = run_suite(
        context(workspace, base_url="http://server.test", concurrency=3), client_factory=factory
    )
    records = read_records(run_dir / RECORDS_NAME)
    assert len(sent) == len(records) == panel_size(workspace)
    assert all(record["hypothesis"] == "bonjour tout le monde" for record in records)


def test_a_stopped_run_sends_no_further_clip_writes_no_run_and_the_next_one_sends_the_rest(
    workspace, run_again
):
    together = threading.Barrier(2, timeout=10)  # both in flight when the stop comes
    sent: list[int] = []
    lock = threading.Lock()

    def respond(request):
        if is_warmup(request.content):
            return httpx.Response(200, json={"text": ""})
        with lock:
            sent.append(len(sent))
            if len(sent) == 2:
                request_stop()  # Ctrl-C while the first two clips are answered
        if len(sent) <= 2:
            together.wait()
        return httpx.Response(200, json={"text": "bonjour tout le monde"})

    def factory(context_, base_url):
        return TranscriptionClient(context_.model, base_url, transport=httpx.MockTransport(respond))

    def evaluate() -> Path:
        return run_suite(
            context(workspace, base_url="http://server.test", concurrency=2), client_factory=factory
        )

    with pytest.raises(Stopped, match="evaluation stopped"):
        evaluate()
    assert len(sent) == 2, "the clips in flight finish, no other is sent"
    assert len(cache_lines(workspace)) == 2, "their answers are kept"
    assert not list(workspace["runs_dir"].rglob(SUMMARY_NAME)), "no run is written as complete"

    run_again()
    run_dir = evaluate()
    assert len(sent) == panel_size(workspace), "the next run sends only the rest"
    assert len(read_records(run_dir / RECORDS_NAME)) == panel_size(workspace)


def test_a_run_warms_its_model_up_and_neither_records_nor_caches_the_warmup(workspace):
    sent: list[bool] = []

    def slow_to_start(request):
        sent.append(is_warmup(request.content))
        if sent.count(True) < 3:
            return httpx.Response(503, text="loading")
        return httpx.Response(200, json={"text": "bonjour tout le monde"})

    def factory(context_, base_url):
        transport = httpx.MockTransport(slow_to_start)
        return TranscriptionClient(context_.model, base_url, transport=transport, sleep=lambda _: 0)

    run_dir = run_suite(context(workspace, base_url="http://server.test"), client_factory=factory)
    clips = panel_size(workspace)
    assert sent == [True] * 3 + [False] * clips, "three warmups, then the clips"
    records = read_records(run_dir / RECORDS_NAME)
    assert len(records) == len(cache_lines(workspace)) == clips
    assert all(record["attempts"] == 1 and not record["error"] for record in records), (
        "the first clip finds the model ready"
    )

    sent.clear()
    run_suite(context(workspace, base_url="http://server.test"), client_factory=factory)
    assert sent == [], "a run with nothing to send warms nothing up"


def test_fresh_sends_every_clip_again(workspace):
    calls: list = []
    server, url = server_answering(counter=calls)
    try:
        first = run_suite(context(workspace, base_url=url))
        second = run_suite(context(workspace, base_url=url, fresh=True))
    finally:
        server.shutdown()
    assert first == second
    assert len(calls) == 2 * panel_size(workspace)
    assert len(cache_lines(workspace)) == 2 * panel_size(workspace)  # nothing is lost
    assert len(read_records(second / RECORDS_NAME)) == panel_size(workspace)


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
        if is_warmup(request.content):
            return httpx.Response(200, json={"text": ""})
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


def test_an_api_key_echoed_by_the_server_never_reaches_a_record(workspace):
    """A record is a file people read and share, so no secret may end up in it.

    Some servers answer an error page that repeats the request headers; the
    client takes the key out of any message it builds, whichever way it leaks.
    """

    def echo(headers):
        body = {"detail": f"rejected {headers.get('Authorization', 'none')}"}
        return (401, json.dumps(body), "application/json")

    server, url = start_transcription_server(echo)
    try:
        with pytest.raises(RunError):
            run_suite(context(workspace, base_url=url, api_key="sk-not-a-real-key"))
    finally:
        server.shutdown()
    run_dir = next((workspace["runs_dir"] / "eval" / "fake-model").iterdir())
    body = (run_dir / RECORDS_NAME).read_text(encoding="utf-8")
    assert "sk-not-a-real-key" not in body
    assert "rejected Bearer <redacted api key>" in body


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
        if state["fail"] and not is_warmup(request.content):
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
    assert len(cache_lines(workspace)) == 2 * panel_size(workspace)  # the failures stay visible
    records = read_records(run_dir / RECORDS_NAME)
    assert len(records) == panel_size(workspace)
    assert not [record for record in records if record["error"]]


def test_another_server_gets_its_own_answers(workspace):
    """The same model on two servers is two experiments: answers never mix.

    The server enters the cache key as a keyed fingerprint, so neither its URL
    nor the fingerprint appears in a run directory.
    """
    calls_a: list = []
    calls_b: list = []
    server_a, url_a = server_answering(counter=calls_a)
    server_b, url_b = server_answering(text="autre chose", counter=calls_b)
    try:
        run_suite(context(workspace, base_url=url_a))
        run_dir = run_suite(context(workspace, base_url=url_b))
        assert len(calls_b) == panel_size(workspace)
        records = read_records(run_dir / RECORDS_NAME)
        assert {record["hypothesis"] for record in records} == {"autre chose"}
        # The same server written two ways is the same server.
        run_suite(context(workspace, base_url=f"{url_a}/v1"))
    finally:
        server_a.shutdown()
        server_b.shutdown()
    assert len(calls_a) == panel_size(workspace)
    for name in (IDENTITY_NAME, RECORDS_NAME, SUMMARY_NAME, REPORT_NAME):
        text = (run_dir / name).read_text(encoding="utf-8")
        assert url_a not in text and url_b not in text and "endpoint" not in text


def test_the_endpoint_fingerprint_is_keyed_and_stays_in_the_cache(workspace):
    """The fingerprint is compared, never published, and the key is local."""
    server, url = server_answering()
    try:
        run_dir = run_suite(context(workspace, base_url=url))
    finally:
        server.shutdown()
    key_files = list((workspace["runs_dir"] / "transcripts").glob("*.key.json"))
    assert len(key_files) == 1
    fingerprint = json.loads(key_files[0].read_text(encoding="utf-8"))["endpoint_fingerprint"]
    assert len(fingerprint) == 64
    for name in (IDENTITY_NAME, SUMMARY_NAME, REPORT_NAME):
        assert fingerprint not in (run_dir / name).read_text(encoding="utf-8")

    key_path = workspace["runs_dir"] / ".identity-key"
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    key = bytes.fromhex(key_path.read_text(encoding="utf-8").strip())
    assert len(key) == 32
    # The same URL under another key is a different fingerprint, and the same
    # key gives the same one: nothing about the URL can be read off the value.
    assert identity_fingerprint(server_root(url), key) == fingerprint
    assert identity_fingerprint(server_root(url), bytes(32)) != fingerprint


def test_a_panel_with_few_groups_says_its_interval_is_approximate(workspace):
    """The bootstrap resamples groups, so four speakers give a poor interval.

    The fake panel has six speakers and the report must say so; the same panel
    seen with a threshold of exactly that many groups must not, which pins the
    rule to the group count rather than to the panel name.
    """
    server, url = server_answering()
    try:
        run_dir = run_suite(context(workspace, base_url=url))
    finally:
        server.shutdown()
    report = (run_dir / REPORT_NAME).read_text(encoding="utf-8")
    assert "few groups (n < 20): interval approximate" in report
    summary = json.loads((run_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
    groups = summary["panels"][PANEL]["groups"]
    assert 1 <= groups < 20
    assert summary["panels"][PANEL]["interval_caution"]

    # A panel with enough groups gets no caution, at the same numbers.
    monkey = pytest.MonkeyPatch()
    monkey.setattr(stats, "MINIMUM_GROUPS_FOR_INTERVALS", groups)
    try:
        again = score_run(run_dir, workspace["panels_dir"], workspace["data_dir"], draws=50)
    finally:
        monkey.undo()
    assert "interval_caution" not in again["panels"][PANEL]
    assert "interval approximate" not in (run_dir / REPORT_NAME).read_text(encoding="utf-8")


def test_a_hypothesis_that_cannot_be_scored_is_counted_in_the_report(workspace):
    """A scoring error costs one utterance, not the panel.

    The reader is total (tests/test_eval_normalize.py fuzzes it), so this breaks
    it on purpose: the run has to finish, the panel has to keep the utterances
    that scored, and the report has to say how many it lost and which.
    """
    server, url = server_answering()
    try:
        run_dir = run_suite(context(workspace, base_url=url))
    finally:
        server.shutdown()

    original = metrics._counts_for
    broken_id = "fake:000-000"

    def counts_or_fails(item):
        if item.id == broken_id:
            raise KeyError("'°'")
        return original(item)

    monkey = pytest.MonkeyPatch()
    monkey.setattr(metrics, "_counts_for", counts_or_fails)
    try:
        summary = score_run(run_dir, workspace["panels_dir"], workspace["data_dir"], draws=50)
    finally:
        monkey.undo()
    section = summary["panels"][PANEL]
    assert section["scoring_errors"]["count"] == 1
    assert section["scoring_errors"]["items"][0]["id"] == broken_id
    assert "KeyError" in section["scoring_errors"]["items"][0]["error"]
    assert section["scored_items"] == section["items"] - 1
    assert section["wer"]["normalized"]["utterances"] == section["items"] - 1
    report = (run_dir / REPORT_NAME).read_text(encoding="utf-8")
    assert f"**1 hypothesis could not be scored**: `{broken_id}`" in report

    # Nothing is broken when every hypothesis scores.
    clean = score_run(run_dir, workspace["panels_dir"], workspace["data_dir"], draws=50)
    assert clean["panels"][PANEL]["scoring_errors"] == {"count": 0, "items": []}
    assert "could not be scored" not in (run_dir / REPORT_NAME).read_text(encoding="utf-8")


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


def _run_model(workspace, url: str, name: str, factory=None) -> Path:
    """Run the same suite under a second model name, for the comparison tests."""
    changed = context(workspace, base_url=url)
    changed.model = ModelConfig(name=name, model="served-id")
    return run_suite(changed, client_factory=factory)


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
    # The pairing uses the ids both runs answered, which is what the header says:
    # a recorded id whose request failed is not in the comparison.
    assert "Paired over the ids both runs answered" in render_comparison(comparison)


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
    with pytest.raises(ValueError, match="share no answered ids"):
        compare_runs(first, empty, workspace["panels_dir"], workspace["data_dir"], draws=50)


def test_the_report_says_when_a_panel_cost_more_requests_than_clips(workspace):
    """One retry is invisible in a latency median; the request count is not."""
    state = {"first": True}

    def respond(request):
        if state["first"] and not is_warmup(request.content):
            state["first"] = False
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"text": "bonjour tout le monde"})

    def factory(context_, base_url):
        return TranscriptionClient(
            context_.model,
            base_url,
            transport=httpx.MockTransport(respond),
            sleep=lambda _: None,
        )

    run_dir = run_suite(context(workspace, base_url="http://server.test"), client_factory=factory)
    report = (run_dir / REPORT_NAME).read_text(encoding="utf-8")
    items = panel_size(workspace)
    # One retry, so one more request than the panel has clips. The retried clip
    # is the first one of the run: cold, so out of the latency, in the count.
    assert f"{items + 1} requests in total" in report
    assert f"{items - 1} warm calls, 1 cold excluded" in report


def test_a_panel_with_no_reference_text_reports_no_wer(tmp_path):
    """References are the only source of WER; without them the report says n/a."""
    data_dir = tmp_path / "datasets"
    clips = [
        make_clip(f"empty:{index:03d}-000", group=f"empty:spk{index}", text="")
        for index in range(4)
    ]
    for clip in clips:
        write_tone(data_dir / "empty-fr" / clip.audio, seconds=1.0)
    write_manifest(data_dir / "empty-fr", clips)
    panels_dir = tmp_path / "panels"
    build_panels("empty-fr", clips, panels_dir, built="2026-11-07")
    server, url = server_answering(text="bonjour tout le monde")
    try:
        run_dir = run_suite(
            RunContext(
                model=ModelConfig(name="fake-model", model="served-id"),
                suite=SuiteSpec(name="full", panels=["empty-fr-full"]),
                panels_dir=panels_dir,
                data_dir=data_dir,
                runs_root=tmp_path / "runs",
                base_url=url,
                verbose=False,
            )
        )
    finally:
        server.shutdown()
    report = (run_dir / REPORT_NAME).read_text(encoding="utf-8")
    assert "| normalized | n/a |" in report
    summary = json.loads((run_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
    wer = summary["panels"]["empty-fr-full"]["wer"]["normalized"]
    assert wer["value"] is None
    assert wer["reference_words"] == 0
    # "bonjour tout le monde" is four words, invented again in every clip the
    # panel holds (the sealed panel took one of the four groups first).
    assert wer["errors"] == 4 * summary["panels"]["empty-fr-full"]["items"]


def test_compare_leaves_out_the_ids_a_run_failed_on(workspace, tmp_path):
    """A model that fails a third of the panel must not be compared as if it did not.

    The failures of run B are excluded from the paired difference (they have no
    hypothesis to pair with), and every excluded id is listed in the comparison.
    """
    state = {"failing": False, "order": []}

    def respond(request):
        if is_warmup(request.content):
            return httpx.Response(200, json={"text": ""})
        # The clip, not the request: the multipart boundary differs every time.
        content = request.content
        start = content.index(b"RIFF")
        key = hashlib.sha256(content[start : start + 4096]).hexdigest()
        if key not in state["order"]:
            state["order"].append(key)
        # Every third clip of the panel fails, retries included, so the failing
        # set is the same whatever order the requests arrive in.
        if state["failing"] and state["order"].index(key) % 3 == 0:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(200, json={"text": "bonjour tout le monde"})

    def factory(context_, base_url):
        return TranscriptionClient(
            context_.model,
            base_url,
            transport=httpx.MockTransport(respond),
            sleep=lambda _: None,
        )

    url = "http://server.test"
    run_a = _run_model(workspace, url, "model-a", factory)
    state["failing"] = True
    run_b_root = tmp_path / "runs-b"
    with pytest.raises(RunError):
        run_suite(
            context(
                workspace,
                base_url=url,
                runs_root=run_b_root,
                model=ModelConfig(name="model-b", model="served-id"),
            ),
            client_factory=factory,
        )
    comparison = compare_runs(
        run_a,
        run_b_root / "eval" / "model-b" / "fast",
        workspace["panels_dir"],
        workspace["data_dir"],
        draws=50,
    )
    section = comparison["panels"][PANEL]
    failed = section["excluded"]["failed_in_b"]
    assert failed
    assert section["items"] == panel_size(workspace) - len(failed)
    for clip_id in failed:
        assert clip_id in section["excluded"]["ids"]

    text = render_comparison(comparison)
    assert "ids left out" in text
    assert failed[0] in text


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


def served_context(workspace, serve_spec, runs_root, name: str = "local-model"):
    """A run context whose model is a checkpoint served here, not a remote server."""
    changed = context(workspace, base_url="", runs_root=runs_root)
    changed.model = ModelConfig(name=name, model="parakeet-tdt-0.6b-v3", serve=serve_spec)
    return changed


def fake_serving(calls: list, text: str = "bonjour tout le monde"):
    """Serve a stand-in backend, on a new port for every run.

    The runner starts a server in a child process in real use; the same HTTP
    contract is served in-process here, so the test needs no NeMo and no model.
    A fresh port per run is the point of the test that uses it: a cache key that
    followed the URL would send every clip again.
    """

    def transcribe(audio: bytes, language: str = "") -> str:
        if not is_warmup(audio):
            calls.append((audio, language))
        return text

    @contextmanager
    def serving(spec, model_id):
        server = TranscriptionServer(transcribe, model_id, identity=spec.identity()).start()
        try:
            yield server.base_url
        finally:
            server.stop()

    return serving


def test_a_served_checkpoint_is_identified_by_its_content_not_its_port(
    workspace, tmp_path, monkeypatch
):
    """Two runs of one checkpoint share their answers; another checkpoint does not.

    A temporary server listens on a fresh port on every run, so a cache key that
    followed the URL would ask the same checkpoint again for every clip of every
    run (D-017). The identity of a served model is its checkpoint instead, and a
    copy of that file is the same checkpoint.
    """
    checkpoint = tmp_path / "run-a.nemo"
    checkpoint.write_bytes(b"first-weights")
    copy = tmp_path / "copied.nemo"
    copy.write_bytes(checkpoint.read_bytes())
    other = tmp_path / "run-b.nemo"
    other.write_bytes(b"second-weights")
    runs_root = tmp_path / "runs"
    calls: list = []
    monkeypatch.setattr(runner, "serving", fake_serving(calls))

    first = run_suite(served_context(workspace, ServeSpec(checkpoint=str(checkpoint)), runs_root))
    assert len(calls) == panel_size(workspace)

    # The same weights, another path and another port: nothing is sent again.
    run_suite(served_context(workspace, ServeSpec(checkpoint=str(copy)), runs_root))
    assert len(calls) == panel_size(workspace)

    # Another checkpoint, on another port too: every clip is a new question.
    run_suite(served_context(workspace, ServeSpec(checkpoint=str(other)), runs_root))
    assert len(calls) == 2 * panel_size(workspace)

    # The identity, not the port: one cache file for each checkpoint, and no run
    # directory names a path.
    assert len(list((runs_root / "transcripts").glob("*.jsonl"))) == 2
    identity = (first / IDENTITY_NAME).read_text(encoding="utf-8")
    assert str(tmp_path) not in identity
    assert json.loads(identity)["model"]["serve"]["checkpoint"].startswith("sha256=")


def test_serving_a_checkpoint_uses_the_port_the_server_chose(workspace, tmp_path, monkeypatch):
    """The run talks to the server the runner started, not to a URL from a variable."""
    checkpoint = tmp_path / "one.nemo"
    checkpoint.write_bytes(b"weights")
    calls: list = []
    monkeypatch.setattr(runner, "serving", fake_serving(calls))
    changed = served_context(workspace, ServeSpec(checkpoint=str(checkpoint)), tmp_path / "runs")
    # The config names a variable that is not set: a run that needed it would fail.
    assert not os.environ.get("FAKE_URL")
    run_dir = run_suite(changed)
    assert len(calls) == panel_size(workspace)
    identity = (run_dir / IDENTITY_NAME).read_text(encoding="utf-8")
    # The served address is a temporary local port; it belongs in no record.
    assert "127.0.0.1" not in identity and "http://" not in identity


def test_a_served_run_that_fails_still_stops_its_server(workspace, tmp_path, monkeypatch):
    """A server left behind would keep a GPU busy; the runner must always stop it."""
    checkpoint = tmp_path / "broken.nemo"
    checkpoint.write_bytes(b"weights")
    used: list[str] = []

    def failing(audio: bytes, language: str = "") -> str:
        raise RuntimeError("no GPU memory left")

    @contextmanager
    def serving(spec, model_id):
        server = TranscriptionServer(failing, model_id, identity=spec.identity()).start()
        used.append(server.base_url)
        try:
            yield server.base_url
        finally:
            server.stop()

    monkeypatch.setattr(runner, "serving", serving)

    def no_backoff(context_, base_url):
        return TranscriptionClient(context_.model, base_url, sleep=lambda _: None)

    changed = served_context(workspace, ServeSpec(checkpoint=str(checkpoint)), tmp_path / "runs")
    with pytest.raises(RunError, match="no answer"):
        run_suite(changed, client_factory=no_backoff)
    # The failures are recorded, and the port the run used answers nothing now.
    assert read_records(changed.runs_root / "eval" / "local-model" / "fast" / RECORDS_NAME)
    assert len(used) == 1
    with pytest.raises(httpx.TransportError):
        httpx.get(f"{used[0]}/v1/models", timeout=2)


def test_a_serve_config_pointed_at_a_server_belongs_to_that_server(workspace, tmp_path):
    """With ``--base-url`` the run talks to an endpoint, not to the checkpoint in its config.

    A run that was pointed somewhere else must key its answers on the address it
    used, must not describe a checkpoint it never loaded - and must not need that
    file to exist.
    """
    never_served = tmp_path / "never-served.nemo"
    runs_root = tmp_path / "runs"
    calls: list = []

    def transcribe(audio: bytes, language: str = "") -> str:
        if not is_warmup(audio):
            calls.append(audio)
        return "bonjour tout le monde"

    first = TranscriptionServer(transcribe, "parakeet-tdt-0.6b-v3").start()
    second = TranscriptionServer(transcribe, "parakeet-tdt-0.6b-v3").start()
    try:
        spec = ServeSpec(checkpoint=str(never_served))

        def endpoint(base_url: str):
            changed = served_context(workspace, spec, runs_root, name="endpoint")
            changed.base_url = base_url
            return changed

        run_dir = run_suite(endpoint(first.base_url))
        assert len(calls) == panel_size(workspace)

        # The same address is the same endpoint: nothing is sent twice.
        run_suite(endpoint(first.base_url))
        assert len(calls) == panel_size(workspace)

        # Another address is another endpoint: every clip is a new question.
        run_suite(endpoint(second.base_url))
        assert len(calls) == 2 * panel_size(workspace)
    finally:
        first.stop()
        second.stop()

    assert len(list((runs_root / "transcripts").glob("*.jsonl"))) == 2
    identity = json.loads((run_dir / IDENTITY_NAME).read_text(encoding="utf-8"))
    assert identity["model"]["serve"] is None


def test_a_server_that_reports_another_checkpoint_is_refused(workspace, tmp_path, monkeypatch):
    """A server holding other weights would poison the cache; the run stops first."""
    checkpoint = tmp_path / "asked-for.nemo"
    checkpoint.write_bytes(b"the weights the config asks for")
    calls: list = []

    @contextmanager
    def lying_serving(spec, model_id):
        server = TranscriptionServer(
            lambda audio, language="": calls.append(audio) or "bonjour tout le monde",
            model_id,
            # A checkpoint that is not the one the config asks for.
            identity={**spec.identity(), "checkpoint": "sha256=" + "0" * 64},
        ).start()
        try:
            yield server.base_url
        finally:
            server.stop()

    monkeypatch.setattr(runner, "serving", lying_serving)
    changed = served_context(workspace, ServeSpec(checkpoint=str(checkpoint)), tmp_path / "runs")
    with pytest.raises(RunError, match="as this config asks"):
        run_suite(changed)
    assert calls == []


def test_a_server_that_reports_no_identity_is_refused(workspace, tmp_path, monkeypatch):
    """An answer that cannot be attributed to a checkpoint is not worth keeping."""
    checkpoint = tmp_path / "asked-for.nemo"
    checkpoint.write_bytes(b"the weights the config asks for")

    @contextmanager
    def silent_serving(spec, model_id):
        def silent(audio: bytes, language: str = "") -> str:
            return "bonjour tout le monde"

        server = TranscriptionServer(silent, model_id).start()
        try:
            yield server.base_url
        finally:
            server.stop()

    monkeypatch.setattr(runner, "serving", silent_serving)
    changed = served_context(workspace, ServeSpec(checkpoint=str(checkpoint)), tmp_path / "runs")
    with pytest.raises(RunError, match="reports no identity"):
        run_suite(changed)
