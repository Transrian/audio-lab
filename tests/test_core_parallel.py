"""Sending N requests at a time (D-022), in the items' order, and stopping cleanly."""

from __future__ import annotations

import threading
import time

import pytest

from audiolab.core.parallel import map_requests
from audiolab.core.stop import Stopped, request_stop
from audiolab.eval.client import TranscriptionError


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


def test_a_stop_while_three_are_in_flight_returns_the_three_and_starts_no_other():
    in_flight = threading.Barrier(3, timeout=10)
    stop = threading.Event()
    started: list[int] = []

    def ask(number: int) -> int:
        started.append(number)
        in_flight.wait()  # 0, 1 and 2 are all in flight
        if number == 1:
            stop.set()
        else:
            assert stop.wait(timeout=10), "0 and 2 finish after the stop"
        return number

    assert map_requests(ask, range(10), concurrency=3, should_stop=stop.is_set) == [0, 1, 2]
    assert sorted(started) == [0, 1, 2]


def test_the_stop_it_watches_by_default_is_the_stop_of_the_process():
    def ask(number: int) -> int:
        if number == 1:
            request_stop()
        return number

    assert map_requests(ask, range(10)) == [0, 1]


def test_the_first_failure_is_raised_not_a_later_one_it_caused():
    failing = threading.Event()

    def ask(number: int) -> int:
        if number == 1:
            raise TranscriptionError("HTTP 500")
        assert failing.wait(timeout=10)
        raise Stopped("ended early because of the failure")

    with pytest.raises(TranscriptionError, match="HTTP 500"):
        map_requests(ask, range(2), concurrency=2, failing=failing)


def test_a_concurrency_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        map_requests(str, [1], concurrency=0)
