import time

import pytest

from gmail_sort.gmail import TokenBucket


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.t += seconds


def bucket(rate=100.0):
    clock = FakeClock()
    return TokenBucket(rate, clock=clock.now, sleep=clock.sleep), clock


def test_take_within_budget_does_not_sleep():
    b, clock = bucket()
    b.take(50)
    b.take(50)
    assert clock.slept == []


def test_take_over_budget_sleeps_the_deficit():
    b, clock = bucket()
    b.take(100)
    b.take(50)
    assert sum(clock.slept) == pytest.approx(0.5)


def test_pacing_a_long_run_waits_exactly_the_arithmetic():
    # 200 fetches of 20 units = 4000 units; the first 100 are free, the rest
    # take (4000 - 100) / 100 = 39 seconds.
    b, clock = bucket()
    for _ in range(200):
        b.take(20)
    assert sum(clock.slept) == pytest.approx(39.0)


def test_bucket_refills_over_time():
    b, clock = bucket()
    b.take(100)
    clock.t += 1.0
    b.take(100)
    assert clock.slept == []


def test_refill_never_exceeds_capacity():
    b, clock = bucket()
    b.take(100)
    clock.t += 3600
    b.take(100)
    b.take(1)
    assert sum(clock.slept) == pytest.approx(0.01)


def test_taking_more_than_capacity_is_rejected():
    b, _ = bucket()
    with pytest.raises(ValueError, match="capacity"):
        b.take(101)


def test_the_limiter_never_touches_real_time():
    b, clock = bucket()
    start = time.monotonic()
    for _ in range(200):
        b.take(50)
    assert time.monotonic() - start < 0.1
    assert sum(clock.slept) == pytest.approx(99.0)
