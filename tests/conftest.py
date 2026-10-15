"""LLM stand-ins. `cfg()` and friends live in tests/helpers.py."""

import pytest


class RuleLLM:
    """Answers UNSORTED for everything and counts how often it was asked."""

    def __init__(self):
        self.calls = 0

    def classify(self, batch, labels):
        self.calls += 1
        return [None] * len(batch)


class QueueLLM:
    """Returns labels from a queue, one per message, counting calls."""

    def __init__(self, labels):
        self.queue = list(labels)
        self.calls = 0

    def classify(self, batch, labels):
        self.calls += 1
        return [self.queue.pop(0) if self.queue else None for _ in batch]


@pytest.fixture
def rule_llm():
    return RuleLLM()


@pytest.fixture
def llm_returning():
    def make(labels):
        return QueueLLM(labels)
    return make
