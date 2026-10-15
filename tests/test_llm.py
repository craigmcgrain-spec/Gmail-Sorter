import json

import httpx
import pytest

from gmail_sort.config import ConfigError
from gmail_sort.llm import (AnthropicClassifier, OpenAIClassifier, estimate,
                            make_classifier, sanitize)
from tests.helpers import cfg, row

LABELS = ["Newsletters", "Receipts"]


def anthropic(handler):
    return AnthropicClassifier(
        httpx.Client(transport=httpx.MockTransport(handler)), "key",
        "claude-haiku-4-5")


def openai(handler):
    return OpenAIClassifier(
        httpx.Client(transport=httpx.MockTransport(handler)), "key", "gpt-4o-mini")


def reply(text):
    def handler(request):
        return httpx.Response(200, json={"content": [{"type": "text", "text": text}]})
    return handler


# sanitize -------------------------------------------------------------------

def test_sanitize_keeps_exact_matches():
    assert sanitize(["Newsletters", "Receipts"], LABELS) == ["Newsletters", "Receipts"]


def test_sanitize_rejects_an_invented_label():
    assert sanitize(["Totally New Label"], LABELS) == [None]


def test_sanitize_trims_surrounding_whitespace():
    assert sanitize(["  Receipts  "], LABELS) == ["Receipts"]


def test_sanitize_maps_unsorted_and_empty_to_none():
    assert sanitize(["UNSORTED", "", None], LABELS) == [None, None, None]


def test_sanitize_never_matches_a_substring():
    assert sanitize(["Newsletters/CI"], ["Newsletters", "Newsletters/CI"]) == ["Newsletters/CI"]
    assert sanitize(["Newsletters/CI"], ["Newsletters"]) == [None]


def test_sanitize_maps_one_to_one_without_padding():
    assert sanitize(["Newsletters"], LABELS) == ["Newsletters"]


# anthropic ------------------------------------------------------------------

def test_anthropic_posts_the_batch_and_parses_labels():
    def handler(request):
        body = json.loads(request.content)
        assert body["temperature"] == 0
        assert "Newsletters" in body["messages"][0]["content"]
        assert request.headers["x-api-key"] == "key"
        return httpx.Response(200, json={"content": [
            {"type": "text", "text": '["Newsletters", "Receipts"]'}]})

    assert anthropic(handler).classify([row("m1"), row("m2")], LABELS) == \
        ["Newsletters", "Receipts"]


def test_anthropic_retries_once_on_unparseable_output_then_gives_up():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"content": [{"type": "text", "text": "I cannot"}]})

    assert anthropic(handler).classify([row("m1")], LABELS) == [None]
    assert calls["n"] == 2


def test_anthropic_pads_a_short_response_so_labels_cannot_misalign():
    # A one-entry reply for a two-message batch must not shift onto message 2.
    assert anthropic(reply('["Receipts"]')).classify(
        [row("m1"), row("m2")], LABELS) == ["Receipts", None]


def test_anthropic_returns_none_for_an_empty_batch_without_calling_out():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={})

    assert anthropic(handler).classify([], LABELS) == []
    assert calls["n"] == 0


# openai ---------------------------------------------------------------------

def test_openai_posts_the_batch_and_parses_labels():
    def handler(request):
        body = json.loads(request.content)
        assert body["temperature"] == 0
        assert request.headers["authorization"] == "Bearer key"
        return httpx.Response(200, json={"choices": [
            {"message": {"content": '["Receipts"]'}}]})

    assert openai(handler).classify([row("m1")], LABELS) == ["Receipts"]


# a locally served model -----------------------------------------------------

LOCAL = "http://192.168.1.154:8080/v1"


def test_openai_posts_to_the_configured_base_url():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [
            {"message": {"content": '["Receipts"]'}}]})

    client = OpenAIClassifier(
        httpx.Client(transport=httpx.MockTransport(handler)), None, "Qwen3.8-2B",
        base_url=LOCAL)
    assert client.classify([row("m1")], LABELS) == ["Receipts"]
    assert seen["url"] == f"{LOCAL}/chat/completions"


def test_openai_omits_the_auth_header_when_no_key_is_configured():
    # llama-server and Unsloth Studio need no key; sending an empty bearer
    # token would be worse than sending none.
    seen = {}

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [
            {"message": {"content": '["Receipts"]'}}]})

    client = OpenAIClassifier(
        httpx.Client(transport=httpx.MockTransport(handler)), None, "Qwen3.8-2B",
        base_url=LOCAL)
    assert client.classify([row("m1")], LABELS) == ["Receipts"]
    assert seen["auth"] is None


def test_make_classifier_points_at_a_local_server_without_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    classifier = make_classifier(
        cfg(llm_provider="openai", llm_model="Qwen3.8-2B", base_url=LOCAL,
            api_key_env=""))
    assert classifier is not None


# wiring ---------------------------------------------------------------------

def test_make_classifier_names_the_missing_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        make_classifier(cfg())


def test_make_classifier_rejects_an_unknown_provider():
    with pytest.raises(ConfigError, match="provider"):
        make_classifier(cfg(llm_provider="gemini"))


def test_estimate_reports_count_and_tokens():
    count, tokens = estimate([row(f"m{i}") for i in range(10)], LABELS)
    assert count == 10
    assert tokens > 0
