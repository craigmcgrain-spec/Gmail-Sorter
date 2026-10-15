"""The provider-swappable LLM fallback, reached only by messages no rule matched.

`sanitize` is the safety boundary and is deliberately strict: a reply that is not
exactly one of the user's approved labels becomes `UNSORTED`. This program must
never create or apply a label the user did not approve.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Protocol, Sequence

import httpx

from gmail_sort.config import Config, ConfigError
from gmail_sort.store import MessageRow

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

MAX_TOKENS = 4096
# Rough characters-per-token used only for the --estimate report.
CHARS_PER_TOKEN = 4

PROMPT_TEMPLATE = """Sort each email into exactly one existing label.

Available labels:
{labels}

Reply with a JSON array of strings holding one entry per message, in order.
Each entry must be exactly one of the labels above, or "UNSORTED" if none fit.
Reply with the JSON array only, with no commentary.

Messages:
{messages}"""


class Classifier(Protocol):
    def classify(
        self, batch: Sequence[MessageRow], labels: Sequence[str]
    ) -> list[str | None]:
        ...


def sanitize(
    raw: Sequence[str | None], labels: Sequence[str]
) -> list[str | None]:
    """Map each reply to an approved label or `None`. One entry in, one out.

    No substring or fuzzy matching: "Newsletters/CI" never silently becomes
    "Newsletters". Padding a short reply out to the batch length is the
    caller's job.
    """
    allowed = set(labels)
    result: list[str | None] = []
    for entry in raw:
        if isinstance(entry, str) and entry.strip() in allowed:
            result.append(entry.strip())
        else:
            result.append(None)
    return result


def estimate(
    messages: Sequence[MessageRow], labels: Sequence[str]
) -> tuple[int, int]:
    """`(message_count, rough_token_count)` for the messages that would be sent."""
    tokens = len(_render(labels, list(messages))) // CHARS_PER_TOKEN
    return len(messages), tokens


def make_classifier(
    cfg: Config, *, http: httpx.Client | None = None
) -> Classifier:
    client = http or httpx.Client(timeout=120.0)
    if cfg.llm_provider == "anthropic":
        return AnthropicClassifier(client, _api_key("ANTHROPIC_API_KEY"), cfg.llm_model)
    if cfg.llm_provider == "openai":
        return OpenAIClassifier(client, _api_key("OPENAI_API_KEY"), cfg.llm_model)
    raise ConfigError(f"unknown llm provider: {cfg.llm_provider!r}")


class AnthropicClassifier:
    def __init__(self, http: httpx.Client, api_key: str, model: str):
        self._http = http
        self._api_key = api_key
        self._model = model

    def classify(
        self, batch: Sequence[MessageRow], labels: Sequence[str]
    ) -> list[str | None]:
        return _classify_with(lambda chunk: self._call(chunk, labels), batch, labels)

    def _call(self, batch: Sequence[MessageRow], labels: Sequence[str]) -> str:
        response = self._http.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json={
                "model": self._model,
                "max_tokens": MAX_TOKENS,
                "temperature": 0,
                "messages": [
                    {"role": "user", "content": _render(labels, list(batch))}
                ],
            },
        )
        response.raise_for_status()
        for block in response.json().get("content") or ():
            if block.get("type") == "text":
                return block.get("text", "")
        return ""


class OpenAIClassifier:
    def __init__(self, http: httpx.Client, api_key: str, model: str):
        self._http = http
        self._api_key = api_key
        self._model = model

    def classify(
        self, batch: Sequence[MessageRow], labels: Sequence[str]
    ) -> list[str | None]:
        return _classify_with(lambda chunk: self._call(chunk, labels), batch, labels)

    def _call(self, batch: Sequence[MessageRow], labels: Sequence[str]) -> str:
        response = self._http.post(
            OPENAI_URL,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self._model,
                "temperature": 0,
                "max_tokens": MAX_TOKENS,
                "messages": [
                    {"role": "user", "content": _render(labels, list(batch))}
                ],
            },
        )
        response.raise_for_status()
        choices = response.json().get("choices") or ()
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content", "") or ""


def _classify_with(
    call: Callable[[Sequence[MessageRow]], str],
    batch: Sequence[MessageRow],
    labels: Sequence[str],
) -> list[str | None]:
    """One retry on unparseable output, then give up rather than guess."""
    if not batch:
        return []
    for _ in range(2):
        parsed = _parse_array(call(batch))
        if parsed is not None:
            mapped = sanitize(parsed, labels)
            if len(mapped) < len(batch):
                mapped += [None] * (len(batch) - len(mapped))
            return mapped[: len(batch)]
    return [None] * len(batch)


def _parse_array(text) -> list | None:
    if not isinstance(text, str):
        return None
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, list) else None


def _render(labels: Sequence[str], batch: Sequence[MessageRow]) -> str:
    messages = []
    for index, message in enumerate(batch, start=1):
        messages.append(f"{index}. from: {message.from_addr} | subject: {message.subject}")
        if message.excerpt:
            messages.append(f"   {message.excerpt}")
    return PROMPT_TEMPLATE.format(
        labels="\n".join(f"- {label}" for label in labels),
        messages="\n".join(messages),
    )


def _api_key(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(
            f"{name} is not set. Export it, or choose a different llm.provider."
        )
    return value
