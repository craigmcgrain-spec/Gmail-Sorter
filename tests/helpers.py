"""Factories shared by every later test module."""

from __future__ import annotations

from pathlib import Path

from gmail_sort.config import Config, Rule
from gmail_sort.store import Decision, MessageRow, Store

GUARD_QUERY = "has:nouserlabels -is:starred -is:important -larger:5M"
LABELS = ("Newsletters", "Receipts", "Notifications/CI")

CONFIG_TOML = """
[candidates]
query = "has:nouserlabels -is:starred -is:important -larger:5M"

[fetch]
excerpt_chars = 2000
store_bodies = true

[llm]
provider = "anthropic"
model = "claude-haiku-4-5"
max_messages_per_run = 0

[taxonomy]
labels = ["Newsletters", "Receipts", "Notifications/CI"]

[[rules]]
name = "ci"
priority = 10
label = "Notifications/CI"
from_domain = ["github.com"]
"""


def row(message_id: str = "m1", **overrides) -> MessageRow:
    base = dict(
        id=message_id,
        thread_id="t1",
        from_addr="a@b.com",
        from_domain="b.com",
        subject="Hi",
        list_id=None,
        category="CATEGORY_PERSONAL",
        size_estimate=1000,
        label_ids=("INBOX",),
        excerpt=None,
        fetched_at="2026-10-15T00:00:00Z",
    )
    base.update(overrides)
    return MessageRow(**base)


def ci_rule() -> Rule:
    return Rule(name="ci", priority=10, label="Notifications/CI",
                from_domain=("github.com",))


def cfg(**overrides) -> Config:
    base = dict(
        query=GUARD_QUERY,
        min_size_bytes=5 * 1024**2,
        size_suffix="5M",
        excerpt_chars=2000,
        store_bodies=True,
        llm_provider="anthropic",
        llm_model="claude-haiku-4-5",
        max_messages_per_run=0,
        labels=LABELS,
        rules=(ci_rule(),),
    )
    base.update(overrides)
    return Config(**base)


def write_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(CONFIG_TOML)
    return path


def seed_decision(store: Store, message_id: str, label: str | None,
                  version: str = "v") -> None:
    store.upsert_messages([row(message_id)])
    store.put_decision(
        Decision(
            message_id=message_id,
            label=label,
            source="rule",
            rule_name="r",
            classifier_version=version,
            decided_at="2026-10-15T00:00:00Z",
        )
    )
