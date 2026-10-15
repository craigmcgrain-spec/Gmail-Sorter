"""Load and validate config.toml, and derive the classifier version.

Nothing here repairs a bad config: `load_config` either returns a fully valid
`Config` or raises `ConfigError` naming the offending value.
"""

from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

DEFAULT_QUERY = "has:nouserlabels -is:starred -is:important -larger:5M"
DEFAULT_SIZE = "5M"
DEFAULT_EXCERPT_CHARS = 2000
PROVIDERS = ("anthropic", "openai")

# Gmail size suffixes. K/M/G are binary here: the guard query is authoritative
# and this local re-check is the belt-and-braces, so a threshold either side of
# 5 MiB is harmless.
_SIZE_SUFFIXES = {"K": 1024, "M": 1024**2, "G": 1024**3}

_MATCH_FIELDS = ("from_domain", "from_re", "subject_re", "list_id", "category")

# Gmail's own labels. Adding one of these to a message changes the message
# rather than filing it, so they are reserved and guarded against.
SYSTEM_LABEL_IDS = frozenset(
    {
        "INBOX",
        "UNREAD",
        "STARRED",
        "IMPORTANT",
        "SENT",
        "DRAFT",
        "SPAM",
        "TRASH",
        "CATEGORY_PERSONAL",
        "CATEGORY_SOCIAL",
        "CATEGORY_PROMOTIONS",
        "CATEGORY_UPDATES",
        "CATEGORY_FORUMS",
    }
)


class ConfigError(ValueError):
    """Raised for any invalid configuration value."""


@dataclass(frozen=True)
class Rule:
    name: str
    priority: int
    label: str
    from_domain: tuple[str, ...] = ()
    from_re: str | None = None
    subject_re: str | None = None
    list_id: str | None = None
    category: str | None = None


@dataclass(frozen=True)
class Config:
    query: str
    min_size_bytes: int
    size_suffix: str
    excerpt_chars: int
    store_bodies: bool
    llm_provider: str
    llm_model: str
    max_messages_per_run: int
    labels: tuple[str, ...]
    rules: tuple[Rule, ...]


def parse_size(text: str) -> int:
    """Parse a Gmail size such as "5M", "512K", "1G" or bare bytes."""
    raw = str(text).strip()
    if not raw:
        raise ConfigError(f"empty size: {text!r}")
    suffix = raw[-1].upper()
    if suffix in _SIZE_SUFFIXES:
        digits, multiplier = raw[:-1], _SIZE_SUFFIXES[suffix]
    else:
        digits, multiplier = raw, 1
    if not digits.isdigit():
        raise ConfigError(f"not a size: {text!r}")
    return int(digits) * multiplier


def _larger_token(query: str) -> str:
    """The size text of the query's last `-larger:` clause, else the default."""
    for token in reversed(query.split()):
        if token.startswith("-larger:"):
            return token[len("-larger:") :]
    return DEFAULT_SIZE


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    try:
        raw = tomllib.loads(config_path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config not found: {config_path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {config_path}: {exc}") from exc

    candidates = raw.get("candidates", {})
    fetch = raw.get("fetch", {})
    llm = raw.get("llm", {})
    taxonomy = raw.get("taxonomy", {})

    query = candidates.get("query", DEFAULT_QUERY)
    if not isinstance(query, str) or not query.strip():
        raise ConfigError(f"candidates.query must be a non-empty string, got {query!r}")
    size_suffix = _larger_token(query)
    min_size_bytes = parse_size(size_suffix)

    provider = llm.get("provider", PROVIDERS[0])
    if provider not in PROVIDERS:
        raise ConfigError(
            f"unknown llm provider: {provider!r} (expected one of {', '.join(PROVIDERS)})"
        )
    model = llm.get("model")
    if not model:
        raise ConfigError("llm.model is required")

    labels = tuple(taxonomy.get("labels", ()))
    if not labels:
        raise ConfigError(
            "taxonomy.labels is empty — no label can ever be applied. "
            "Uncomment the [taxonomy] block in your config."
        )
    _reject_reserved_labels(labels)

    rules = tuple(_load_rules(raw.get("rules", ()), labels))

    return Config(
        query=query,
        min_size_bytes=min_size_bytes,
        size_suffix=size_suffix,
        excerpt_chars=int(fetch.get("excerpt_chars", DEFAULT_EXCERPT_CHARS)),
        store_bodies=bool(fetch.get("store_bodies", True)),
        llm_provider=provider,
        llm_model=model,
        max_messages_per_run=int(llm.get("max_messages_per_run", 0)),
        labels=labels,
        rules=rules,
    )


def _reject_reserved_labels(labels: tuple[str, ...]) -> None:
    """Refuse a taxonomy label that is really one of Gmail's own labels.

    A label named `TRASH` resolves to Gmail's Trash label id, so applying it
    moves mail to Trash; `INBOX` and `UNREAD` change Inbox and read state; and
    `STARRED`/`IMPORTANT` would add the very states this tool refuses to touch.
    """
    for label in labels:
        stripped = label.strip()
        if stripped.upper() in SYSTEM_LABEL_IDS:
            raise ConfigError(
                f"taxonomy.labels contains the reserved Gmail system label "
                f"{stripped!r}. Adding it would change the message itself rather "
                f"than file it — rename it, for example {stripped.title() + ' Mail'!r}."
            )


def _load_rules(raw_rules: Iterable[dict], labels: tuple[str, ...]) -> list[Rule]:
    rules: list[Rule] = []
    seen: set[str] = set()
    for entry in raw_rules:
        name = entry.get("name")
        if not name:
            raise ConfigError(f"rule is missing a name: {entry!r}")
        if name in seen:
            raise ConfigError(f"duplicate rule name: {name!r}")
        seen.add(name)

        label = entry.get("label")
        if label not in labels:
            raise ConfigError(
                f"rule {name!r} names a label that is not in taxonomy.labels: {label!r}"
            )

        for field in ("from_re", "subject_re"):
            pattern = entry.get(field)
            if pattern is None:
                continue
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ConfigError(
                    f"rule {name!r} has an invalid {field}: {pattern!r}: {exc}"
                ) from exc

        values = {field: entry.get(field) for field in _MATCH_FIELDS}
        values["from_domain"] = tuple(entry.get("from_domain", ()))
        if not any(values[field] for field in _MATCH_FIELDS):
            raise ConfigError(
                f"rule {name!r} needs at least one match field "
                f"({', '.join(_MATCH_FIELDS)})"
            )

        rules.append(
            Rule(
                name=name,
                priority=int(entry.get("priority", 0)),
                label=label,
                from_domain=values["from_domain"],
                from_re=values["from_re"],
                subject_re=values["subject_re"],
                list_id=values["list_id"],
                category=values["category"],
            )
        )
    return rules


def classifier_version(
    labels: Iterable[str],
    rules: Iterable[Rule],
    prompt_template: str,
    model: str,
) -> str:
    """A hash of everything that can change a decision.

    Edit a rule, the taxonomy, the prompt or the model and every cached decision
    is invalidated; change nothing and cached decisions stand, so a re-run costs
    no API calls.
    """
    payload = {
        "labels": sorted(labels),
        "rules": sorted(
            (
                {
                    "name": rule.name,
                    "priority": rule.priority,
                    "label": rule.label,
                    "from_domain": list(rule.from_domain),
                    "from_re": rule.from_re,
                    "subject_re": rule.subject_re,
                    "list_id": rule.list_id,
                    "category": rule.category,
                }
                for rule in rules
            ),
            key=lambda item: item["name"],
        ),
        "prompt": prompt_template,
        "model": model,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
