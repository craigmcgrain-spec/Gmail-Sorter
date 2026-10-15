"""The deterministic half of classification: pure, no I/O.

`classify` accepts anything carrying `from_domain`, `from_addr`, `subject`,
`list_id` and `category`, so it works with `store.MessageRow` and with plain
stand-ins in tests.
"""

from __future__ import annotations

import re
from typing import Sequence

from gmail_sort.config import Rule


def classify(message, rules: Sequence[Rule]) -> tuple[str, str] | None:
    """Return `(label, rule_name)` for the first matching rule, else `None`.

    First match wins, ordered by `(priority, name)`. Every match field present
    on a rule must match; a rule with no match fields cannot exist (config
    validation rejects it).
    """
    for rule in sorted(rules, key=lambda item: (item.priority, item.name)):
        if _matches(message, rule):
            return rule.label, rule.name
    return None


def _matches(message, rule: Rule) -> bool:
    domain = (getattr(message, "from_domain", "") or "").lower()
    if rule.from_domain and domain not in {
        candidate.lower() for candidate in rule.from_domain
    }:
        return False

    if rule.from_re is not None and not _search(
        rule.from_re, getattr(message, "from_addr", "")
    ):
        return False

    if rule.subject_re is not None and not _search(
        rule.subject_re, getattr(message, "subject", "")
    ):
        return False

    if rule.list_id is not None and rule.list_id != (getattr(message, "list_id", None) or ""):
        return False

    if rule.category is not None and rule.category != (getattr(message, "category", None) or ""):
        return False

    return True


def _search(pattern: str, value) -> bool:
    # re caches compiled patterns internally, so there is nothing to memoise.
    return re.search(pattern, value or "", re.IGNORECASE) is not None
