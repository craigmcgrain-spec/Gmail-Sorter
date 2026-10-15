"""The safety boundary.

Gmail's search query narrows the candidate set, but search is thread-oriented
(`-is:starred` can surface a conversation containing one unstarred message) and
`larger:` is approximate. This module re-checks each message against the guards
immediately before anything is written.
"""

from __future__ import annotations

from typing import Iterable

from gmail_sort.config import Config

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


def guard_query(cfg: Config) -> str:
    """The verbatim candidate query; guarded mail is never even listed."""
    return cfg.query


def user_label_ids(labels: Iterable) -> frozenset[str]:
    """The ids of labels the user made, ignoring Gmail's own system labels.

    Accepts `gmail.gmail.Label` or a plain dict with `id`/`type`, because the
    test fake returns dicts.
    """
    ids = set()
    for label in labels:
        if isinstance(label, dict):
            kind, label_id = label.get("type"), label.get("id")
        else:
            kind, label_id = label.type, label.id
        if kind == "user" and label_id:
            ids.add(label_id)
    return frozenset(ids)


def violation(
    message,
    *,
    user_label_ids: frozenset[str],
    min_size_bytes: int,
) -> str | None:
    """Why this message must not be touched, or `None` if it is fair game.

    Checked in this order so the most consequential reason is the one reported.
    """
    ids = set(getattr(message, "label_ids", ()) or ())
    if "STARRED" in ids:
        return "starred"
    if "IMPORTANT" in ids:
        return "important"
    if ids & user_label_ids:
        return "labelled"
    # >= rather than >: the query's `larger:` is approximate, this is not.
    if int(getattr(message, "size_estimate", 0) or 0) >= min_size_bytes:
        return "oversize"
    return None
