"""Local SQLite cache: fetched messages, classifier decisions, run log.

This is what makes the tool cheap to iterate on. Bodies and decisions are
cached keyed by `classifier_version`, so editing a rule and re-running
`classify` re-examines only what actually changed, with no API calls at all.

`run_messages` doubles as the undo log.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Sequence

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
  id            TEXT PRIMARY KEY,
  thread_id     TEXT NOT NULL,
  from_addr     TEXT NOT NULL,
  from_domain   TEXT NOT NULL,
  subject       TEXT,
  list_id       TEXT,
  category      TEXT,
  size_estimate INTEGER NOT NULL,
  label_ids     TEXT NOT NULL,
  excerpt       TEXT,
  fetched_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
  message_id         TEXT PRIMARY KEY,
  label              TEXT,
  source             TEXT NOT NULL,
  rule_name          TEXT,
  classifier_version TEXT NOT NULL,
  decided_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
  id          INTEGER PRIMARY KEY,
  kind        TEXT NOT NULL,
  dry_run     INTEGER NOT NULL,
  started_at  TEXT NOT NULL,
  finished_at TEXT,
  config_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_messages (
  run_id     INTEGER NOT NULL REFERENCES runs(id),
  message_id TEXT NOT NULL,
  label      TEXT NOT NULL,
  action     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS run_messages_run ON run_messages(run_id);
CREATE INDEX IF NOT EXISTS decisions_version ON decisions(classifier_version);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class MessageRow:
    id: str
    thread_id: str
    from_addr: str
    from_domain: str
    subject: str
    list_id: str | None
    category: str | None
    size_estimate: int
    label_ids: tuple[str, ...]
    excerpt: str | None
    fetched_at: str


@dataclass(frozen=True)
class Decision:
    message_id: str
    label: str | None
    source: str
    rule_name: str | None
    classifier_version: str
    decided_at: str


@dataclass(frozen=True)
class RunRow:
    id: int
    kind: str
    dry_run: bool
    started_at: str
    finished_at: str | None
    config_hash: str


class Store:
    def __init__(self, path: str | Path):
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc_info) -> bool:
        self.close()
        return False

    def close(self) -> None:
        self._conn.close()

    # messages ---------------------------------------------------------------

    def upsert_messages(self, rows: Iterable[MessageRow]) -> None:
        payload = [
            (
                row.id,
                row.thread_id,
                row.from_addr,
                row.from_domain,
                row.subject,
                row.list_id,
                row.category,
                row.size_estimate,
                json.dumps(list(row.label_ids)),
                row.excerpt,
                row.fetched_at,
            )
            for row in rows
        ]
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO messages (id, thread_id, from_addr,"
                " from_domain, subject, list_id, category, size_estimate,"
                " label_ids, excerpt, fetched_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                payload,
            )

    def get_message(self, message_id: str) -> MessageRow | None:
        cur = self._conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,))
        found = cur.fetchone()
        return _message(found) if found else None

    def has_message(self, message_id: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM messages WHERE id = ?", (message_id,)
            ).fetchone()
            is not None
        )

    def iter_messages(self) -> Iterator[MessageRow]:
        for found in self._conn.execute("SELECT * FROM messages ORDER BY id"):
            yield _message(found)

    # decisions --------------------------------------------------------------

    def put_decision(self, decision: Decision) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO decisions (message_id, label, source,"
                " rule_name, classifier_version, decided_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    decision.message_id,
                    decision.label,
                    decision.source,
                    decision.rule_name,
                    decision.classifier_version,
                    decision.decided_at,
                ),
            )

    def get_decision(
        self, message_id: str, classifier_version: str
    ) -> Decision | None:
        cur = self._conn.execute(
            "SELECT * FROM decisions WHERE message_id = ? AND classifier_version = ?",
            (message_id, classifier_version),
        )
        found = cur.fetchone()
        return _decision(found) if found else None

    def iter_decisions(self, classifier_version: str) -> Iterator[Decision]:
        for found in self._conn.execute(
            "SELECT * FROM decisions WHERE classifier_version = ? ORDER BY message_id",
            (classifier_version,),
        ):
            yield _decision(found)

    # runs and the undo log --------------------------------------------------

    def start_run(self, kind: str, dry_run: bool, config_hash: str) -> int:
        with self._conn:
            cur = self._conn.execute(
                "INSERT INTO runs (kind, dry_run, started_at, config_hash)"
                " VALUES (?, ?, ?, ?)",
                (kind, int(dry_run), now(), config_hash),
            )
        return int(cur.lastrowid)

    def finish_run(self, run_id: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE runs SET finished_at = ? WHERE id = ?", (now(), run_id)
            )

    def get_run(self, run_id: int) -> RunRow | None:
        cur = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        found = cur.fetchone()
        if not found:
            return None
        return RunRow(
            id=int(found["id"]),
            kind=found["kind"],
            dry_run=bool(found["dry_run"]),
            started_at=found["started_at"],
            finished_at=found["finished_at"],
            config_hash=found["config_hash"],
        )

    def log_change(
        self, run_id: int, message_id: str, label: str, action: str
    ) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO run_messages (run_id, message_id, label, action)"
                " VALUES (?, ?, ?, ?)",
                (run_id, message_id, label, action),
            )

    def iter_changes(self, run_id: int) -> Iterator[tuple[str, str, str]]:
        for found in self._conn.execute(
            "SELECT message_id, label, action FROM run_messages"
            " WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        ):
            yield (found["message_id"], found["label"], found["action"])

    def applied_pairs(self) -> set[tuple[str, str]]:
        """(message_id, label) pairs applied and not since undone.

        Read in run_id then rowid order so the last action for a pair wins. This
        is what makes a second `apply` issue no batchModify calls, and what
        re-enables a pair after `undo`.
        """
        latest: dict[tuple[str, str], str] = {}
        for found in self._conn.execute(
            "SELECT message_id, label, action FROM run_messages"
            " ORDER BY run_id, rowid"
        ):
            latest[(found["message_id"], found["label"])] = found["action"]
        return {pair for pair, action in latest.items() if action == "add"}


def _message(found: sqlite3.Row) -> MessageRow:
    return MessageRow(
        id=found["id"],
        thread_id=found["thread_id"],
        from_addr=found["from_addr"],
        from_domain=found["from_domain"],
        subject=found["subject"] or "",
        list_id=found["list_id"],
        category=found["category"],
        size_estimate=int(found["size_estimate"]),
        label_ids=tuple(json.loads(found["label_ids"])),
        excerpt=found["excerpt"],
        fetched_at=found["fetched_at"],
    )


def _decision(found: sqlite3.Row) -> Decision:
    return Decision(
        message_id=found["message_id"],
        label=found["label"],
        source=found["source"],
        rule_name=found["rule_name"],
        classifier_version=found["classifier_version"],
        decided_at=found["decided_at"],
    )
