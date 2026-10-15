# Gmail Sort Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python CLI that labels a large Gmail backlog via deterministic rules with an LLM fallback, safely and re-runnably, so the user can review each label and delete what they don't want.

**Architecture:** Four commands (`survey`, `classify`, `apply`, `undo`) over a local SQLite cache. Guards collapse into one Gmail search query so guarded mail is never fetched; candidates are re-validated locally before any write. Rules run offline against cached metadata; only unmatched messages reach the LLM, and every decision is cached against a config hash so re-tuning costs no API calls.

**Tech Stack:** Python 3.11+, `httpx` for the five Gmail REST calls, `google-auth-oauthlib` for the OAuth installed-app flow, `sqlite3` from the stdlib, `pytest` for tests.

**Spec:** `docs/superpowers/specs/2026-10-15-gmail-sort-design.md`

## Global Constraints

- Package name `gmail_sort`, seven modules: `cli.py`, `config.py`, `gmail.py`, `store.py`, `rules.py`, `guards.py`, `llm.py`.
- OAuth scope is exactly `https://www.googleapis.com/auth/gmail.modify`.
- `apply` populates **only** `addLabelIds`. `removeLabelIds` is set by `undo` and by nothing else. Nothing in this program deletes a message.
- Guard query default, verbatim: `has:nouserlabels -is:starred -is:important -larger:5M`
- Large-attachment threshold default: `5M`. Size suffixes are `K` = 1024, `M` = 1024², `G` = 1024³ bytes.
- `batchModify` chunk size is exactly 1000 message IDs per request.
- Gmail quota costs used by the rate limiter: `messages.list` 5, `messages.get` 20, `messages.batchModify` 50, `labels.create` 5, `labels.list` 1. Token bucket rate: 100 units/second.
- `excerpt_chars` default 2000; `store_bodies` default true.
- `apply` refuses to write without `--yes`; without it, it prints the plan and exits 0.
- `survey` writes its proposal to `config.proposed.toml` with the `[taxonomy]` block commented out.
- Runtime dependencies are limited to `httpx` and `google-auth-oauthlib`. No Google discovery client.
- Every command's stages commit progress to SQLite as they go, so an interrupted run resumes.

## Review Focus

Inputs and conditions the spec implies but does not spell out, most likely to bite a real user first. Each has a test in the task that owns the code.

1. **A mailbox larger than one `messages.list` page.** `maxResults` caps at 500, so a 50,000-message backlog is ~100 pages. A `pageToken` loop that stops early silently labels only the first page and reports success.
2. **A message deleted between `classify` and `apply`.** `batchModify` rejects the whole chunk, so one stale ID must not abort a run over 50,000 messages.
3. **An LLM reply outside the taxonomy.** Free-text, a hallucinated label, or a near-miss must become `UNSORTED`. This program must never create or apply a label the user did not approve.
4. **A cached decision whose label left the taxonomy.** Editing the taxonomy invalidates via the config hash, but a decision cached before the edit and not yet re-classified must be dropped at apply time, not applied.
5. **Zero candidates.** A new or tiny mailbox, or one where everything is guarded, must exit 0 with zero counts rather than crashing on an empty sequence or a division by zero.

---

### Task 1: Project skeleton, config loading and validation

**Files:**
- Create: `pyproject.toml`, `gmail_sort/__init__.py`, `gmail_sort/config.py`, `config.example.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class Rule` — frozen dataclass: `name: str`, `priority: int`, `label: str`, `from_domain: tuple[str, ...]`, `from_re: str | None`, `subject_re: str | None`, `list_id: str | None`, `category: str | None`.
  - `class Config` — frozen dataclass: `query: str`, `min_size_bytes: int`, `size_suffix: str`, `excerpt_chars: int`, `store_bodies: bool`, `llm_provider: str`, `llm_model: str`, `max_messages_per_run: int`, `labels: tuple[str, ...]`, `rules: tuple[Rule, ...]`.
  - `class ConfigError(ValueError)`.
  - `load_config(path: str | Path) -> Config`
  - `parse_size(text: str) -> int`
  - `classifier_version(labels: Iterable[str], rules: Iterable[Rule], prompt_template: str, model: str) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
import pytest
from pathlib import Path
from gmail_sort.config import ConfigError, classifier_version, load_config, parse_size

MINIMAL = """
[candidates]
query = "has:nouserlabels -is:starred -is:important -larger:5M"

[llm]
provider = "anthropic"
model = "claude-haiku-4-5"

[taxonomy]
labels = ["Newsletters", "Receipts"]

[[rules]]
name = "newsletters"
priority = 10
label = "Newsletters"
list_id = "news.example.com"
"""

def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(body)
    return p

def test_load_minimal_config(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.min_size_bytes == 5 * 1024 * 1024
    assert cfg.excerpt_chars == 2000
    assert cfg.store_bodies is True
    assert cfg.labels == ("Newsletters", "Receipts")
    assert cfg.rules[0].name == "newsletters"

def test_rule_naming_unknown_label_is_rejected(tmp_path):
    bad = MINIMAL.replace('label = "Newsletters"', 'label = "Nope"')
    with pytest.raises(ConfigError, match="Nope"):
        load_config(write(tmp_path, bad))

def test_duplicate_rule_names_are_rejected(tmp_path):
    dup = MINIMAL + """
[[rules]]
name = "newsletters"
priority = 20
label = "Receipts"
list_id = "other.example.com"
"""
    with pytest.raises(ConfigError, match="duplicate rule name"):
        load_config(write(tmp_path, dup))

def test_invalid_regex_is_rejected(tmp_path):
    bad = MINIMAL.replace('list_id = "news.example.com"', 'subject_re = "([unclosed"')
    with pytest.raises(ConfigError, match="subject_re"):
        load_config(write(tmp_path, bad))

def test_rule_with_no_match_fields_is_rejected(tmp_path):
    bad = MINIMAL.replace('list_id = "news.example.com"', "")
    with pytest.raises(ConfigError, match="at least one match field"):
        load_config(write(tmp_path, bad))

def test_unknown_llm_provider_is_rejected(tmp_path):
    bad = MINIMAL.replace('provider = "anthropic"', 'provider = "gemini"')
    with pytest.raises(ConfigError, match="provider"):
        load_config(write(tmp_path, bad))

def test_parse_size_suffixes():
    assert parse_size("5M") == 5 * 1024**2
    assert parse_size("1000000") == 1000000
    assert parse_size("1G") == 1024**3
    assert parse_size("512K") == 512 * 1024

def test_classifier_version_is_stable_and_sensitive():
    args = (["A"], (), "prompt", "model")
    assert classifier_version(*args) == classifier_version(*args)
    for i, changed in enumerate([
        (["B"], (), "prompt", "model"),
        (["A"], (), "other prompt", "model"),
        (["A"], (), "prompt", "other-model"),
    ]):
        assert classifier_version(*changed) != classifier_version(*args), i
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gmail_sort.config'`

- [ ] **Step 3: Implement `gmail_sort/config.py`**

Python 3.11 `tomllib` for parsing. `load_config` validates and raises `ConfigError` with a message naming the offending value; it does not repair anything. `parse_size` accepts a trailing `K`/`M`/`G` (case-insensitive) or bare digits. `min_size_bytes` comes from the last whitespace-separated token of `query` if it matches `-larger:`, else defaults to `5M`. `size_suffix` keeps the original text so `survey` can print it. `classifier_version` is `sha256` over the JSON-serialised inputs with `sort_keys=True`, hex-truncated to 16 chars.

Also create `config.example.toml` with the `[candidates]`, `[fetch]`, `[llm]`, `[taxonomy]` and one commented `[[rules]]` example exactly as they appear in the spec's Configuration section, and `pyproject.toml` declaring the package, `requires-python = ">=3.11"`, dependencies `httpx` and `google-auth-oauthlib`, and a `[project.scripts]` entry `gmail-sort = "gmail_sort.cli:main"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml config.example.toml gmail_sort/ tests/test_config.py
git commit -m "feat: config loading and validation"
```

---

### Task 2: Rule engine

**Files:**
- Create: `gmail_sort/rules.py`
- Test: `tests/test_rules.py`

**Interfaces:**
- Consumes: `config.Rule`; `store.MessageRow` (Task 4 defines it — for this task use a local stand-in with the same field names, and switch the import in Task 5 when `store.py` lands).
- Produces: `classify(message: MessageRow, rules: Sequence[Rule]) -> tuple[str, str] | None` returning `(label, rule_name)`, or `None` when no rule matches.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rules.py
from types import SimpleNamespace
from gmail_sort.config import Rule
from gmail_sort.rules import classify

def msg(**kw):
    base = dict(id="m1", thread_id="t1", from_addr="a@b.com", from_domain="b.com",
                subject="Hello", list_id=None, category="CATEGORY_PERSONAL",
                size_estimate=1000, label_ids=("INBOX",), excerpt=None,
                fetched_at="2026-10-15T00:00:00Z")
    base.update(kw)
    return SimpleNamespace(**base)

def test_matches_from_domain():
    rule = Rule(name="r", priority=1, label="L", from_domain=("b.com",))
    assert classify(msg(), [rule]) == ("L", "r")

def test_does_not_match_other_domain():
    rule = Rule(name="r", priority=1, label="L", from_domain=("other.com",))
    assert classify(msg(), [rule]) is None

def test_lowest_priority_number_wins():
    rules = [Rule(name="late", priority=20, label="B", from_domain=("b.com",)),
             Rule(name="early", priority=10, label="A", from_domain=("b.com",))]
    assert classify(msg(), rules) == ("A", "early")

def test_equal_priority_is_resolved_by_rule_name():
    rules = [Rule(name="bbb", priority=10, label="B", from_domain=("b.com",)),
             Rule(name="aaa", priority=10, label="A", from_domain=("b.com",))]
    assert classify(msg(), rules) == ("A", "aaa")

def test_from_regex_and_subject_regex_are_case_insensitive():
    rule = Rule(name="r", priority=1, label="L", subject_re="^hello\\b")
    assert classify(msg(subject="HELLO world"), [rule]) == ("L", "r")

def test_list_id_and_category_match():
    rule = Rule(name="r", priority=1, label="L", list_id="news.example.com",
                category="CATEGORY_UPDATES")
    assert classify(msg(list_id="news.example.com",
                        category="CATEGORY_UPDATES"), [rule]) == ("L", "r")
    assert classify(msg(list_id="news.example.com"), [rule]) is None

def test_multiple_match_fields_all_must_match():
    rule = Rule(name="r", priority=1, label="L", from_domain=("b.com",),
                subject_re="^nope")
    assert classify(msg(), [rule]) is None

def test_no_rules_returns_none():
    assert classify(msg(), []) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rules.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gmail_sort.rules'`

- [ ] **Step 3: Implement `gmail_sort/rules.py`**

Pure function, no I/O. Sort rules once per call by `(priority, name)`. `from_domain` matches if `message.from_domain` is in the tuple (exact, lowercased). Regex fields use `re.search` with `re.IGNORECASE`, compiled at module level per call. Fields that are `None` on the rule are skipped; every field present on the rule must match.

`classify` accepts anything with those attribute names, so it works with both the dataclass and the test stand-in.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rules.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add gmail_sort/rules.py tests/test_rules.py
git commit -m "feat: pure rule engine with priority ordering"
```

---

### Task 3: Guard logic

**Files:**
- Create: `gmail_sort/guards.py`
- Test: `tests/test_guards.py`

**Interfaces:**
- Consumes: `config.Config`; a message object with `label_ids: tuple[str, ...]` and `size_estimate: int`.
- Produces:
  - `SYSTEM_LABEL_IDS: frozenset[str]` — `INBOX, UNREAD, STARRED, IMPORTANT, SENT, DRAFT, SPAM, TRASH` plus the nine `CATEGORY_*` ids.
  - `violation(message, *, user_label_ids: frozenset[str], min_size_bytes: int) -> str | None` — returns `"starred"`, `"important"`, `"labelled"`, `"oversize"`, or `None`. Checks in that order.
  - `user_label_ids(labels: Iterable[Label]) -> frozenset[str]`
  - `guard_query(cfg: Config) -> str`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_guards.py
from types import SimpleNamespace
from gmail_sort.config import Config
from gmail_sort.guards import SYSTEM_LABEL_IDS, guard_query, user_label_ids, violation

LIMIT = 5 * 1024 * 1024

def msg(**kw):
    base = dict(label_ids=("INBOX", "Label_1"), size_estimate=1000)
    base.update(kw)
    return SimpleNamespace(**base)

def test_clean_message_passes():
    assert violation(msg(label_ids=("INBOX",)), user_label_ids=frozenset({"Label_1"}),
                     min_size_bytes=LIMIT) is None

def test_starred_is_rejected():
    assert violation(msg(label_ids=("INBOX", "STARRED")), user_label_ids=frozenset(),
                     min_size_bytes=LIMIT) == "starred"

def test_important_is_rejected():
    assert violation(msg(label_ids=("INBOX", "IMPORTANT")), user_label_ids=frozenset(),
                     min_size_bytes=LIMIT) == "important"

def test_user_label_is_rejected():
    assert violation(msg(), user_label_ids=frozenset({"Label_1"}),
                     min_size_bytes=LIMIT) == "labelled"

def test_category_label_is_not_a_user_label():
    # CATEGORY_* ids are system labels; they must not read as "already labelled".
    assert violation(msg(label_ids=("INBOX", "CATEGORY_PROMOTIONS")),
                     user_label_ids=frozenset({"Label_1", "Label_2"}),
                     min_size_bytes=LIMIT) is None

def test_oversize_is_rejected_at_the_boundary():
    assert violation(msg(size_estimate=LIMIT), user_label_ids=frozenset(),
                     min_size_bytes=LIMIT) == "oversize"
    assert violation(msg(size_estimate=LIMIT - 1), user_label_ids=frozenset(),
                     min_size_bytes=LIMIT) is None

def test_user_label_ids_filters_by_type():
    labels = [{"id": "Label_1", "type": "user"}, {"id": "INBOX", "type": "system"}]
    assert user_label_ids(labels) == frozenset({"Label_1"})

def test_system_label_ids_covers_gmail_system_labels():
    for lid in ("INBOX", "UNREAD", "STARRED", "IMPORTANT", "SENT", "DRAFT",
                "SPAM", "TRASH", "CATEGORY_PERSONAL", "CATEGORY_SOCIAL",
                "CATEGORY_PROMOTIONS", "CATEGORY_UPDATES", "CATEGORY_FORUMS"):
        assert lid in SYSTEM_LABEL_IDS

def test_guard_query_is_verbatim_from_config():
    cfg = Config(query="has:nouserlabels -is:starred", min_size_bytes=LIMIT,
                 size_suffix="5M", excerpt_chars=2000, store_bodies=True,
                 llm_provider="anthropic", llm_model="m", max_messages_per_run=0,
                 labels=("L",), rules=())
    assert guard_query(cfg) == "has:nouserlabels -is:starred"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_guards.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gmail_sort.guards'`

- [ ] **Step 3: Implement `gmail_sort/guards.py`**

`user_label_ids` accepts either a `Label` dataclass or a plain dict, since the fake client in tests returns dicts. `violation` takes `message.label_ids` and tests `STARRED`, then `IMPORTANT`, then intersection with `user_label_ids`, then `size_estimate >= min_size_bytes`. The `>=` matters: `larger:` in Gmail is approximate but the local check is authoritative.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_guards.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add gmail_sort/guards.py tests/test_guards.py
git commit -m "feat: guard re-validation and guard query"
```

---

### Task 4: SQLite store

**Files:**
- Create: `gmail_sort/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class MessageRow` — frozen dataclass: `id`, `thread_id`, `from_addr`, `from_domain`, `subject`, `list_id`, `category`, `size_estimate`, `label_ids: tuple[str, ...]`, `excerpt: str | None`, `fetched_at`.
  - `class Decision` — frozen dataclass: `message_id`, `label: str | None`, `source: str`, `rule_name: str | None`, `classifier_version`, `decided_at`.
  - `class RunRow` — frozen dataclass: `id: int`, `kind: str`, `dry_run: bool`, `started_at: str`, `finished_at: str | None`, `config_hash: str`.
  - `class Store` with `__init__(path: str | Path)`, `close()`, `upsert_messages(rows)`, `get_message(message_id) -> MessageRow | None`, `has_message(message_id) -> bool`, `iter_messages()`, `put_decision(d)`, `get_decision(message_id, classifier_version) -> Decision | None`, `iter_decisions(classifier_version)`, `start_run(kind, dry_run, config_hash) -> int`, `finish_run(run_id)`, `log_change(run_id, message_id, label, action)`, `iter_changes(run_id)`, `get_run(run_id) -> RunRow | None`, `applied_pairs() -> set[tuple[str, str]]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_store.py
from gmail_sort.store import Decision, MessageRow, Store

def row(mid="m1", **kw):
    base = dict(id=mid, thread_id="t1", from_addr="a@b.com", from_domain="b.com",
                subject="Hi", list_id=None, category="CATEGORY_PERSONAL",
                size_estimate=1000, label_ids=("INBOX",), excerpt=None,
                fetched_at="2026-10-15T00:00:00Z")
    base.update(kw)
    return MessageRow(**base)

def test_schema_is_created_on_open(tmp_path):
    with Store(tmp_path / "db.sqlite") as s:
        assert s.get_message("missing") is None

def test_upsert_is_idempotent_and_updates(tmp_path):
    with Store(tmp_path / "db.sqlite") as s:
        s.upsert_messages([row()])
        s.upsert_messages([row(subject="Changed")])
        assert s.get_message("m1").subject == "Changed"
        assert len(list(s.iter_messages())) == 1

def test_label_ids_round_trip_as_a_tuple(tmp_path):
    with Store(tmp_path / "db.sqlite") as s:
        s.upsert_messages([row(label_ids=("INBOX", "Label_1"))])
        assert s.get_message("m1").label_ids == ("INBOX", "Label_1")

def test_decision_is_scoped_to_classifier_version(tmp_path):
    with Store(tmp_path / "db.sqlite") as s:
        s.upsert_messages([row()])
        d = Decision(message_id="m1", label="L", source="rule", rule_name="r",
                     classifier_version="v1", decided_at="2026-10-15T00:00:00Z")
        s.put_decision(d)
        assert s.get_decision("m1", "v1").label == "L"
        assert s.get_decision("m1", "v2") is None

def test_put_decision_replaces_the_same_version(tmp_path):
    with Store(tmp_path / "db.sqlite") as s:
        s.upsert_messages([row()])
        for lab in ("L", "M"):
            s.put_decision(Decision(message_id="m1", label=lab, source="rule",
                                    rule_name="r", classifier_version="v1",
                                    decided_at="2026-10-15T00:00:00Z"))
        assert s.get_decision("m1", "v1").label == "M"
        assert len(list(s.iter_decisions("v1"))) == 1

def test_unsorted_decision_round_trips_as_none_label(tmp_path):
    with Store(tmp_path / "db.sqlite") as s:
        s.upsert_messages([row()])
        s.put_decision(Decision(message_id="m1", label=None, source="llm",
                                rule_name=None, classifier_version="v1",
                                decided_at="2026-10-15T00:00:00Z"))
        assert s.get_decision("m1", "v1").label is None

def test_run_lifecycle_and_change_log(tmp_path):
    with Store(tmp_path / "db.sqlite") as s:
        rid = s.start_run("apply", dry_run=False, config_hash="h1")
        s.log_change(rid, "m1", "L", "add")
        s.finish_run(rid)
        assert list(s.iter_changes(rid)) == [("m1", "L", "add")]
        r = s.get_run(rid)
        assert r.kind == "apply" and r.dry_run is False and r.finished_at is not None

def test_applied_pairs_counts_only_adds_not_since_removed(tmp_path):
    with Store(tmp_path / "d.sqlite") as s:
        rid = s.start_run("apply", dry_run=False, config_hash="h")
        s.log_change(rid, "m1", "L", "add")
        s.log_change(rid, "m2", "L", "add")
        s.log_change(rid, "m2", "L", "remove")
        assert s.applied_pairs() == {("m1", "L")}

def test_iter_messages_streams_without_loading_everything(tmp_path):
    with Store(tmp_path / "db.sqlite") as s:
        s.upsert_messages([row(f"m{i}") for i in range(50)])
        assert sum(1 for _ in s.iter_messages()) == 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gmail_sort.store'`

- [ ] **Step 3: Implement `gmail_sort/store.py`**

`sqlite3` with `PRAGMA journal_mode=WAL` and `PRAGMA synchronous=NORMAL`. Four tables exactly as the spec's Storage section defines them, created with `CREATE TABLE IF NOT EXISTS`. `label_ids` is stored as a JSON array and decoded back into a tuple. `Store` is a context manager (`__enter__`/`__exit__` calling `close`). `iter_messages` and `iter_decisions` and `iter_changes` are generators over the cursor, not list-returning. Timestamps are `datetime.now(timezone.utc).isoformat()`. `applied_pairs` returns the `(message_id, label)` pairs whose most recent `run_messages` row is an `add`, read in `run_id` then rowid order so the last action per pair wins.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_store.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add gmail_sort/store.py tests/test_store.py
git commit -m "feat: sqlite store with decisions, runs and undo log"
```

---

### Task 5: GmailClient protocol and in-memory fake

**Files:**
- Create: `gmail_sort/gmail.py` (protocol and types only in this task), `tests/fake_gmail.py`, `tests/helpers.py`
- Test: `tests/test_fake_gmail.py`

**Interfaces:**
- Consumes: `store.MessageRow`.
- Produces:
  - `class Label` — frozen dataclass: `id: str`, `name: str`, `type: str`.
  - `class GmailClient(Protocol)` with:
    - `iter_message_ids(query: str) -> Iterator[str]`
    - `get_message(message_id: str, *, full: bool = False) -> MessageRow`
    - `list_labels() -> list[Label]`
    - `create_label(name: str) -> Label`
    - `batch_modify(message_ids: Sequence[str], *, add: Sequence[str] = (), remove: Sequence[str] = ()) -> None`
    - `page_size: int` (500 in the real client)
    - `calls: list[tuple]` — recorded `(method, args)` triples, for tests
  - `class FakeGmailClient` in `tests/fake_gmail.py`, constructed as `FakeGmailClient(messages: dict[str, MessageRow], *, page_size: int = 500, label_ids: list[Label] | None = None, fail_on: set[str] | None = None)`.

Both `batch_modify` implementations chunk at 1000 internally, so callers never think about it and the fake and the real client are indistinguishable to `apply`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_fake_gmail.py
import pytest
from tests.fake_gmail import FakeGmailClient
from tests.helpers import row

def test_iter_message_ids_pages_past_the_page_size():
    msgs = {f"m{i}": row(f"m{i}") for i in range(1200)}
    fake = FakeGmailClient(msgs, page_size=500)
    assert list(fake.iter_message_ids("in:anywhere")) == list(msgs)
    assert 2 < len([c for c in fake.calls if c[0] == "iter_message_ids"])

def test_batch_modify_chunks_at_1000():
    fake = FakeGmailClient({})
    ids = [f"m{i}" for i in range(2500)]
    fake.batch_modify(ids, add=["Label_1"])
    calls = [c for c in fake.calls if c[0] == "batch_modify"]
    assert [len(c[2]["ids"]) for c in calls] == [1000, 1000, 500]

def test_batch_modify_applies_add_and_remove_to_state():
    fake = FakeGmailClient({})
    fake.batch_modify(["m1"], add=["Label_1"])
    fake.batch_modify(["m1"], remove=["Label_1"])
    assert fake.modified == [("m1", ("Label_1",), ()), ("m1", (), ("Label_1",))]

def test_batch_modify_raises_when_configured_to_fail():
    fake = FakeGmailClient({}, fail_on={"m2"})
    with pytest.raises(Exception, match="m2"):
        fake.batch_modify(["m1", "m2"], add=["Label_1"])

def test_create_label_nests_and_is_idempotent():
    fake = FakeGmailClient({})
    a = fake.create_label("Notifications/Traffic")
    assert a.name == "Notifications/Traffic"
    assert fake.create_label("Notifications/Traffic") == a
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fake_gmail.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gmail_sort.gmail'`

- [ ] **Step 3: Implement the protocol and the fake**

`gmail.py` in this task holds only `Label`, `GmailClient` and a `chunked(seq, size)` helper used by both implementations. `FakeGmailClient` records every call in `self.calls` as `(method_name, args_tuple, kwargs_dict)` and maintains `self.modified` as the ordered list of applied `(message_id, add, remove)` triples. `fail_on` simulates a message that no longer exists by raising from `batch_modify`. When a label name is not in `label_ids`, `create_label` assigns the next free id in creation order starting at `Label_1`; the apply and undo tests depend on `Receipts` being the first label created in a fresh fake. `label_ids` accepts `Label` dataclasses or plain dicts with `id`/`name`/`type`.

Create `tests/helpers.py` with the four factories every later task shares:

- `row(message_id: str, **overrides) -> MessageRow` — a complete, valid `MessageRow` with `from_domain="b.com"`, `label_ids=("INBOX",)` and `fetched_at="2026-10-15T00:00:00Z"`.
- `cfg(**overrides) -> Config` — a `Config` whose `query` is the verbatim guard query from Global Constraints (so `min_size_bytes` is 5 MiB), `excerpt_chars=2000`, `store_bodies=True`, `labels=("Newsletters", "Receipts", "Notifications/CI")`, and one rule matching `from_domain=("github.com",)` to `Notifications/CI` at priority 10.
- `write_cfg(tmp_path: Path) -> Path` — writes `cfg()` to `tmp_path/"config.toml"` as TOML and returns the path.
- `seed_decision(store: Store, message_id: str, label: str | None, version: str = "v") -> None` — upserts `row(message_id)` and inserts a `Decision(source="rule")` for that version.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fake_gmail.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add gmail_sort/gmail.py tests/fake_gmail.py tests/helpers.py tests/test_fake_gmail.py
git commit -m "feat: GmailClient protocol and in-memory fake"
```

---

### Task 6: Rate limiter and REST client

**Files:**
- Modify: `gmail_sort/gmail.py` (append `TokenBucket` and `RestGmailClient`)
- Test: `tests/test_ratelimit.py`, `tests/test_rest_client.py`

**Interfaces:**
- Consumes: `Label`, `GmailClient`, `chunked` from Task 5; `store.MessageRow`.
- Produces:
  - `QUOTA = {"list": 5, "get": 20, "batch_modify": 50, "labels_create": 5, "labels_list": 1}`
  - `class TokenBucket` — `__init__(self, rate_per_sec: float = 100.0, clock=time.monotonic, sleep=time.sleep)`; `take(self, units: int) -> None`.
  - `class RestGmailClient` — `__init__(self, http: httpx.Client, access_token: str, *, token_refresher: Callable[[], str] | None = None, bucket: TokenBucket | None = None)`; implements `GmailClient`.
  - `def parse_message(payload: dict, *, full: bool, excerpt_chars: int) -> MessageRow`

- [ ] **Step 1: Write the failing rate-limiter tests**

```python
# tests/test_ratelimit.py
from gmail_sort.gmail import TokenBucket

class FakeClock:
    def __init__(self): self.t = 0.0; self.slept = []
    def now(self): return self.t
    def sleep(self, s): self.slept.append(s); self.t += s

def test_take_within_budget_does_not_sleep():
    c = FakeClock(); b = TokenBucket(100.0, clock=c.now, sleep=c.sleep)
    b.take(50); b.take(50)
    assert c.slept == []

def test_take_over_budget_sleeps_the_deficit():
    c = FakeClock(); b = TokenBucket(100.0, clock=c.now, sleep=c.sleep)
    b.take(100)
    b.take(50)
    assert sum(c.slept) == pytest.approx(0.5)

def test_bucket_refills_over_time():
    c = FakeClock(); b = TokenBucket(100.0, clock=c.now, sleep=c.sleep)
    b.take(6000)
    c.t += 60
    b.take(6000)
    assert c.slept == [pytest.approx(0.0)] or sum(c.slept) < 1.0

def test_no_test_ever_sleeps_for_real():
    import time
    start = time.monotonic()
    FakeClock_bucket = TokenBucket(100.0, clock=FakeClock().now, sleep=FakeClock().sleep)
    FakeClock_bucket.take(10_000)
    assert time.monotonic() - start < 0.1
```

- [ ] **Step 2: Write the failing REST client tests**

```python
# tests/test_rest_client.py
import httpx, pytest
from gmail_sort.gmail import RestGmailClient, TokenBucket, parse_message
from tests.helpers import row

def client(handler, **kw):
    return RestGmailClient(httpx.Client(transport=httpx.MockTransport(handler)),
                           "tok", bucket=TokenBucket(1e9), **kw)

def test_list_walks_every_page_token():
    def handler(request):
        q = dict(httpx.URL(str(request.url)).params)
        pages = {"": {"messages": [{"id": "m1"}], "nextPageToken": "p2"},
                 "p2": {"messages": [{"id": "m2"}], "nextPageToken": "p3"},
                 "p3": {"messages": [{"id": "m3"}]}}
        body = pages[q.get("pageToken", "")]
        return httpx.Response(200, json=body)
    assert list(client(handler).iter_message_ids("q")) == ["m1", "m2", "m3"]

def test_list_requests_500_per_page_with_the_query():
    seen = {}
    def handler(request):
        seen.update(dict(httpx.URL(str(request.url)).params))
        return httpx.Response(200, json={"messages": []})
    list(client(handler).iter_message_ids("has:nouserlabels"))
    assert seen["maxResults"] == "500"
    assert seen["q"] == "has:nouserlabels"

def test_list_tolerates_an_empty_result():
    assert list(client(lambda r: httpx.Response(200, json={})).iter_message_ids("q")) == []

def test_retries_on_429_with_backoff_then_succeeds():
    calls = {"n": 0}
    slept = []
    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": {"code": 429}})
        return httpx.Response(200, json={"labels": []})
    c = client(handler)
    c._sleep = slept.append
    assert c.list_labels() == []
    assert calls["n"] == 3 and slept == [1.0, 2.0]

def test_gives_up_after_five_attempts():
    c = client(lambda r: httpx.Response(500))
    c._sleep = lambda s: None
    with pytest.raises(Exception):
        c.list_labels()

def test_401_refreshes_the_token_once_and_retries():
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(401, json={"error": {"code": 401}})
        assert request.headers["authorization"] == "Bearer fresh"
        return httpx.Response(200, json={"labels": []})
    c = RestGmailClient(httpx.Client(transport=httpx.MockTransport(handler)), "stale",
                        token_refresher=lambda: "fresh", bucket=TokenBucket(1e9))
    assert c.list_labels() == []
    assert calls["n"] == 2

def test_parse_message_derives_domain_and_headers():
    payload = {"id": "m1", "threadId": "t1", "labelIds": ["INBOX", "Label_1"],
               "sizeEstimate": 1234,
               "payload": {"headers": [
                   {"name": "From", "value": "Alice <Alice@Example.COM>"},
                   {"name": "Subject", "value": "Hi there"},
                   {"name": "List-Id", "value": "news.example.com"}]}}
    m = parse_message(payload, full=False, excerpt_chars=2000)
    assert m.from_addr == "Alice@Example.COM"
    assert m.from_domain == "example.com"
    assert m.subject == "Hi there"
    assert m.list_id == "news.example.com"
    assert m.size_estimate == 1234
    assert m.label_ids == ("INBOX", "Label_1")
    assert m.excerpt is None

def test_parse_message_extracts_and_truncates_a_text_body():
    import base64
    body = base64.urlsafe_b64encode(b"x" * 5000).decode()
    payload = {"id": "m1", "threadId": "t1", "labelIds": [], "sizeEstimate": 1,
               "payload": {"headers": [], "mimeType": "text/plain",
                           "body": {"data": body}}}
    m = parse_message(payload, full=True, excerpt_chars=100)
    assert len(m.excerpt) == 100

def test_parse_message_handles_a_missing_from_header():
    m = parse_message({"id": "m1", "threadId": "t1", "labelIds": [],
                       "sizeEstimate": 1, "payload": {"headers": []}},
                      full=False, excerpt_chars=2000)
    assert m.from_addr == "" and m.from_domain == ""

def test_parse_message_prefers_text_plain_in_multipart():
    import base64
    enc = lambda s: base64.urlsafe_b64encode(s).decode()
    payload = {"id": "m1", "threadId": "t1", "labelIds": [], "sizeEstimate": 1,
               "payload": {"headers": [], "mimeType": "multipart/alternative", "parts": [
                   {"mimeType": "text/html", "body": {"data": enc(b"<p>html</p>")}},
                   {"mimeType": "text/plain", "body": {"data": enc(b"plain text")}}]}}
    assert parse_message(payload, full=True, excerpt_chars=2000).excerpt == "plain text"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_ratelimit.py tests/test_rest_client.py -v`
Expected: FAIL — `ImportError: cannot import name 'TokenBucket'`

- [ ] **Step 4: Implement `TokenBucket` and `RestGmailClient`**

`TokenBucket` keeps `tokens` and `last` timestamp, refills from the injected clock, and sleeps only the deficit. It never calls `time.sleep` directly, so tests use a fake clock and nothing sleeps.

`RestGmailClient`:
- Base URL `https://gmail.googleapis.com/gmail/v1/users/me`, bearer auth from `access_token` on every request.
- `iter_message_ids` loops `pageToken` until absent, `maxResults=500`, `q=query`. Yields IDs lazily. An absent `messages` key yields nothing rather than raising (Review Focus #1 and #5).
- `get_message` sends `format=metadata` with `metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=List-Id` when `full=False`, and `format=full` when `full=True`.
- `list_labels` GETs `/labels`, maps to `Label`, and follows `nextPageToken`.
- `create_label` POSTs `/labels` with `{"name": name, "labelListVisibility": "labelShow", "messageListVisibility": "show"}`.
- `batch_modify` POSTs `/messages/batchModify` with `{"ids": chunk, "addLabelIds": [...], "removeLabelIds": [...]}`, chunked via `chunked(ids, 1000)`. Omits empty `addLabelIds`/`removeLabelIds` keys.
- Every call does `self._bucket.take(QUOTA[...])` first, and retries 429/5xx with truncated exponential backoff (1, 2, 4, 8, 16 seconds, capped at 64), maximum 5 attempts. Backoff sleeps via `self._sleep` so tests inject a recorder. A 401 triggers one `token_refresher()` call and one retry.
- `parse_message` is a module-level pure function: derive `from_domain` from the `From` header by lowercasing everything after the last `@` up to `>`; missing headers become `""`; `list_id` is `None` when absent; category is the first `CATEGORY_*` id in `labelIds`. Body extraction uses base64url decoding, prefers `text/plain`, and truncates to `excerpt_chars`. Any base64 or decode failure yields `excerpt=None` rather than raising.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_ratelimit.py tests/test_rest_client.py -v`
Expected: PASS, 24 tests.

- [ ] **Step 6: Commit**

```bash
git add gmail_sort/gmail.py tests/test_ratelimit.py tests/test_rest_client.py
git commit -m "feat: rate-limited Gmail REST client with pagination and backoff"
```

---

### Task 7: `survey` command

**Files:**
- Create: `gmail_sort/cli.py`
- Test: `tests/test_survey.py`

**Interfaces:**
- Consumes: `Config`, `load_config`; `GmailClient`; `MessageRow`; `guards.user_label_ids`, `guards.guard_query`.
- Produces:
  - `class Survey` — frozen dataclass: `total: int`, `domains: list[tuple[str, int]]`, `labels: list[tuple[str, int]]`, `categories: list[tuple[str, int]]`, `existing_labels: list[str]`.
  - `def survey(client: GmailClient, cfg: Config, *, top: int = 50) -> Survey`
  - `def render_proposal(s: Survey, cfg: Config) -> str` — returns the `config.proposed.toml` text with `[taxonomy]` commented out.
  - `def main(argv: Sequence[str] | None = None) -> int`

`survey` CLI: `gmail-sort survey [--config PATH] [--db PATH] [--out PATH] [--top N] [--sample N]`.

Senders, categories and label counts come from the first `--sample` candidates
(default 1000), because fetching a message to read its sender costs 20 quota
units where listing it costs a fraction of one. The candidate total is exact.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_survey.py
from gmail_sort.cli import render_proposal, survey
from tests.fake_gmail import FakeGmailClient
from tests.helpers import cfg, row, write_cfg

def test_counts_domains_and_categories():
    msgs = {f"m{i}": row(f"m{i}", from_domain=d, category=c)
            for i, (d, c) in enumerate([("a.com", "CATEGORY_PROMOTIONS"),
                                        ("a.com", "CATEGORY_PROMOTIONS"),
                                        ("b.com", "CATEGORY_UPDATES")])}
    s = survey(FakeGmailClient(msgs), cfg())
    assert s.total == 3
    assert s.domains[:2] == [("a.com", 2), ("b.com", 1)]
    assert s.categories[0] == ("CATEGORY_PROMOTIONS", 2)

def test_survey_makes_no_writes():
    fake = FakeGmailClient({}, label_ids=[{"id": "Label_1", "name": "Old", "type": "user"}])
    survey(fake, cfg())
    assert not any(c[0] in ("batch_modify", "create_label") for c in fake.calls)
    assert fake.modified == []

def test_survey_uses_the_guard_query():
    fake = FakeGmailClient({})
    survey(fake, cfg())
    assert fake.calls[0][1][0].startswith("has:nouserlabels")

def test_zero_candidates_is_not_an_error():
    s = survey(FakeGmailClient({}), cfg())
    assert s.total == 0 and s.domains == [] and s.categories == []

def test_proposal_comments_out_the_taxonomy():
    text = render_proposal(survey(FakeGmailClient({}), cfg()), cfg())
    assert "# [taxonomy]" in text
    assert "\n[taxonomy]" not in text

def test_proposal_suggests_labels_from_top_domains():
    msgs = {f"m{i}": row(f"m{i}", from_domain="github.com") for i in range(5)}
    text = render_proposal(survey(FakeGmailClient(msgs), cfg()), cfg())
    assert "github.com" in text

def test_proposal_always_parses_as_toml_when_uncommented():
    import tomllib
    msgs = {f"m{i}": row(f"m{i}", from_domain="github.com") for i in range(5)}
    text = render_proposal(survey(FakeGmailClient(msgs), cfg()), cfg())
    uncommented = "\n".join(l[2:] if l.startswith("# ") else l for l in text.splitlines())
    tomllib.loads(uncommented)

def test_main_survey_writes_the_proposal_file(tmp_path, monkeypatch):
    from gmail_sort import cli
    msgs = {f"m{i}": row(f"m{i}") for i in range(3)}
    monkeypatch.setattr(cli, "_open_client", lambda cfg, db: FakeGmailClient(msgs))
    out = tmp_path / "config.proposed.toml"
    assert cli.main(["survey", "--config", str(write_cfg(tmp_path)),
                     "--db", str(tmp_path / "d.sqlite"), "--out", str(out)]) == 0
    assert out.exists() and "# [taxonomy]" in out.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_survey.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gmail_sort.cli'`

- [ ] **Step 3: Implement `survey` and the CLI shell**

`cli.py` uses `argparse` with a subparser per command. `main` returns an `int` exit code and catches `ConfigError` to print a message and return 2. `_open_client(cfg, db) -> GmailClient` is the seam the tests monkeypatch; its real implementation does the OAuth flow (`google-auth-oauthlib` `InstalledAppFlow.from_client_secrets_file`, scope from Global Constraints, token cached in `<db_dir>/token.json`) and constructs `RestGmailClient`.

`survey` walks `iter_message_ids(guard_query(cfg))`, fetches each with `full=False` to get `from_domain` and category, tallies with `collections.Counter`, applies the `top` limit to `domains`, and lists existing labels via `list_labels()`. It writes nothing to Gmail and nothing to SQLite.

`render_proposal` emits a valid TOML document whose `[candidates]`, `[fetch]` and `[llm]` blocks carry the current values and whose `[taxonomy]` block is entirely `# `-prefixed, with a suggested label per top domain as a commented list. Because commented lines are stripped of `# ` by the parser in the test, the file must remain valid TOML when uncommented — so emit whole-line comments only.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_survey.py -v`
Expected: PASS, 10 tests.

- [ ] **Step 5: Commit**

```bash
git add gmail_sort/cli.py tests/test_survey.py
git commit -m "feat: survey command proposing a taxonomy"
```

---

### Task 8: LLM classifier

**Files:**
- Create: `gmail_sort/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `config.Config`, `store.MessageRow`.
- Produces:
  - `PROMPT_TEMPLATE: str`
  - `class Classifier(Protocol)`: `classify(self, batch: Sequence[MessageRow], labels: Sequence[str]) -> list[str | None]`
  - `class AnthropicClassifier` and `class OpenAIClassifier` — `__init__(self, http: httpx.Client, api_key: str, model: str)`
  - `def make_classifier(cfg: Config, *, http: httpx.Client | None = None) -> Classifier`
  - `def sanitize(raw: Sequence[str | None], labels: Sequence[str]) -> list[str | None]`
  - `def estimate(messages: Sequence[MessageRow], labels: Sequence[str]) -> tuple[int, int]` — `(message_count, estimated_tokens)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm.py
import httpx, pytest
from gmail_sort.llm import AnthropicClassifier, sanitize, estimate
from tests.helpers import row

LABELS = ["Newsletters", "Receipts"]

def test_sanitize_keeps_exact_matches():
    assert sanitize(["Newsletters", "Receipts"], LABELS) == ["Newsletters", "Receipts"]

def test_sanitize_rejects_an_invented_label():
    assert sanitize(["Totally New Label"], LABELS) == [None]

def test_sanitize_trims_whitespace_and_ignores_case_differences_in_case_only():
    assert sanitize(["  Receipts  "], LABELS) == ["Receipts"]

def test_sanitize_maps_unsorted_and_empty_to_none():
    assert sanitize(["UNSORTED", "", None], LABELS) == [None, None, None]

def test_sanitize_never_matches_a_substring():
    assert sanitize(["Newsletters/CI"], ["Newsletters", "Newsletters/CI"]) == ["Newsletters/CI"]
    assert sanitize(["Newsletters/CI"], ["Newsletters"]) == [None]

def test_sanitize_handles_a_short_response():
    assert sanitize(["Newsletters"], LABELS) == ["Newsletters", None]

def test_anthropic_posts_the_batch_and_parses_labels():
    def handler(request):
        body = __import__("json").loads(request.content)
        assert body["temperature"] == 0
        assert "Newsletters" in body["messages"][0]["content"]
        return httpx.Response(200, json={"content": [{"type": "text",
                                "text": '["Newsletters", "Receipts"]'}]})
    c = AnthropicClassifier(httpx.Client(transport=httpx.MockTransport(handler)),
                            "key", "claude-haiku-4-5")
    assert c.classify([row("m1"), row("m2")], LABELS) == ["Newsletters", "Receipts"]

def test_anthropic_retries_once_on_unparseable_output_then_gives_up():
    calls = {"n": 0}
    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"content": [{"type": "text", "text": "I cannot"}]})
    c = AnthropicClassifier(httpx.Client(transport=httpx.MockTransport(handler)),
                            "key", "m")
    assert c.classify([row("m1")], LABELS) == [None]
    assert calls["n"] == 2

def test_estimate_reports_count_and_tokens():
    n, tokens = estimate([row(f"m{i}") for i in range(10)], LABELS)
    assert n == 10 and tokens > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gmail_sort.llm'`

- [ ] **Step 3: Implement `gmail_sort/llm.py`**

`PROMPT_TEMPLATE` is a module constant that names the taxonomy, states that the only acceptable answers are one of the listed labels or `UNSORTED`, and asks for a JSON array of strings with one entry per message in order. It is hashed into `classifier_version`, so changing it invalidates cached decisions.

`sanitize` is the safety boundary and is deliberately strict: exact match against `labels` after `strip()`; `"UNSORTED"` (any case) and empty become `None`; no substring or fuzzy matching. It maps one entry in to one out; padding a short reply up to the batch length is `_classify_with`'s job, so a one-entry reply can never shift onto the second message. This is what enforces Review Focus #3.

`sanitize` is applied to **every** provider's output, and the providers themselves do no interpretation. Each provider posts one request per batch, parses a JSON array out of the reply, and on unparseable output retries exactly once before returning all `None`. `temperature` is 0. `make_classifier` reads the API key from `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` and raises `ConfigError` naming the missing variable.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_llm.py -v`
Expected: PASS, 14 tests.

- [ ] **Step 5: Commit**

```bash
git add gmail_sort/llm.py tests/test_llm.py
git commit -m "feat: provider-swappable LLM classifier with strict sanitize"
```

---

### Task 9: `classify` command

**Files:**
- Modify: `gmail_sort/cli.py`
- Test: `tests/test_classify.py`

**Interfaces:**
- Consumes: everything from Tasks 1–8.
- Produces:
  - `def classifier_version_for(cfg: Config) -> str` — `config.classifier_version(cfg.labels, cfg.rules, llm.PROMPT_TEMPLATE, cfg.llm_model)`; the single place the version is computed, so tests and `classify` agree on it.
  - `def classify(client: GmailClient, cfg: Config, store: Store, *, llm: Classifier | None = None, limit: int | None = None) -> ClassifyStats`
  - `class ClassifyStats` — frozen dataclass: `candidates: int`, `cached: int`, `rule_matched: int`, `llm_matched: int`, `unsorted: int`, `llm_calls: int`, `gets: int` (format=full fetches), `skipped_guard: int`, `estimated_tokens: int`.

`classify` CLI: `gmail-sort classify [--config PATH] [--db PATH] [--estimate] [--limit N] [--no-llm]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_classify.py
from gmail_sort.cli import classify, classifier_version_for
from gmail_sort.store import Store
from tests.fake_gmail import FakeGmailClient
from tests.helpers import cfg, row

def version():
    return classifier_version_for(cfg())

def run(tmp_path, msgs, **kw):
    # Not a `with` block: the caller asserts on decisions afterwards.
    store = Store(tmp_path / "d.sqlite")
    return classify(FakeGmailClient(msgs), cfg(), store, **kw), store

def test_rule_match_avoids_the_llm_entirely(tmp_path, rule_llm):
    msgs = {"m1": row("m1", from_domain="github.com")}
    stats, s = run(tmp_path, msgs, llm=rule_llm)
    assert stats.rule_matched == 1
    assert rule_llm.calls == 0
    assert s.get_decision("m1", version()).label == "Notifications/CI"

def test_unmatched_message_reaches_the_llm_and_is_cached(tmp_path, llm_returning):
    msgs = {"m1": row("m1", from_domain="unknown.com")}
    llm = llm_returning(["Receipts"])
    stats, s = run(tmp_path, msgs, llm=llm)
    assert stats.llm_matched == 1 and stats.llm_calls == 1

def test_second_run_makes_no_llm_calls_and_no_gets(tmp_path, llm_returning):
    msgs = {"m1": row("m1", from_domain="unknown.com")}
    llm = llm_returning(["Receipts"])
    with Store(tmp_path / "d.sqlite") as s:
        classify(FakeGmailClient(msgs), cfg(), s, llm=llm)
        fake2 = FakeGmailClient(msgs)
        stats = classify(fake2, cfg(), s, llm=llm)
    assert stats.cached == 1
    assert llm.calls == 1
    assert [c for c in fake2.calls if c[0] == "get_message"] == []

def test_changing_a_rule_invalidates_the_cache(tmp_path, llm_returning):
    msgs = {"m1": row("m1", from_domain="unknown.com")}
    with Store(tmp_path / "d.sqlite") as s:
        classify(FakeGmailClient(msgs), cfg(), s, llm=llm_returning(["Receipts"]))
        stats = classify(FakeGmailClient(msgs), cfg_with_extra_rule(), s,
                         llm=llm_returning(["Receipts"]))
    assert stats.llm_calls == 1

def test_unsorted_decisions_are_recorded_as_none(tmp_path, llm_returning):
    msgs = {"m1": row("m1", from_domain="unknown.com")}
    stats, s = run(tmp_path, msgs, llm=llm_returning([None]))
    assert stats.unsorted == 1
    assert s.get_decision("m1", version()).label is None

def test_full_fetch_only_for_messages_that_reach_the_llm(tmp_path, llm_returning):
    msgs = {"m1": row("m1", from_domain="github.com"),
            "m2": row("m2", from_domain="unknown.com")}
    stats, _ = run(tmp_path, msgs, llm=llm_returning(["Receipts"]))
    assert stats.gets == 1

def test_guard_violations_are_dropped_before_classifying(tmp_path, rule_llm):
    msgs = {"m1": row("m1", from_domain="github.com", label_ids=("INBOX", "STARRED"))}
    stats, s = run(tmp_path, msgs, llm=rule_llm)
    assert stats.candidates == 0
    assert s.get_decision("m1", version()) is None

def test_limit_caps_llm_bound_messages(tmp_path, llm_returning):
    msgs = {f"m{i}": row(f"m{i}", from_domain="unknown.com") for i in range(10)}
    llm = llm_returning(["Receipts"] * 10)
    stats, _ = run(tmp_path, msgs, llm=llm, limit=4)
    assert stats.llm_calls <= 4

def test_zero_candidates_exits_cleanly(tmp_path, rule_llm):
    stats, _ = run(tmp_path, {}, llm=rule_llm)
    assert stats.candidates == 0 and stats.llm_calls == 0

def test_no_llm_flag_leaves_unmatched_unsorted(tmp_path, rule_llm):
    msgs = {"m1": row("m1", from_domain="unknown.com")}
    stats, s = run(tmp_path, msgs, llm=None)
    assert stats.llm_calls == 0
    assert s.get_decision("m1", version()).source == "rule"
```

Add `tests/conftest.py` with the `rule_llm`, `llm_returning` and `cfg_with_extra_rule` helpers only — `cfg()` and `write_cfg()` already live in `tests/helpers.py` from Task 5. `RuleLLM` counts calls and returns `None` for everything; `llm_returning(labels)` pops from a queue and counts calls. `cfg_with_extra_rule()` is `cfg()` with one additional rule appended, so `classifier_version` changes and the cache is invalidated.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_classify.py -v`
Expected: FAIL — `ImportError: cannot import name 'classify'`

- [ ] **Step 3: Implement `classify`**

Walk `client.iter_message_ids(guard_query(cfg))`:

1. For each ID, skip if `store.get_decision(id, version)` exists — increment `cached`. **No fetch happens for cached messages** (the message row is already in `store.messages`).
2. Otherwise fetch with `format=metadata`. Re-validate guards with `violation(...)` against the label set from `list_labels()`; a violation is counted and skipped without being stored as a candidate.
3. Upsert the `MessageRow`, run `rules.classify`. A match is stored as a `Decision(source="rule")`.
4. Unmatched messages collect into an LLM batch. If `store_bodies` and the excerpt is not already cached, re-fetch that message with `format=full` — this is the only `format=full` call, and it is counted in `stats.gets`.
5. Send batches to the classifier, `sanitize` the labels, and store `Decision(source="llm")`, using `None` for `UNSORTED`.

`--limit N` caps how many messages reach the LLM. `--no-llm` skips step 4 and 5 entirely, recording unmatched messages as `source="rule"` with label `None`. `--estimate` runs steps 1–3 with no fetches for bodies, prints `estimate(...)` output, and exits without calling the LLM.

`classify` never writes to Gmail.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_classify.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
git add gmail_sort/cli.py tests/test_classify.py tests/conftest.py
git commit -m "feat: classify command with rule fast-path and cached LLM fallback"
```

---

### Task 10: `apply` command

**Files:**
- Modify: `gmail_sort/cli.py`
- Test: `tests/test_apply.py`

**Interfaces:**
- Consumes: everything from Tasks 1–9.
- Produces:
  - `def plan_apply(store: Store, cfg: Config, *, version: str) -> dict[str, list[str]]` — label name to message IDs, empty when nothing to do. Excludes pairs already applied and not since undone.
  - `def apply_plan(client: GmailClient, store: Store, cfg: Config, plan: dict[str, list[str]], *, dry_run: bool, run_id: int) -> ApplyStats`
  - `class ApplyStats` — frozen dataclass: `labels: int`, `messages: int`, `applied: int`, `skipped_missing: int`, `skipped_guard: int`, `batches: int`.

`apply` CLI: `gmail-sort apply [--config PATH] [--db PATH] [--yes] [--limit N]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_apply.py
import pytest
from gmail_sort.cli import apply_plan, plan_apply
from gmail_sort.store import Decision, Store
from tests.fake_gmail import FakeGmailClient
from tests.helpers import row, cfg, seed_decision

def test_dry_run_writes_nothing(tmp_path):
    with Store(tmp_path / "d.sqlite") as s:
        seed_decision(s, "m1", "Receipts")
        fake = FakeGmailClient({"m1": row("m1")})
        rid = s.start_run("apply", dry_run=True, config_hash="h")
        stats = apply_plan(fake, s, cfg(), plan_apply(s, cfg(), version="v"),
                           dry_run=True, run_id=rid)
    assert fake.modified == [] and stats.applied == 0

def test_apply_adds_labels_and_never_removes_them(tmp_path):
    with Store(tmp_path / "d.sqlite") as s:
        seed_decision(s, "m1", "Receipts")
        fake = FakeGmailClient({"m1": row("m1")})
        rid = s.start_run("apply", dry_run=False, config_hash="h")
        apply_plan(fake, s, cfg(), plan_apply(s, cfg(), version="v"),
                   dry_run=False, run_id=rid)
    assert fake.modified == [("m1", ("Label_1",), ())]

def test_missing_message_does_not_abort_the_run(tmp_path):
    with Store(tmp_path / "d.sqlite") as s:
        for m in ("m1", "m2", "m3"):
            seed_decision(s, m, "Receipts")
        fake = FakeGmailClient({"m1": row("m1"), "m2": row("m2"), "m3": row("m3")},
                               fail_on={"m2"})
        rid = s.start_run("apply", dry_run=False, config_hash="h")
        stats = apply_plan(fake, s, cfg(), plan_apply(s, cfg(), version="v"),
                           dry_run=False, run_id=rid)
    assert stats.applied == 2 and stats.skipped_missing == 1

def test_stale_decision_label_is_dropped_not_applied(tmp_path):
    with Store(tmp_path / "d.sqlite") as s:
        s.upsert_messages([row("m1")])
        s.put_decision(Decision(message_id="m1", label="Removed From Taxonomy",
                                source="rule", rule_name="r",
                                classifier_version="v", decided_at="2026-10-15T00:00:00Z"))
        plan = plan_apply(s, cfg(), version="v")
    assert plan == {}

def test_guard_is_revalidated_immediately_before_writing(tmp_path):
    with Store(tmp_path / "d.sqlite") as s:
        seed_decision(s, "m1", "Receipts")
        # the message gained a user label between classify and apply
        s.upsert_messages([row("m1", label_ids=("INBOX", "Label_9"))])
        fake = FakeGmailClient({"m1": row("m1", label_ids=("INBOX", "Label_9"))},
                               label_ids=[{"id": "Label_9", "name": "Existing",
                                           "type": "user"}])
        rid = s.start_run("apply", dry_run=False, config_hash="h")
        stats = apply_plan(fake, s, cfg(), plan_apply(s, cfg(), version="v"),
                           dry_run=False, run_id=rid)
    assert fake.modified == [] and stats.skipped_guard == 1

def test_second_apply_issues_no_further_batch_calls(tmp_path):
    with Store(tmp_path / "d.sqlite") as s:
        seed_decision(s, "m1", "Receipts")
        seed_decision(s, "m2", "Receipts")
        fake = FakeGmailClient({"m1": row("m1"), "m2": row("m2")})
        for _ in range(2):
            rid = s.start_run("apply", dry_run=False, config_hash="h")
            apply_plan(fake, s, cfg(), plan_apply(s, cfg(), version="v"),
                       dry_run=False, run_id=rid)
    assert len([c for c in fake.calls if c[0] == "batch_modify"]) == 1

def test_plan_groups_by_label(tmp_path):
    with Store(tmp_path / "d.sqlite") as s:
        seed_decision(s, "m1", "Receipts")
        seed_decision(s, "m2", "Newsletters")
        seed_decision(s, "m3", "Receipts")
        plan = plan_apply(s, cfg(), version="v")
    assert sorted(plan) == ["Newsletters", "Receipts"]
    assert sorted(plan["Receipts"]) == ["m1", "m3"]

def test_unsorted_decisions_are_not_planned(tmp_path):
    with Store(tmp_path / "d.sqlite") as s:
        seed_decision(s, "m1", None)
        assert plan_apply(s, cfg(), version="v") == {}

def test_changes_are_logged_before_the_call(tmp_path):
    with Store(tmp_path / "d.sqlite") as s:
        seed_decision(s, "m1", "Receipts")
        fake = FakeGmailClient({"m1": row("m1")}, fail_on={"m1"})
        rid = s.start_run("apply", dry_run=False, config_hash="h")
        apply_plan(fake, s, cfg(), plan_apply(s, cfg(), version="v"),
                   dry_run=False, run_id=rid)
        assert list(s.iter_changes(rid)) == [("m1", "Receipts", "add")]

def test_cli_apply_without_yes_is_a_dry_run(tmp_path, monkeypatch):
    from gmail_sort import cli
    msgs = {"m1": row("m1")}
    fake = FakeGmailClient(msgs)
    monkeypatch.setattr(cli, "_open_client", lambda cfg, db: fake)
    with Store(tmp_path / "d.sqlite") as s:
        seed_decision(s, "m1", "Receipts")
    assert cli.main(["apply", "--config", str(write_cfg(tmp_path)),
                     "--db", str(tmp_path / "d.sqlite")]) == 0
    assert fake.modified == []

def test_cli_apply_with_yes_writes(tmp_path, monkeypatch):
    from gmail_sort import cli
    fake = FakeGmailClient({"m1": row("m1")})
    monkeypatch.setattr(cli, "_open_client", lambda cfg, db: fake)
    with Store(tmp_path / "d.sqlite") as s:
        seed_decision(s, "m1", "Receipts")
    assert cli.main(["apply", "--config", str(write_cfg(tmp_path)),
                     "--db", str(tmp_path / "d.sqlite"), "--yes"]) == 0
    assert fake.modified == [("m1", ("Label_1",), ())]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_apply.py -v`
Expected: FAIL — `ImportError: cannot import name 'plan_apply'`

- [ ] **Step 3: Implement `plan_apply` and `apply_plan`**

`plan_apply` reads `store.iter_decisions(version)`, keeps entries whose `label` is non-`None` **and** is present in `cfg.labels` (Review Focus #4), drops any `(message_id, label)` pair already returned by `store.applied_pairs()`, and groups the rest by label. The exclusion is what makes a second `apply` free and is what re-enables a pair after `undo`. It makes no API calls, so it is trivially testable in isolation.

`apply_plan`, after `start_run`:

1. For each label in the plan, resolve its ID via `list_labels()`, calling `create_label` if absent and caching the result for the run.
2. Re-validate every message with `violation(...)` using the freshly resolved `user_label_ids`, dropping and counting `skipped_guard` (Review Focus #4's sibling: a message that gained a label since classification).
3. Log each `(message_id, label, "add")` to `run_messages` **before** the `batch_modify` call, so a crash mid-write over-reports rather than under-reports.
4. Call `batch_modify(ids, add=[label_id])` — never `remove`. Count `batches`.
5. Catch a `batch_modify` failure for a chunk, retry the chunk's IDs one at a time, count successes as `applied` and failures as `skipped_missing`, and continue with the next chunk (Review Focus #2). Never abort the run.

`dry_run=True` runs steps 1–3 against a throwaway plan and returns without any `batch_modify` call; it logs nothing to `run_messages`. `main` prints the plan summary and, without `--yes`, returns 0 before calling `apply_plan` with `dry_run=False`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_apply.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
git add gmail_sort/cli.py tests/test_apply.py
git commit -m "feat: apply command with dry-run default and guard re-validation"
```

---

### Task 11: `undo` command

**Files:**
- Modify: `gmail_sort/cli.py`
- Test: `tests/test_undo.py`

**Interfaces:**
- Consumes: everything from Tasks 1–10.
- Produces:
  - `def plan_undo(store: Store, run_id: int) -> dict[str, list[str]]` — label name to message IDs, from `run_messages` where `action == "add"`.
  - `def undo_run(client: GmailClient, store: Store, run_id: int, *, dry_run: bool) -> UndoStats` with `class UndoStats` — `labels: int`, `messages: int`, `undone: int`, `skipped_missing: int`.

`undo` CLI: `gmail-sort undo RUN_ID [--config PATH] [--db PATH] [--yes]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_undo.py
from gmail_sort.cli import apply_plan, plan_apply, plan_undo, undo_run
from gmail_sort.store import Store
from tests.fake_gmail import FakeGmailClient
from tests.helpers import row, cfg, seed_decision

def applied(tmp_path):
    s = Store(tmp_path / "d.sqlite")
    seed_decision(s, "m1", "Receipts")
    seed_decision(s, "m2", "Receipts")
    fake = FakeGmailClient({"m1": row("m1"), "m2": row("m2")})
    rid = s.start_run("apply", dry_run=False, config_hash="h")
    apply_plan(fake, s, cfg(), plan_apply(s, cfg(), version="v"),
               dry_run=False, run_id=rid)
    return s, fake, rid

def test_round_trip_restores_the_original_labels(tmp_path):
    s, fake, rid = applied(tmp_path)
    before = fake.modified.copy()
    undo_run(fake, s, rid, dry_run=False)
    assert fake.modified[len(before):] == [("m1", (), ("Label_1",)),
                                           ("m2", (), ("Label_1",))]

def test_plan_undo_reads_the_run_log(tmp_path):
    s, _, rid = applied(tmp_path)
    assert plan_undo(s, rid) == {"Receipts": ["m1", "m2"]}

def test_undo_is_scoped_to_one_run(tmp_path):
    s, fake, rid = applied(tmp_path)
    seed_decision(s, "m9", "Newsletters")
    rid2 = s.start_run("apply", dry_run=False, config_hash="h")
    apply_plan(fake, s, cfg(), plan_apply(s, cfg(), version="v"),
               dry_run=False, run_id=rid2)
    assert plan_undo(s, rid) == {"Receipts": ["m1", "m2"]}
    assert plan_undo(s, rid2) == {"Newsletters": ["m9"]}

def test_undo_skips_a_message_that_no_longer_exists(tmp_path):
    s, fake, rid = applied(tmp_path)
    fake.fail_on = {"m2"}
    stats = undo_run(fake, s, rid, dry_run=False)
    assert stats.undone == 1 and stats.skipped_missing == 1

def test_undo_dry_run_writes_nothing(tmp_path):
    s, fake, rid = applied(tmp_path)
    before = list(fake.modified)
    undo_run(fake, s, rid, dry_run=True)
    assert fake.modified == before

def test_undo_of_a_run_with_no_changes_is_a_no_op(tmp_path):
    with Store(tmp_path / "d.sqlite") as s:
        rid = s.start_run("classify", dry_run=False, config_hash="h")
        fake = FakeGmailClient({})
        assert undo_run(fake, s, rid, dry_run=False).undone == 0
    assert fake.modified == []

def test_undo_logs_remove_actions(tmp_path):
    s, fake, rid = applied(tmp_path)
    undo_run(fake, s, rid, dry_run=False)
    assert ("m1", "Receipts", "remove") in list(s.iter_changes(rid))

def test_plan_apply_re_enables_the_pair_after_undo(tmp_path):
    s, fake, rid = applied(tmp_path)
    assert plan_apply(s, cfg(), version="v") == {}
    undo_run(fake, s, rid, dry_run=False)
    assert plan_apply(s, cfg(), version="v") == {"Receipts": ["m1", "m2"]}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_undo.py -v`
Expected: FAIL — `ImportError: cannot import name 'plan_undo'`

- [ ] **Step 3: Implement `plan_undo` and `undo_run`**

`plan_undo` reads `run_messages` for the run, keeps `action == "add"`, and groups by label. `undo_run` resolves label IDs (never creating one), logs `(message_id, label, "remove")` before each call, then calls `batch_modify(ids, remove=[label_id])`. A failing chunk is retried one ID at a time with failures counted as `skipped_missing`, mirroring `apply_plan`. `dry_run` prints and returns.

`main` for `undo` requires the run ID, an existing run (else exit 2 with a message naming it), and `--yes` to write.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_undo.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Run the whole suite**

Run: `pytest -q`
Expected: PASS, 110 tests, no failures.

- [ ] **Step 6: Commit**

```bash
git add gmail_sort/cli.py tests/test_undo.py
git commit -m "feat: undo command reversing one run from the change log"
```

---

### Task 12: README, OAuth setup walkthrough and live smoke test

**Files:**
- Create: `README.md`
- Modify: `config.example.toml` (only if the walkthrough reveals a stale value)

**Interfaces:**
- Consumes: the finished CLI.
- Produces: no code.

- [ ] **Step 1: Write `README.md`**

Cover, in this order:

1. **What it does and what it will never do** — labels only; never deletes, never removes a label, never touches starred, Important, mail with a large attachment, or mail that already has a label.
2. **Setup** — create a Google Cloud project, enable the Gmail API, configure an OAuth consent screen as External/Testing with your own address as a test user, create a Desktop app OAuth client, download `client_secret.json`, place it beside the database. Note the scope is `gmail.modify` and that Testing-mode tokens expire after 7 days, so re-running `survey` re-triggers consent.
3. **The four commands** with a worked first-run sequence: `survey` → edit `config.proposed.toml` into `config.toml` and *uncomment the taxonomy* → `classify --estimate` → `classify` → `apply` (dry run) → `apply --yes` → `undo RUN_ID`.
4. **Timing and cost expectations** — the spec's Cost table, including that a cold `classify` over 50,000 messages is hours of wall clock because `messages.get` costs 20 quota units against a 6,000/minute ceiling, and that re-runs are near-instant because decisions and bodies are cached.
5. **Troubleshooting** — `403 insufficientPermissions` (wrong scope or a stale token: delete `token.json`), `429` (expected; the bucket handles it, a persistent one means concurrent runs), and "nothing was labelled" (check that the taxonomy block is uncommented and that rules name active labels).

- [ ] **Step 2: Run the live smoke test**

Run, against the real mailbox:

```bash
python -m gmail_sort survey --db /tmp/gmail-sort-smoke.sqlite --out /tmp/proposed.toml
```

Expected: the OAuth consent screen appears once; the command prints a candidate count and top domains; `/tmp/proposed.toml` exists and contains a commented `[taxonomy]` block. This is the only step that touches the real mailbox, and `survey` cannot write to Gmail.

Verify nothing was written:

```bash
python -c "import tomllib,sys; t=tomllib.load(open('/tmp/proposed.toml','rb')); assert 'taxonomy' not in t, 'taxonomy must stay commented out'; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Confirm the guard query selects what you expect**

In the Gmail search box, paste the exact query from Global Constraints:

```
has:nouserlabels -is:starred -is:important -larger:5M
```

Expected: the result count is in the same order of magnitude as `survey`'s reported candidate count, and the visible list contains no starred or Important mail. A large discrepancy means the config query was edited away from the default.

- [ ] **Step 4: Commit**

```bash
git add README.md config.example.toml
git commit -m "docs: setup walkthrough, cost expectations and smoke test"
```
