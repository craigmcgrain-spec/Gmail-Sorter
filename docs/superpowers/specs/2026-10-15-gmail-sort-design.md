# Gmail Sort — Design

Date: 2026-10-15

## Problem

A Gmail mailbox has accumulated 5,000–50,000 messages that were never sorted.
Reviewing them by hand is impractical, and Gmail's own filters only see headers,
so they cannot categorise most of it.

The goal is a **re-runnable batch job** that applies labels to the backlog so
that the resulting label set is a review surface: the user opens a label, sees
what is in it, and deletes what they no longer want. The tool's responsibility
ends at labelling.

## Goals

- Label a 5,000–50,000 message backlog so it can be reviewed and deleted in bulk.
- Be safely re-runnable: a second run must not re-do work or re-spend money.
- Be cheap to iterate on. Getting a taxonomy right takes several attempts, and
  each attempt must not mean another full walk of the mailbox.
- Never lose data, never make an irreversible change.

## Non-goals

- No daemon, no watcher, no `users.watch`. This is a batch job, not a service.
- No deletion, ever. Not to Trash, not to Spam.
- No changes to Inbox state, read state, or star state.
- No removal of existing labels.
- No creation of server-side Gmail filters (`settings.filters.*` needs a broader
  scope than `gmail.modify`, and header-only matching cannot reach the tail this
  tool exists for).
- No multi-account support.

## Decisions from brainstorming

| Decision | Choice |
|---|---|
| Job shape | Re-runnable batch over the existing backlog |
| Action on a message | Apply labels only; user reviews and deletes |
| Classification | Deterministic rules fast-path, LLM for the leftovers |
| Taxonomy source | Proposed from mailbox data, approved by the user, saved to config |
| Stack | Python CLI + Gmail API (OAuth) |
| LLM backend | Swappable behind one interface |
| Off-limits mail | Starred, Important, and messages with a large attachment |

Decisions taken at spec-writing time, where the user left the choice open:

- `apply` performs a dry run unless `--yes` is passed.
- `survey` writes a **commented-out draft** taxonomy, not an active one. An
  unreviewed active taxonomy is the single way this tool labels 50,000 messages
  wrongly in one run.
- Message bodies are cached locally (truncated), so re-tuning rules and prompts
  costs no API calls at all.

## Verified API facts

These are load-bearing, so they were checked against primary sources rather than
recalled.

| Fact | Value | Source |
|---|---|---|
| Search operators | `has:nouserlabels`, `has:userlabels`, `larger:`/`size:` with `K`/`M`/`G`, `is:important`, `is:starred` all exist | [Gmail help](https://support.google.com/mail/answer/7190?hl=en) |
| `messages.batchModify` limit | 1000 IDs per request | [API ref](https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.messages/batchModify) |
| `messages.list` cost | 5 quota units | [Usage limits](https://developers.google.com/workspace/gmail/api/reference/quota) |
| `messages.get` cost | 20 quota units | same |
| `messages.batchModify` cost | 50 quota units | same |
| `labels.create` cost | 5 quota units | same |
| Per-minute per-user quota | 6,000 quota units | same |

The binding constraint is **20 units per `messages.get` against 6,000 units per
minute**: 300 message fetches per minute, per user. A cold `classify` over 50,000
messages therefore takes roughly 2.8 hours of wall clock. This shapes the design
more than anything else and is why redundant fetches are treated as expensive.

## Architecture

```
survey    → list candidates, count senders/domains/categories, propose taxonomy
classify  → rules + LLM over candidates, decisions cached in SQLite   (no Gmail writes)
apply     → batched batchModify, appends labels                       (Gmail writes)
undo RUN  → removes exactly what one run added                        (Gmail writes)
```

Each stage commits its progress to SQLite as it goes, so interrupting a stage and
re-running it resumes rather than restarts.

### Candidate selection

Every guard collapses into a single Gmail query, stored in config:

```
has:nouserlabels -is:starred -is:important -larger:5M
```

| Clause | Meaning |
|---|---|
| `has:nouserlabels` | Already-labelled mail is the user's own judgement; leave it alone. Also what makes re-runs idempotent. |
| `-is:starred` | Never touch starred mail. |
| `-is:important` | Never touch Important mail. |
| `-larger:5M` | Never touch mail with a large attachment. Threshold is a config knob. |

Guards expressed as a query means guarded messages are never listed, never
fetched, and never paid for.

**Guards are re-validated locally** after a message is fetched, before any write.
Two reasons: Gmail search is thread-oriented, so `-is:starred` can surface a
conversation that contains an unstarred message; and `larger:` is approximate
whereas `sizeEstimate` is the value the API actually reports. The query narrows;
the local check decides.

### Repo layout

```
gmail_sort/
  cli.py         # argparse, the four commands, dry-run printing
  config.py      # load + validate config.toml
  gmail.py       # GmailClient protocol, REST implementation, rate limiter
  store.py       # SQLite schema and queries
  rules.py       # pure rule engine
  llm.py         # provider-swappable classifier
tests/
  fake_gmail.py  # in-memory GmailClient
  ...
config.example.toml
pyproject.toml
README.md
```

## Configuration

`config.toml`:

```toml
[candidates]
query = "has:nouserlabels -is:starred -is:important -larger:5M"

[fetch]
# Body excerpt length cached for LLM classification, in characters.
excerpt_chars = 2000
# Cache truncated bodies so re-tuning rules and prompts makes no API calls.
store_bodies = true

[llm]
provider = "anthropic"          # anthropic | openai
model = "claude-haiku-4-5"
max_messages_per_run = 0        # 0 = unlimited; positive value caps LLM spend

[taxonomy]
# Active labels. classify may only assign one of these.
labels = [
  "Newsletters",
  "Receipts",
  "Notifications/CI",
  "Notifications/Social",
]

# First match wins, lowest priority number first. Every rule must name an
# active taxonomy label. A rule matches if any listed field matches; at least
# one field must be present.
[[rules]]
name = "ci-notifications"
priority = 10
label = "Notifications/CI"
from_domain = ["github.com", "gitlab.com", "circleci.com"]
```

A rule matches on any combination of `from_domain`, `from` (regex), `subject`
(regex), `list_id`, and Gmail `category`. Validation fails fast, at config load,
on an unknown label, a duplicate rule name, an invalid regex, or an unknown LLM
provider.

`survey` writes its proposal to `config.proposed.toml`, with the `[taxonomy]`
block commented out.

## Storage

SQLite, four tables:

```sql
CREATE TABLE messages (
  id            TEXT PRIMARY KEY,
  thread_id     TEXT NOT NULL,
  from_addr     TEXT NOT NULL,
  from_domain   TEXT NOT NULL,
  subject       TEXT,
  list_id       TEXT,
  category      TEXT,
  size_estimate INTEGER NOT NULL,
  label_ids     TEXT NOT NULL,      -- JSON array, as of fetched_at
  excerpt       TEXT,               -- truncated body, only when store_bodies
  fetched_at    TEXT NOT NULL
);

CREATE TABLE decisions (
  message_id         TEXT PRIMARY KEY REFERENCES messages(id),
  label              TEXT,          -- NULL means UNSORTED
  source             TEXT NOT NULL, -- 'rule' | 'llm'
  rule_name          TEXT,
  classifier_version TEXT NOT NULL,
  decided_at         TEXT NOT NULL
);

CREATE TABLE runs (
  id          INTEGER PRIMARY KEY,
  kind        TEXT NOT NULL,        -- 'classify' | 'apply' | 'undo'
  dry_run     INTEGER NOT NULL,
  started_at  TEXT NOT NULL,
  finished_at TEXT,
  config_hash TEXT NOT NULL
);

CREATE TABLE run_messages (
  run_id     INTEGER NOT NULL REFERENCES runs(id),
  message_id TEXT NOT NULL,
  label      TEXT NOT NULL,
  action     TEXT NOT NULL          -- 'add' | 'remove'
);
```

`run_messages` **is** the undo log.

`classifier_version` is a hash of the taxonomy, the rules, the prompt template,
and the model id. Any change to those invalidates cached decisions, which is the
point: edit a rule, re-run `classify`, and only messages whose decision actually
changed are re-examined. Because bodies are cached, re-examination makes no API
calls at all. This is the economics that justify the cache's existence.

## Rules engine

`rules.py` exposes one pure function:

```python
def classify(message: MessageRow, rules: list[Rule]) -> Decision | None
```

First match by priority wins. No I/O, no Gmail types, so the whole engine is
table-testable. Returning `None` means "no rule matched, send to the LLM".

## LLM classifier

Only messages with no rule match and no cached decision reach the LLM.

- Prompt: the taxonomy, then the message's from/subject/excerpt.
- Output constrained to exactly one taxonomy label or `UNSORTED`; anything else
  is retried once and then recorded as `UNSORTED` rather than guessed.
- Temperature 0.
- Result cached permanently against `classifier_version`.

`llm.py` exposes one interface — `classify(batch) -> list[str | None]` — with
provider implementations selected by config. The provider is the only thing that
changes between backends.

`classify --estimate` reports how many messages would reach the LLM and the
estimated token count, without calling anyone. `max_messages_per_run` caps spend
for a run.

## Apply stage

1. Read decisions where `source` is `rule` or `llm` and label is non-NULL.
2. Group message IDs by target label.
3. Create any missing label (`labels.create`; `/` in a name produces Gmail's
   nesting) and cache its ID.
4. Re-validate guards against each message's cached metadata; drop and report
   anything that now fails.
5. `batchModify` in chunks of 1000, populating **only** `addLabelIds`.
   `removeLabelIds` is never set.
6. Record every change in `run_messages` *before* the call, so a crash mid-write
   leaves a log that at worst over-reports what was applied.

`addLabelIds` is idempotent: applying a label a message already has is a no-op.

Without `--yes`, `apply` prints the planned changes and exits. Nothing is written.

## Pacing

A token bucket refills at 100 units/second (6,000/minute) with per-method costs
(5 list, 20 get, 50 batchModify), so the client throttles *before* Gmail does.
429 and 5xx responses use truncated exponential backoff, maximum 5 attempts,
capped at 64s, then fail the run with progress committed.

Redundant fetches are avoided deliberately: the rules pass uses
`format=metadata` with a header allow-list, and a `format=full` fetch happens
only for messages that reach the LLM. When rules match a large share of the
mailbox, the average cost per message stays close to 20 units rather than 40.

## Safety

- `apply` requires `--yes`; otherwise dry run.
- Guards re-validated per message immediately before writing.
- Labels are only ever added. Never removed, never a delete, never a state change.
- Every applied change is recorded and reversible with `undo <run-id>`.
- `undo` removes exactly the labels in `run_messages` for that run. Documented
  caveat: if the user adds the same label manually afterwards, undo removes it
  too. Messages that have since been deleted are logged and skipped rather than
  aborting the undo.

## Testing

| Level | Coverage |
|---|---|
| Unit | Rule engine: priority order, each match field, no-match. Guard logic: starred, important, oversize, re-check rejects. `classifier_version` changes on rule/taxonomy/prompt/model change and not otherwise. Chunking at the 1000 boundary. |
| Integration | `classify` and `apply` end-to-end against the in-memory `FakeGmailClient`. Apply→undo round trip restores the original labels. |
| Idempotency | A second `classify` makes zero LLM calls and zero `messages.get` calls. A second `apply` performs zero `batchModify` calls. |
| Live smoke | `survey` against the real mailbox. Read-only, so it is safe as the one manual check. |

The rate limiter is tested against a fake clock; no test sleeps.

## Dependencies

- `google-auth-oauthlib` for the OAuth installed-app flow and token refresh —
  the fiddly part, so it uses the boring documented library.
- `httpx` for the five REST calls (`messages.list`, `messages.get`,
  `messages.batchModify`, `labels.list`, `labels.create`). Chosen over
  `google-api-python-client` because five small functions are easier to reason
  about and far easier to fake in tests than a discovery-based client object.

OAuth scope: `https://www.googleapis.com/auth/gmail.modify`.

## Cost

Gmail API usage is free at these volumes; the daily billing threshold is
80,000,000 units per project.

| Workload | Quota units |
|---|---|
| `survey` over 50,000 messages | ~500 (list calls only) |
| Cold `classify`, 100% LLM path | ~2,000,000 → ~5.5 h |
| Cold `classify`, 60% rule match | ~1,400,000 → ~3.9 h |
| Warm `classify` (all cached) | 0 |
| `apply`, 50,000 messages at 1000/batch | ~2,500 |

LLM cost is reported by `classify --estimate` before spending.
