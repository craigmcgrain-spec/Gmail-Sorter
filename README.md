# gmail-sort

Labels a large Gmail backlog so you can review each label and delete what you no
longer want. Rules handle the bulk; an LLM picks up whatever the rules miss.

## What it does

Four commands over a local SQLite cache:

| Command | Writes to Gmail? | What it does |
|---|---|---|
| `survey` | no | Counts candidates matching your guard query and proposes a taxonomy |
| `classify` | no | Decides a label per message (rules first, LLM for the rest), cached locally |
| `apply` | **yes** | Adds the decided labels, in batches of 1000 |
| `undo RUN_ID` | **yes** | Removes exactly the labels one `apply` run added |

## What it will never do

- **Never deletes.** Not to Trash, not to Spam. Nothing in this program removes a message.
- **Never removes a label**, except `undo`, which removes only what that run added.
- **Never touches** starred mail, Important mail, mail with an attachment over the
  size threshold, or mail that already has a label of yours.
- **Never changes Inbox or read state.** Archiving is not implemented.
- **Never touches a message it has not listed first.** The guards are part of the
  Gmail search query, so guarded mail is never fetched or paid for.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && .venv/bin/pip install -e .
```

Optionally, to run the test suite (135 tests, no network or credentials needed):

```bash
.venv/bin/pip install -e ".[dev]" && .venv/bin/python -m pytest -q
```

Then create an OAuth client, once:

1. Create a Google Cloud project and enable the **Gmail API**.
2. Configure the OAuth consent screen as **External**, and add your own address as a
   **test user** (the app never needs verification for this).
3. Create credentials of type **Desktop app** and download the JSON.
4. Save it as `client_secret.json` next to your database (`gmail-sort.sqlite`).

That is everything needed before the first command. `survey` needs no `config.toml` and no
API key, and the database file is created on first use. The first run opens a browser for
consent and caches the token in `token.json` beside the database.

The scope requested is `https://www.googleapis.com/auth/gmail.modify`, the narrowest
that can add labels. The token is cached in `token.json` beside the database, with mode
`600`. Both files are git-ignored.

> In **Testing** publishing status Google expires refresh tokens after 7 days, so you
> will be asked to consent again roughly weekly. Publishing the app, or using a
> Workspace-internal app, avoids that.

## First run

```bash
# 1. Look before touching anything. Read-only.
.venv/bin/python -m gmail_sort survey --db gmail-sort.sqlite

# 2. Edit the proposal: uncomment the [taxonomy] block and fix the labels.
#    (Delete the leading "# " from each line of that block.)
mv config.proposed.toml config.toml
$EDITOR config.toml

# 3. What would the LLM cost?
.venv/bin/python -m gmail_sort classify --estimate

# 4. Decide labels. Writes nothing to Gmail. Resumable; safe to Ctrl-C.
.venv/bin/python -m gmail_sort classify

# 5. See exactly what would change.
.venv/bin/python -m gmail_sort apply

# 6. Do it.
.venv/bin/python -m gmail_sort apply --yes

# 7. Changed your mind about a run?
.venv/bin/python -m gmail_sort undo 1 --yes
```

Re-running `classify` costs nothing: decisions and message bodies are cached and keyed by
a hash of your taxonomy, rules, prompt and model. Edit a rule and only messages whose
decision actually changed are re-examined — from the local cache, without refetching.

## Configuration

`config.toml`:

```toml
[candidates]
query = "has:nouserlabels -is:starred -is:important -larger:5M"

[fetch]
excerpt_chars = 2000
store_bodies = true

[llm]
provider = "anthropic"        # anthropic | openai
model = "claude-haiku-4-5"
max_messages_per_run = 0      # 0 = unlimited

[taxonomy]
labels = ["Newsletters", "Receipts", "Notifications/CI"]

[[rules]]
name = "ci-notifications"
priority = 10
label = "Notifications/CI"
from_domain = ["github.com", "gitlab.com"]
```

Guard query, in order of consequence:

| Clause | Meaning |
|---|---|
| `has:nouserlabels` | Leave mail you already sorted alone. Also what makes re-runs idempotent. |
| `-is:starred` | Never touch starred mail. |
| `-is:important` | Never touch Important mail. |
| `-larger:5M` | Never touch mail with an attachment over 5 MB. |

A rule matches when **every** field it declares matches. First match wins, ordered by
`priority` then `name`. Every rule's `label` must appear in `taxonomy.labels`, or the
config is rejected at load.

A label name containing `/` becomes nested in Gmail (`Notifications/CI` shows under
`Notifications`).

### Environment

```bash
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY, matching llm.provider
```

`classify --estimate` needs no key.

## Timing and cost

Gmail's limit is 6,000 quota units per minute per user, and `messages.get` costs 20 of
them — 300 fetches per minute. That, not CPU, is what makes this slow:

| Workload | Quota units | Wall clock |
|---|---|---|
| `survey` over 50,000 messages | ~500 to list + ~20,000 to sample 1,000 senders | ~3.5 min |
| Cold `classify`, everything reaching the LLM | ~2,000,000 | ~5.5 h |
| Cold `classify`, 60% caught by rules | ~1,400,000 | ~3.9 h |
| Warm `classify` (all cached) | 0 | seconds |
| `apply` over 50,000 messages | ~2,500 | seconds |

The Gmail API itself is free at these volumes. LLM cost is what `--estimate` reports.

Useful flags: `classify --limit N` caps messages sent to the LLM this run;
`apply --limit N` caps labels applied; `classify --no-llm` runs rules only.

## Troubleshooting

- **`error: config not found`** — copy `config.example.toml` to `config.toml`.
- **`taxonomy.labels is empty`** — you have not uncommented the `[taxonomy]` block.
- **`403 insufficientPermissions`** — wrong scope, or a stale token. Delete
  `token.json` and re-run.
- **`client_secret.json not found at ...`** — it lives next to the database, not next to
  the config.
- **`429`, or `403 userRateLimitExceeded`** — expected under load; the client paces itself
  and backs off. A persistent one means another run is using the same quota.
- **Nothing was labelled** — check the taxonomy block is uncommented and each rule names
  an active label. `apply` prints how many messages each guard rejected.
- **A thread looks partly labelled** — Gmail's UI is thread-oriented; this labels
  individual messages.

## Caveats

- **Guards are enforced against the mailbox as it was when each message was
  classified.** If you star a message, mark it Important or label it by hand *after*
  `classify` has decided it, `apply` will still label it — and re-running `classify` will
  not notice either, because a decided message is never refetched (that caching is what
  makes re-tuning free). To force a fresh look at the guards, delete the database and
  re-classify.
- **A label named like one of Gmail's own labels is rejected at load.** `TRASH`, `SPAM`,
  `INBOX`, `UNREAD`, `STARRED`, `IMPORTANT` and the `CATEGORY_*` names would change the
  message rather than file it, so pick another name.
- **`undo` removes the label even if you added it by hand afterwards.** It reverses the
  exact message/label pairs that run logged.
- Nested labels are created with `labelListVisibility: labelShow`; a label hidden from
  Gmail's sidebar will not appear until you show it.
- The survey's sender breakdown is sampled (`--sample`, default 1000), so a sender
  concentrated deep in the backlog may not be proposed. The candidate total is exact.
