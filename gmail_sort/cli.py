"""The four commands: survey, classify, apply, undo.

`_open_client` is the seam every test replaces, so the commands are exercised
end-to-end against the in-memory fake without touching Gmail.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import httpx

from gmail_sort import rules
from gmail_sort.config import Config, ConfigError, classifier_version, load_config
from gmail_sort.gmail import GmailClient, RestGmailClient, chunked
from gmail_sort.guards import SYSTEM_LABEL_IDS, guard_query, user_label_ids, violation
from gmail_sort.llm import PROMPT_TEMPLATE, Classifier, estimate, make_classifier
from gmail_sort.store import Decision, MessageRow, Store, now

SCOPE = "https://www.googleapis.com/auth/gmail.modify"

DEFAULT_DB = "gmail-sort.sqlite"
DEFAULT_PROPOSAL = "config.proposed.toml"
DEFAULT_SAMPLE = 1000

# Messages per LLM request. Small enough that one bad batch is cheap to retry.
LLM_BATCH_SIZE = 50


@dataclass(frozen=True)
class Survey:
    total: int
    sampled: int
    domains: list[tuple[str, int]]
    categories: list[tuple[str, int]]
    labels: list[tuple[str, int]]
    existing_labels: list[str]


@dataclass(frozen=True)
class ClassifyStats:
    candidates: int = 0
    cached: int = 0
    rule_matched: int = 0
    llm_matched: int = 0
    unsorted: int = 0
    llm_calls: int = 0
    gets: int = 0
    skipped_guard: int = 0
    estimated_tokens: int = 0


def classifier_version_for(cfg: Config) -> str:
    """The single place the decision cache key is computed."""
    return classifier_version(
        cfg.labels, cfg.rules, PROMPT_TEMPLATE, cfg.llm_model
    )


def classify(
    client: GmailClient,
    cfg: Config,
    store: Store,
    *,
    llm: Classifier | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> ClassifyStats:
    """Rules first, the LLM only for what is left, every decision cached.

    Never writes to Gmail. With `dry_run` nothing is persisted either, which is
    what `--estimate` uses.
    """
    version = classifier_version_for(cfg)
    known_user_labels = user_label_ids(client.list_labels())
    counters: Counter[str] = Counter()
    pending: list[MessageRow] = []

    for message_id in client.iter_message_ids(guard_query(cfg)):
        if store.get_decision(message_id, version) is not None:
            counters["cached"] += 1
            continue

        message = client.get_message(message_id)
        reason = violation(
            message,
            user_label_ids=known_user_labels,
            min_size_bytes=cfg.min_size_bytes,
        )
        if reason is not None:
            counters["skipped_guard"] += 1
            continue

        counters["candidates"] += 1
        if not dry_run:
            store.upsert_messages([message])

        match = rules.classify(message, cfg.rules)
        if match is not None:
            label, rule_name = match
            if not dry_run:
                _save_decision(store, message.id, label, "rule", rule_name, version)
            counters["rule_matched"] += 1
            continue
        pending.append(message)

    if dry_run:
        counters["estimated_tokens"] = estimate(pending, cfg.labels)[1]
        return _stats(counters)

    if llm is None:
        # --no-llm: record the leftovers as undecided rather than guessing.
        for message in pending:
            _save_decision(store, message.id, None, "rule", None, version)
            counters["unsorted"] += 1
        return _stats(counters)

    for chunk in chunked(pending[:limit] if limit is not None else pending,
                         LLM_BATCH_SIZE):
        bodies: list[MessageRow] = []
        for message in chunk:
            body = message
            if message.excerpt is None:
                # The only format=full fetch in the program.
                body = client.get_message(message.id, full=True)
                counters["gets"] += 1
                if cfg.store_bodies:
                    store.upsert_messages([body])
            bodies.append(body)

        answers = llm.classify(bodies, cfg.labels)
        counters["llm_calls"] += 1
        for message, label in zip(bodies, answers):
            _save_decision(store, message.id, label, "llm", None, version)
            counters["unsorted" if label is None else "llm_matched"] += 1

    return _stats(counters)


def survey(
    client: GmailClient, cfg: Config, *, top: int = 50, sample: int = DEFAULT_SAMPLE
) -> Survey:
    """Count what is in the candidate set without changing anything.

    Listing every candidate is cheap (5 quota units per 500 messages). Fetching
    a message to read its sender is not (20 units each), so senders, categories
    and labels are counted from the first `sample` candidates. The reported
    total is exact; the breakdowns are a sample.
    """
    candidates = list(client.iter_message_ids(guard_query(cfg)))
    domains: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    labels: Counter[str] = Counter()

    sampled_ids = candidates[:sample]
    for message_id in sampled_ids:
        message = client.get_message(message_id)
        if message.from_domain:
            domains[message.from_domain] += 1
        if message.category:
            categories[message.category] += 1
        for label_id in message.label_ids:
            if label_id not in SYSTEM_LABEL_IDS:
                labels[label_id] += 1

    existing = sorted(
        label.name for label in client.list_labels() if label.type == "user"
    )

    return Survey(
        total=len(candidates),
        sampled=len(sampled_ids),
        domains=domains.most_common(top),
        categories=categories.most_common(top),
        labels=labels.most_common(top),
        existing_labels=existing,
    )


def render_proposal(result: Survey, cfg: Config, *, top: int = 50) -> str:
    """A valid TOML config whose [taxonomy] block is commented out for review.

    An unreviewed active taxonomy is the one way this tool labels 50,000
    messages wrongly in a single run, so the block ships commented.
    """
    labels = [domain for domain, _count in result.domains[:top]]
    lines = [
        "[candidates]",
        f"query = {_toml_string(cfg.query)}",
        "",
        "[fetch]",
        f"excerpt_chars = {cfg.excerpt_chars}",
        f"store_bodies = {_toml_bool(cfg.store_bodies)}",
        "",
        "[llm]",
        f"provider = {_toml_string(cfg.llm_provider)}",
        f"model = {_toml_string(cfg.llm_model)}",
        f"max_messages_per_run = {cfg.max_messages_per_run}",
        "",
        f"# {result.total} candidates in your mailbox.",
    ]
    if result.sampled < result.total:
        lines.append(
            f"# Senders below are counted from a sample of {result.sampled}."
        )
    lines += [
        "#",
        "# The block below is commented out on purpose. Edit the labels to taste,",
        '# then remove the leading "# " from each of its lines to activate it.',
        "# [taxonomy]",
        "# labels = [",
    ]
    lines += [f"#   {_toml_string(label)}," for label in labels]
    lines += ["# ]", ""]

    for index, (domain, _count) in enumerate(result.domains[:top], start=1):
        lines += [
            "# [[rules]]",
            f"# name = {_toml_string(f'sender-{index}')}",
            f"# priority = {100 + index}",
            f"# label = {_toml_string(domain)}",
            f"# from_domain = [{_toml_string(domain)}]",
            "",
        ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "survey":
            return _survey_command(args)
        if args.command == "classify":
            return _classify_command(args)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.print_help()
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gmail-sort",
        description="Label a Gmail backlog so it can be reviewed and deleted.",
    )
    subparsers = parser.add_subparsers(dest="command")

    survey_parser = subparsers.add_parser(
        "survey", help="count candidates and propose a taxonomy (writes nothing)"
    )
    _add_common(survey_parser)
    survey_parser.add_argument("--out", default=DEFAULT_PROPOSAL,
                               help=f"where to write the proposal (default: {DEFAULT_PROPOSAL})")
    survey_parser.add_argument("--top", type=int, default=50,
                               help="how many senders to report (default: 50)")
    survey_parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                               help="how many candidates to fetch for the sender "
                                    f"breakdown (default: {DEFAULT_SAMPLE})")

    classify_parser = subparsers.add_parser(
        "classify", help="decide labels for every candidate (writes nothing to Gmail)"
    )
    _add_common(classify_parser)
    classify_parser.add_argument("--estimate", action="store_true",
                                 help="report what would be sent to the LLM and exit")
    classify_parser.add_argument("--limit", type=int, default=None,
                                 help="cap how many messages reach the LLM this run")
    classify_parser.add_argument("--no-llm", action="store_true",
                                 help="rules only; leave the rest undecided")

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--db", default=DEFAULT_DB)


def _survey_command(args) -> int:
    cfg = load_config(args.config)
    client = _open_client(cfg, args.db)
    result = survey(client, cfg, top=args.top, sample=args.sample)
    out = Path(args.out)
    out.write_text(render_proposal(result, cfg, top=args.top))

    print(f"{result.total} candidates match the guard query.")
    if result.sampled < result.total:
        print(f"Sender breakdown sampled from the first {result.sampled}.")
    for domain, count in result.domains[:10]:
        print(f"  {count:6d}  {domain}")
    print(f"Proposal written to {out} — edit it, then rename to config.toml.")
    return 0


def _classify_command(args) -> int:
    cfg = load_config(args.config)
    with Store(args.db) as store:
        client = _open_client(cfg, args.db)
        if args.estimate:
            stats = classify(client, cfg, store, dry_run=True)
            would_ask = stats.candidates - stats.rule_matched
            print(f"{stats.candidates} candidates: {stats.rule_matched} matched by rules,"
                  f" {would_ask} would go to the LLM")
            print(f"estimated {stats.estimated_tokens} tokens"
                  f" ({stats.skipped_guard} skipped by guards)")
            return 0
        llm = None if args.no_llm else make_classifier(cfg)
        stats = classify(client, cfg, store, llm=llm, limit=args.limit)

    print(f"candidates {stats.candidates}"
          f" (already decided {stats.cached}, skipped by guards {stats.skipped_guard})")
    print(f"labelled: rules {stats.rule_matched}, llm {stats.llm_matched},"
          f" unsorted {stats.unsorted}")
    print("Nothing was written to Gmail. Review with `apply` (dry run by default).")
    return 0


def _save_decision(
    store: Store,
    message_id: str,
    label: str | None,
    source: str,
    rule_name: str | None,
    version: str,
) -> None:
    store.put_decision(
        Decision(
            message_id=message_id,
            label=label,
            source=source,
            rule_name=rule_name,
            classifier_version=version,
            decided_at=now(),
        )
    )


def _stats(counters: Counter) -> ClassifyStats:
    return ClassifyStats(
        candidates=counters["candidates"],
        cached=counters["cached"],
        rule_matched=counters["rule_matched"],
        llm_matched=counters["llm_matched"],
        unsorted=counters["unsorted"],
        llm_calls=counters["llm_calls"],
        gets=counters["gets"],
        skipped_guard=counters["skipped_guard"],
        estimated_tokens=counters["estimated_tokens"],
    )


def _open_client(cfg: Config, db_path: str) -> GmailClient:
    """OAuth the first time, reuse token.json afterwards.

    Deliberately thin: every test replaces this function, and the live smoke
    test is what exercises it.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    directory = Path(db_path).parent
    token_path = directory / "token.json"
    secret_path = directory / "client_secret.json"

    credentials = None
    if token_path.exists():
        credentials = Credentials.from_authorized_user_file(
            str(token_path), [SCOPE]
        )

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            if not secret_path.exists():
                raise ConfigError(
                    f"client_secret.json not found at {secret_path}. Create a "
                    "Desktop app OAuth client in Google Cloud, download its JSON, "
                    "and put it there. See README.md."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(secret_path), [SCOPE])
            credentials = flow.run_local_server(port=0)
        token_path.write_text(credentials.to_json())
        token_path.chmod(0o600)

    def refresh() -> str:
        credentials.refresh(Request())
        token_path.write_text(credentials.to_json())
        return credentials.token

    return RestGmailClient(
        httpx.Client(timeout=60.0),
        credentials.token,
        token_refresher=refresh,
        excerpt_chars=cfg.excerpt_chars,
    )


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _toml_bool(value: bool) -> str:
    return "true" if value else "false"


if __name__ == "__main__":
    sys.exit(main())
