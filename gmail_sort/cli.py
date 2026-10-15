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

from gmail_sort.config import Config, ConfigError, load_config
from gmail_sort.gmail import GmailClient, RestGmailClient
from gmail_sort.guards import SYSTEM_LABEL_IDS, guard_query

SCOPE = "https://www.googleapis.com/auth/gmail.modify"

DEFAULT_DB = "gmail-sort.sqlite"
DEFAULT_PROPOSAL = "config.proposed.toml"
DEFAULT_SAMPLE = 1000


@dataclass(frozen=True)
class Survey:
    total: int
    sampled: int
    domains: list[tuple[str, int]]
    categories: list[tuple[str, int]]
    labels: list[tuple[str, int]]
    existing_labels: list[str]


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
