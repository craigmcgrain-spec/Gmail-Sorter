import tomllib

import pytest

from gmail_sort.cli import _open_client, render_proposal, survey
from gmail_sort.config import ConfigError
from tests.fake_gmail import FakeGmailClient
from tests.helpers import cfg, row, write_cfg


def test_counts_domains_and_categories():
    msgs = {f"m{i}": row(f"m{i}", from_domain=d, category=c)
            for i, (d, c) in enumerate([("a.com", "CATEGORY_PROMOTIONS"),
                                        ("a.com", "CATEGORY_PROMOTIONS"),
                                        ("b.com", "CATEGORY_UPDATES")])}
    result = survey(FakeGmailClient(msgs), cfg())
    assert result.total == 3
    assert result.domains[:2] == [("a.com", 2), ("b.com", 1)]
    assert result.categories[0] == ("CATEGORY_PROMOTIONS", 2)


def test_survey_makes_no_writes():
    fake = FakeGmailClient({}, label_ids=[{"id": "Label_1", "name": "Old",
                                           "type": "user"}])
    survey(fake, cfg())
    assert not any(c[0] in ("batch_modify", "create_label") for c in fake.calls)
    assert fake.modified == []


def test_survey_uses_the_guard_query():
    fake = FakeGmailClient({})
    survey(fake, cfg())
    listing = [c for c in fake.calls if c[0] == "iter_message_ids"]
    assert listing
    assert listing[0][1][0].startswith("has:nouserlabels")


def test_survey_lists_every_candidate_but_samples_the_fetches():
    msgs = {f"m{i}": row(f"m{i}") for i in range(50)}
    fake = FakeGmailClient(msgs)
    result = survey(fake, cfg(), sample=10)
    assert result.total == 50
    assert result.sampled == 10
    assert len([c for c in fake.calls if c[0] == "get_message"]) == 10


def test_zero_candidates_is_not_an_error():
    result = survey(FakeGmailClient({}), cfg())
    assert result.total == 0
    assert result.domains == []
    assert result.categories == []


def test_proposal_comments_out_the_taxonomy():
    text = render_proposal(survey(FakeGmailClient({}), cfg()), cfg())
    assert "# [taxonomy]" in text
    assert "\n[taxonomy]" not in text


def test_proposal_suggests_labels_from_top_domains():
    msgs = {f"m{i}": row(f"m{i}", from_domain="github.com") for i in range(5)}
    text = render_proposal(survey(FakeGmailClient(msgs), cfg()), cfg())
    assert "github.com" in text


def test_uncommenting_the_taxonomy_block_yields_valid_toml():
    msgs = {f"m{i}": row(f"m{i}", from_domain="github.com") for i in range(5)}
    text = render_proposal(survey(FakeGmailClient(msgs), cfg()), cfg())
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("# [taxonomy]"))
    uncommented = [
        line[2:] if line.startswith("# ") else ("" if line == "#" else line)
        for line in lines[start:]
    ]
    parsed = tomllib.loads("\n".join(lines[:start] + uncommented))
    assert parsed["taxonomy"]["labels"] == ["github.com"]


def test_main_survey_writes_the_proposal_file(tmp_path, monkeypatch):
    from gmail_sort import cli

    msgs = {f"m{i}": row(f"m{i}") for i in range(3)}
    monkeypatch.setattr(cli, "_open_client", lambda config, db: FakeGmailClient(msgs))
    out = tmp_path / "config.proposed.toml"
    assert cli.main(["survey", "--config", str(write_cfg(tmp_path)),
                     "--db", str(tmp_path / "d.sqlite"), "--out", str(out)]) == 0
    assert out.exists()
    assert "# [taxonomy]" in out.read_text()


def test_open_client_names_a_missing_client_secret(tmp_path):
    with pytest.raises(ConfigError, match="client_secret.json"):
        _open_client(cfg(), str(tmp_path / "d.sqlite"))


def test_cli_survey_runs_before_a_config_exists(tmp_path, monkeypatch):
    # The whole point of survey is to help produce a config, so it must not
    # require one. Without this the documented first command fails.
    from gmail_sort import cli

    monkeypatch.setattr(cli, "_open_client", lambda config, db: FakeGmailClient({}))
    out = tmp_path / "proposed.toml"
    assert cli.main(["survey", "--config", str(tmp_path / "no-such-config.toml"),
                     "--db", str(tmp_path / "d.sqlite"),
                     "--out", str(out)]) == 0
    text = out.read_text()
    assert "# [taxonomy]" in text
    # and the proposal still carries the default guard query
    assert "has:nouserlabels -is:starred -is:important -larger:5M" in text
