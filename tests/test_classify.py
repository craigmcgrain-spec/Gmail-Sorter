import pytest

from gmail_sort.cli import classify, classifier_version_for
from gmail_sort.store import Store
from tests.fake_gmail import FakeGmailClient
from tests.helpers import cfg, cfg_with_extra_rule, row, write_cfg


def version():
    return classifier_version_for(cfg())


def run(tmp_path, msgs, **kw):
    # Not a `with` block: the caller asserts on decisions afterwards, so the
    # store must still be open when this returns.
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


def test_estimate_persists_nothing(tmp_path):
    # --estimate must not write decisions, or a later real run would treat the
    # estimated messages as already decided and never send them to the LLM.
    msgs = {"m1": row("m1", from_domain="github.com"),
            "m2": row("m2", from_domain="unknown.com")}
    stats, s = run(tmp_path, msgs, dry_run=True)
    assert stats.rule_matched == 1
    assert stats.estimated_tokens > 0
    assert s.get_decision("m1", version()) is None
    assert s.get_message("m1") is None


def test_cli_estimate_needs_no_api_key(tmp_path, monkeypatch):
    from gmail_sort import cli

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "_open_client", lambda config, db: FakeGmailClient(
        {"m1": row("m1", from_domain="github.com")}))
    assert cli.main(["classify", "--config", str(write_cfg(tmp_path)),
                     "--db", str(tmp_path / "d.sqlite"), "--estimate"]) == 0
