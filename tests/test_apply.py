import pytest

from gmail_sort.cli import apply_plan, classifier_version_for, plan_apply
from gmail_sort.store import Decision, Store
from tests.fake_gmail import FakeGmailClient
from tests.helpers import cfg, row, seed_decision, write_cfg


def test_dry_run_writes_nothing(tmp_path):
    with Store(tmp_path / "d.sqlite") as s:
        seed_decision(s, "m1", "Receipts")
        fake = FakeGmailClient({"m1": row("m1")})
        rid = s.start_run("apply", dry_run=True, config_hash="h")
        stats = apply_plan(fake, s, cfg(), plan_apply(s, cfg(), version="v"),
                           dry_run=True, run_id=rid)
    assert fake.modified == []
    assert stats.applied == 0


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
    assert stats.applied == 2
    assert stats.skipped_missing == 1


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
    assert fake.modified == []
    assert stats.skipped_guard == 1


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


def test_missing_label_is_created_once_and_reused(tmp_path):
    with Store(tmp_path / "d.sqlite") as s:
        seed_decision(s, "m1", "Receipts")
        fake = FakeGmailClient({"m1": row("m1")})
        rid = s.start_run("apply", dry_run=False, config_hash="h")
        apply_plan(fake, s, cfg(), plan_apply(s, cfg(), version="v"),
                   dry_run=False, run_id=rid)
    assert len([c for c in fake.calls if c[0] == "create_label"]) == 1


def test_cli_apply_without_yes_is_a_dry_run(tmp_path, monkeypatch):
    from gmail_sort import cli

    fake = FakeGmailClient({"m1": row("m1")})
    monkeypatch.setattr(cli, "_open_client", lambda config, db: fake)
    with Store(tmp_path / "d.sqlite") as s:
        seed_decision(s, "m1", "Receipts", version=classifier_version_for(cfg()))
    assert cli.main(["apply", "--config", str(write_cfg(tmp_path)),
                     "--db", str(tmp_path / "d.sqlite")]) == 0
    assert fake.modified == []


def test_cli_apply_with_yes_writes(tmp_path, monkeypatch):
    from gmail_sort import cli

    fake = FakeGmailClient({"m1": row("m1")})
    monkeypatch.setattr(cli, "_open_client", lambda config, db: fake)
    with Store(tmp_path / "d.sqlite") as s:
        seed_decision(s, "m1", "Receipts", version=classifier_version_for(cfg()))
    assert cli.main(["apply", "--config", str(write_cfg(tmp_path)),
                     "--db", str(tmp_path / "d.sqlite"), "--yes"]) == 0
    assert fake.modified == [("m1", ("Label_1",), ())]
