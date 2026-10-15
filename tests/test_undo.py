from gmail_sort.cli import apply_plan, plan_apply, plan_undo, undo_run
from gmail_sort.store import Store
from tests.fake_gmail import FakeGmailClient
from tests.helpers import cfg, row, seed_decision


def applied(tmp_path):
    """One completed apply run over two messages, ready to be undone."""
    store = Store(tmp_path / "d.sqlite")
    seed_decision(store, "m1", "Receipts")
    seed_decision(store, "m2", "Receipts")
    fake = FakeGmailClient({"m1": row("m1"), "m2": row("m2")})
    rid = store.start_run("apply", dry_run=False, config_hash="h")
    apply_plan(fake, store, cfg(), plan_apply(store, cfg(), version="v"),
               dry_run=False, run_id=rid)
    return store, fake, rid


def test_round_trip_restores_the_original_labels(tmp_path):
    store, fake, rid = applied(tmp_path)
    before = fake.modified.copy()
    undo_run(fake, store, rid, dry_run=False)
    assert fake.modified[len(before):] == [("m1", (), ("Label_1",)),
                                           ("m2", (), ("Label_1",))]


def test_plan_undo_reads_the_run_log(tmp_path):
    store, _, rid = applied(tmp_path)
    assert plan_undo(store, rid) == {"Receipts": ["m1", "m2"]}


def test_undo_is_scoped_to_one_run(tmp_path):
    store, fake, rid = applied(tmp_path)
    seed_decision(store, "m9", "Newsletters")
    rid2 = store.start_run("apply", dry_run=False, config_hash="h")
    apply_plan(fake, store, cfg(), plan_apply(store, cfg(), version="v"),
               dry_run=False, run_id=rid2)
    assert plan_undo(store, rid) == {"Receipts": ["m1", "m2"]}
    assert plan_undo(store, rid2) == {"Newsletters": ["m9"]}


def test_undo_skips_a_message_that_no_longer_exists(tmp_path):
    store, fake, rid = applied(tmp_path)
    fake.fail_on = {"m2"}
    stats = undo_run(fake, store, rid, dry_run=False)
    assert stats.undone == 1
    assert stats.skipped_missing == 1


def test_undo_dry_run_writes_nothing(tmp_path):
    store, fake, rid = applied(tmp_path)
    before = list(fake.modified)
    undo_run(fake, store, rid, dry_run=True)
    assert fake.modified == before


def test_undo_of_a_run_with_no_changes_is_a_no_op(tmp_path):
    with Store(tmp_path / "d.sqlite") as store:
        rid = store.start_run("classify", dry_run=False, config_hash="h")
        fake = FakeGmailClient({})
        assert undo_run(fake, store, rid, dry_run=False).undone == 0
    assert fake.modified == []


def test_undo_logs_remove_actions(tmp_path):
    store, fake, rid = applied(tmp_path)
    undo_run(fake, store, rid, dry_run=False)
    assert ("m1", "Receipts", "remove") in list(store.iter_changes(rid))


def test_plan_apply_re_enables_the_pair_after_undo(tmp_path):
    store, fake, rid = applied(tmp_path)
    assert plan_apply(store, cfg(), version="v") == {}
    undo_run(fake, store, rid, dry_run=False)
    assert plan_apply(store, cfg(), version="v") == {"Receipts": ["m1", "m2"]}


def test_undo_twice_does_not_remove_the_label_from_others(tmp_path):
    # Second undo re-logs removes and re-issues them: batch_modify with an id
    # the message no longer carries is a no-op in Gmail, so this is safe but
    # must not silently do something else.
    store, fake, rid = applied(tmp_path)
    undo_run(fake, store, rid, dry_run=False)
    before = len(fake.modified)
    undo_run(fake, store, rid, dry_run=False)
    assert fake.modified[before:] == [("m1", (), ("Label_1",)),
                                      ("m2", (), ("Label_1",))]


def test_cli_undo_writes_only_with_yes(tmp_path, monkeypatch):
    from gmail_sort import cli

    store, fake, rid = applied(tmp_path)
    store.close()
    monkeypatch.setattr(cli, "_open_client", lambda config, db: fake)
    args = ["undo", str(rid), "--db", str(tmp_path / "d.sqlite")]
    before = list(fake.modified)

    assert cli.main(args) == 0
    assert fake.modified == before
    assert cli.main(args + ["--yes"]) == 0
    assert fake.modified[len(before):] == [("m1", (), ("Label_1",)),
                                           ("m2", (), ("Label_1",))]


def test_cli_undo_of_an_unknown_run_exits_2(tmp_path):
    from gmail_sort import cli

    assert cli.main(["undo", "999", "--db", str(tmp_path / "d.sqlite")]) == 2
