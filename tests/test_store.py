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
