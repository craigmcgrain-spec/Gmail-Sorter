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
