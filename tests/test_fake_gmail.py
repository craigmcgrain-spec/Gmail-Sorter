import pytest
from tests.fake_gmail import FakeGmailClient
from tests.helpers import row

def test_iter_message_ids_pages_past_the_page_size():
    msgs = {f"m{i}": row(f"m{i}") for i in range(1200)}
    fake = FakeGmailClient(msgs, page_size=500)
    assert list(fake.iter_message_ids("in:anywhere")) == list(msgs)
    assert 2 < len([c for c in fake.calls if c[0] == "iter_message_ids"])

def test_batch_modify_chunks_at_1000():
    fake = FakeGmailClient({})
    ids = [f"m{i}" for i in range(2500)]
    fake.batch_modify(ids, add=["Label_1"])
    calls = [c for c in fake.calls if c[0] == "batch_modify"]
    assert [len(c[2]["ids"]) for c in calls] == [1000, 1000, 500]

def test_batch_modify_applies_add_and_remove_to_state():
    fake = FakeGmailClient({})
    fake.batch_modify(["m1"], add=["Label_1"])
    fake.batch_modify(["m1"], remove=["Label_1"])
    assert fake.modified == [("m1", ("Label_1",), ()), ("m1", (), ("Label_1",))]

def test_batch_modify_raises_when_configured_to_fail():
    fake = FakeGmailClient({}, fail_on={"m2"})
    with pytest.raises(Exception, match="m2"):
        fake.batch_modify(["m1", "m2"], add=["Label_1"])

def test_create_label_nests_and_is_idempotent():
    fake = FakeGmailClient({})
    a = fake.create_label("Notifications/Traffic")
    assert a.name == "Notifications/Traffic"
    assert fake.create_label("Notifications/Traffic") == a
