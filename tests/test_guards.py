from types import SimpleNamespace
from gmail_sort.config import Config
from gmail_sort.guards import SYSTEM_LABEL_IDS, guard_query, user_label_ids, violation

LIMIT = 5 * 1024 * 1024

def msg(**kw):
    base = dict(label_ids=("INBOX", "Label_1"), size_estimate=1000)
    base.update(kw)
    return SimpleNamespace(**base)

def test_clean_message_passes():
    assert violation(msg(label_ids=("INBOX",)), user_label_ids=frozenset({"Label_1"}),
                     min_size_bytes=LIMIT) is None

def test_starred_is_rejected():
    assert violation(msg(label_ids=("INBOX", "STARRED")), user_label_ids=frozenset(),
                     min_size_bytes=LIMIT) == "starred"

def test_important_is_rejected():
    assert violation(msg(label_ids=("INBOX", "IMPORTANT")), user_label_ids=frozenset(),
                     min_size_bytes=LIMIT) == "important"

def test_user_label_is_rejected():
    assert violation(msg(), user_label_ids=frozenset({"Label_1"}),
                     min_size_bytes=LIMIT) == "labelled"

def test_category_label_is_not_a_user_label():
    # CATEGORY_* ids are system labels; they must not read as "already labelled".
    assert violation(msg(label_ids=("INBOX", "CATEGORY_PROMOTIONS")),
                     user_label_ids=frozenset({"Label_1", "Label_2"}),
                     min_size_bytes=LIMIT) is None

def test_oversize_is_rejected_at_the_boundary():
    assert violation(msg(size_estimate=LIMIT), user_label_ids=frozenset(),
                     min_size_bytes=LIMIT) == "oversize"
    assert violation(msg(size_estimate=LIMIT - 1), user_label_ids=frozenset(),
                     min_size_bytes=LIMIT) is None

def test_user_label_ids_filters_by_type():
    labels = [{"id": "Label_1", "type": "user"}, {"id": "INBOX", "type": "system"}]
    assert user_label_ids(labels) == frozenset({"Label_1"})

def test_system_label_ids_covers_gmail_system_labels():
    for lid in ("INBOX", "UNREAD", "STARRED", "IMPORTANT", "SENT", "DRAFT",
                "SPAM", "TRASH", "CATEGORY_PERSONAL", "CATEGORY_SOCIAL",
                "CATEGORY_PROMOTIONS", "CATEGORY_UPDATES", "CATEGORY_FORUMS"):
        assert lid in SYSTEM_LABEL_IDS

def test_guard_query_is_verbatim_from_config():
    cfg = Config(query="has:nouserlabels -is:starred", min_size_bytes=LIMIT,
                 size_suffix="5M", excerpt_chars=2000, store_bodies=True,
                 llm_provider="anthropic", llm_model="m", max_messages_per_run=0,
                 labels=("L",), rules=())
    assert guard_query(cfg) == "has:nouserlabels -is:starred"
