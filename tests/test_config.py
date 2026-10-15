import pytest
from pathlib import Path
from gmail_sort.config import ConfigError, classifier_version, load_config, parse_size

MINIMAL = """
[candidates]
query = "has:nouserlabels -is:starred -is:important -larger:5M"

[llm]
provider = "anthropic"
model = "claude-haiku-4-5"

[taxonomy]
labels = ["Newsletters", "Receipts"]

[[rules]]
name = "newsletters"
priority = 10
label = "Newsletters"
list_id = "news.example.com"
"""

def write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.toml"
    p.write_text(body)
    return p

def test_load_minimal_config(tmp_path):
    cfg = load_config(write(tmp_path, MINIMAL))
    assert cfg.min_size_bytes == 5 * 1024 * 1024
    assert cfg.excerpt_chars == 2000
    assert cfg.store_bodies is True
    assert cfg.labels == ("Newsletters", "Receipts")
    assert cfg.rules[0].name == "newsletters"

def test_rule_naming_unknown_label_is_rejected(tmp_path):
    bad = MINIMAL.replace('label = "Newsletters"', 'label = "Nope"')
    with pytest.raises(ConfigError, match="Nope"):
        load_config(write(tmp_path, bad))

def test_duplicate_rule_names_are_rejected(tmp_path):
    dup = MINIMAL + """
[[rules]]
name = "newsletters"
priority = 20
label = "Receipts"
list_id = "other.example.com"
"""
    with pytest.raises(ConfigError, match="duplicate rule name"):
        load_config(write(tmp_path, dup))

def test_invalid_regex_is_rejected(tmp_path):
    bad = MINIMAL.replace('list_id = "news.example.com"', 'subject_re = "([unclosed"')
    with pytest.raises(ConfigError, match="subject_re"):
        load_config(write(tmp_path, bad))

def test_rule_with_no_match_fields_is_rejected(tmp_path):
    bad = MINIMAL.replace('list_id = "news.example.com"', "")
    with pytest.raises(ConfigError, match="at least one match field"):
        load_config(write(tmp_path, bad))

def test_unknown_llm_provider_is_rejected(tmp_path):
    bad = MINIMAL.replace('provider = "anthropic"', 'provider = "gemini"')
    with pytest.raises(ConfigError, match="provider"):
        load_config(write(tmp_path, bad))

def test_parse_size_suffixes():
    assert parse_size("5M") == 5 * 1024**2
    assert parse_size("1000000") == 1000000
    assert parse_size("1G") == 1024**3
    assert parse_size("512K") == 512 * 1024

def test_classifier_version_is_stable_and_sensitive():
    args = (["A"], (), "prompt", "model")
    assert classifier_version(*args) == classifier_version(*args)
    for i, changed in enumerate([
        (["B"], (), "prompt", "model"),
        (["A"], (), "other prompt", "model"),
        (["A"], (), "prompt", "other-model"),
    ]):
        assert classifier_version(*changed) != classifier_version(*args), i

def test_taxonomy_label_named_like_a_gmail_system_label_is_rejected(tmp_path):
    # A label called TRASH resolves to Gmail's own Trash label id, so adding it
    # would move mail to Trash. SPAM, INBOX and UNREAD are equally destructive
    # or state-changing, and STARRED/IMPORTANT would invert the guards.
    for reserved in ("TRASH", "SPAM", "INBOX", "UNREAD", "Important", "Starred"):
        bad = MINIMAL.replace('labels = ["Newsletters", "Receipts"]',
                              f'labels = ["Newsletters", "Receipts", "{reserved}"]')
        with pytest.raises(ConfigError, match="reserved"):
            load_config(write(tmp_path, bad))
