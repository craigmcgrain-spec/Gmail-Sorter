import base64

import httpx
import pytest

from gmail_sort.gmail import RestGmailClient, TokenBucket, parse_message


def client(handler, **kw):
    return RestGmailClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        "tok",
        bucket=TokenBucket(1e9),
        **kw,
    )


def test_list_walks_every_page_token():
    def handler(request):
        params = dict(httpx.URL(str(request.url)).params)
        pages = {
            "": {"messages": [{"id": "m1"}], "nextPageToken": "p2"},
            "p2": {"messages": [{"id": "m2"}], "nextPageToken": "p3"},
            "p3": {"messages": [{"id": "m3"}]},
        }
        return httpx.Response(200, json=pages[params.get("pageToken", "")])

    assert list(client(handler).iter_message_ids("q")) == ["m1", "m2", "m3"]


def test_list_requests_500_per_page_with_the_query():
    seen = {}

    def handler(request):
        seen.update(dict(httpx.URL(str(request.url)).params))
        return httpx.Response(200, json={"messages": []})

    list(client(handler).iter_message_ids("has:nouserlabels"))
    assert seen["maxResults"] == "500"
    assert seen["q"] == "has:nouserlabels"


def test_list_tolerates_an_empty_result():
    empty = lambda request: httpx.Response(200, json={})
    assert list(client(empty).iter_message_ids("q")) == []


def test_list_omits_page_token_on_the_first_request():
    seen = {}

    def handler(request):
        seen.update(dict(httpx.URL(str(request.url)).params))
        return httpx.Response(200, json={"messages": [{"id": "m1"}]})

    list(client(handler).iter_message_ids("q"))
    assert "pageToken" not in seen


def test_get_message_asks_for_metadata_by_default():
    seen = {}

    def handler(request):
        seen["url"] = httpx.URL(str(request.url))
        return httpx.Response(200, json={"id": "m1", "threadId": "t1",
                                         "labelIds": [], "sizeEstimate": 1,
                                         "payload": {"headers": []}})

    client(handler).get_message("m1")
    params = seen["url"].params
    assert params["format"] == "metadata"
    assert params.get_list("metadataHeaders") == ["From", "Subject", "List-Id"]


def test_get_message_asks_for_the_body_when_full():
    seen = {}

    def handler(request):
        seen.update(dict(httpx.URL(str(request.url)).params))
        return httpx.Response(200, json={"id": "m1", "threadId": "t1",
                                         "labelIds": [], "sizeEstimate": 1,
                                         "payload": {"headers": []}})

    client(handler).get_message("m1", full=True)
    assert seen["format"] == "full"


def test_retries_on_429_with_backoff_then_succeeds():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": {"code": 429}})
        return httpx.Response(200, json={"labels": []})

    gmail = client(handler)
    slept = []
    gmail._sleep = slept.append
    assert gmail.list_labels() == []
    assert calls["n"] == 3
    assert slept == [1.0, 2.0]


def test_gives_up_after_five_attempts():
    gmail = client(lambda request: httpx.Response(500))
    gmail._sleep = lambda seconds: None
    with pytest.raises(Exception):
        gmail.list_labels()


def test_401_refreshes_the_token_once_and_retries():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(401, json={"error": {"code": 401}})
        assert request.headers["authorization"] == "Bearer fresh"
        return httpx.Response(200, json={"labels": []})

    gmail = RestGmailClient(
        httpx.Client(transport=httpx.MockTransport(handler)),
        "stale",
        token_refresher=lambda: "fresh",
        bucket=TokenBucket(1e9),
    )
    assert gmail.list_labels() == []
    assert calls["n"] == 2


def test_batch_modify_sends_only_the_keys_it_needs():
    seen = {}

    def handler(request):
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(204)

    client(handler).batch_modify(["m1", "m2"], add=["Label_1"])
    assert seen == {"ids": ["m1", "m2"], "addLabelIds": ["Label_1"]}


def test_batch_modify_of_an_empty_list_makes_no_request():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(204)

    client(handler).batch_modify([])
    assert calls["n"] == 0


def test_parse_message_derives_domain_and_headers():
    payload = {"id": "m1", "threadId": "t1", "labelIds": ["INBOX", "Label_1"],
               "sizeEstimate": 1234,
               "payload": {"headers": [
                   {"name": "From", "value": "Alice <Alice@Example.COM>"},
                   {"name": "Subject", "value": "Hi there"},
                   {"name": "List-Id", "value": "news.example.com"}]}}
    message = parse_message(payload, full=False, excerpt_chars=2000)
    assert message.from_addr == "Alice@Example.COM"
    assert message.from_domain == "example.com"
    assert message.subject == "Hi there"
    assert message.list_id == "news.example.com"
    assert message.size_estimate == 1234
    assert message.label_ids == ("INBOX", "Label_1")
    assert message.excerpt is None


def test_parse_message_extracts_and_truncates_a_text_body():
    body = base64.urlsafe_b64encode(b"x" * 5000).decode()
    payload = {"id": "m1", "threadId": "t1", "labelIds": [], "sizeEstimate": 1,
               "payload": {"headers": [], "mimeType": "text/plain",
                           "body": {"data": body}}}
    assert len(parse_message(payload, full=True, excerpt_chars=100).excerpt) == 100


def test_parse_message_handles_a_missing_from_header():
    message = parse_message(
        {"id": "m1", "threadId": "t1", "labelIds": [], "sizeEstimate": 1,
         "payload": {"headers": []}},
        full=False, excerpt_chars=2000,
    )
    assert message.from_addr == ""
    assert message.from_domain == ""


def test_parse_message_prefers_text_plain_in_multipart():
    encode = lambda text: base64.urlsafe_b64encode(text).decode()
    payload = {"id": "m1", "threadId": "t1", "labelIds": [], "sizeEstimate": 1,
               "payload": {"headers": [], "mimeType": "multipart/alternative", "parts": [
                   {"mimeType": "text/html", "body": {"data": encode(b"<p>html</p>")}},
                   {"mimeType": "text/plain", "body": {"data": encode(b"plain text")}}]}}
    assert parse_message(payload, full=True, excerpt_chars=2000).excerpt == "plain text"


def test_parse_message_survives_undecodable_body():
    payload = {"id": "m1", "threadId": "t1", "labelIds": [], "sizeEstimate": 1,
               "payload": {"headers": [], "mimeType": "text/plain",
                           "body": {"data": "!!!not base64!!!"}}}
    assert parse_message(payload, full=True, excerpt_chars=2000).excerpt is None


def test_parse_message_takes_the_first_category_label():
    payload = {"id": "m1", "threadId": "t1",
               "labelIds": ["INBOX", "CATEGORY_PROMOTIONS", "CATEGORY_UPDATES"],
               "sizeEstimate": 1, "payload": {"headers": []}}
    assert parse_message(payload, full=False,
                         excerpt_chars=2000).category == "CATEGORY_PROMOTIONS"
