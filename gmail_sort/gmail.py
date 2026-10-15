"""The Gmail seam: one small protocol, two implementations.

The real client and the test fake are indistinguishable to `classify`, `apply`
and `undo`, including the internal 1000-id chunking of `batch_modify`.

Only five REST calls are ever made. `httpx` is used directly rather than the
discovery-based Google client: five small functions are easier to reason about
and far easier to fake in tests.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, Sequence

import httpx

from gmail_sort.store import MessageRow, now

API_ROOT = "https://gmail.googleapis.com/gmail/v1/users/me"

# Quota units per call, from Gmail's per-method table.
QUOTA = {"list": 5, "get": 20, "batch_modify": 50, "labels_create": 5, "labels_list": 1}

# The per-user ceiling is 6000 units/minute, so a bucket refilling at 100
# units/second paces us exactly at it and throttles before Gmail does.
QUOTA_UNITS_PER_SECOND = 100.0

MAX_ATTEMPTS = 5
MAX_BACKOFF_SECONDS = 64.0

BATCH_SIZE = 1000


@dataclass(frozen=True)
class Label:
    id: str
    name: str
    type: str


def chunked(items: Sequence, size: int) -> Iterator[list]:
    """Split into consecutive lists of at most `size` items."""
    materialised = list(items)
    for start in range(0, len(materialised), size):
        yield materialised[start : start + size]


class GmailClient(Protocol):
    page_size: int
    calls: list[tuple]

    def iter_message_ids(self, query: str) -> Iterator[str]:
        """Every message id matching `query`, following pageToken to the end."""
        ...

    def get_message(self, message_id: str, *, full: bool = False) -> MessageRow:
        """Metadata only by default; `full` also fetches a body excerpt."""
        ...

    def list_labels(self) -> list[Label]:
        ...

    def create_label(self, name: str) -> Label:
        ...

    def batch_modify(
        self,
        message_ids: Sequence[str],
        *,
        add: Sequence[str] = (),
        remove: Sequence[str] = (),
    ) -> None:
        """Add/remove labels in chunks of 1000. Never used to delete a message."""
        ...


class TokenBucket:
    """Rate limiter whose clock and sleep are injected, so tests never sleep."""

    def __init__(
        self,
        rate_per_sec: float = QUOTA_UNITS_PER_SECOND,
        capacity: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.rate = float(rate_per_sec)
        self.capacity = float(rate_per_sec if capacity is None else capacity)
        self._clock = clock
        self._sleep = sleep
        self._tokens = self.capacity
        self._last = clock()

    def take(self, units: int) -> None:
        """Block until `units` of budget are available, then spend them."""
        units = float(units)
        if units > self.capacity:
            raise ValueError(
                f"cannot take {units} units; capacity is {self.capacity}"
            )
        self._refill()
        if self._tokens < units:
            deficit = units - self._tokens
            self._sleep(deficit / self.rate)
            self._last = self._clock()
            self._tokens += deficit
        self._tokens -= units

    def _refill(self) -> None:
        current = self._clock()
        elapsed = current - self._last
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._last = current


class RestGmailClient:
    page_size = 500

    def __init__(
        self,
        http: httpx.Client,
        access_token: str,
        *,
        token_refresher: Callable[[], str] | None = None,
        bucket: TokenBucket | None = None,
        excerpt_chars: int = 2000,
    ):
        self._http = http
        self._token = access_token
        self._token_refresher = token_refresher
        self._bucket = bucket or TokenBucket()
        self._excerpt_chars = excerpt_chars
        self._sleep = time.sleep
        self.calls: list[tuple] = []

    # GmailClient ------------------------------------------------------------

    def iter_message_ids(self, query: str) -> Iterator[str]:
        token: str | None = None
        while True:
            params = {"q": query, "maxResults": str(self.page_size)}
            if token:
                params["pageToken"] = token
            self.calls.append(("iter_message_ids", (query,), {"pageToken": token}))
            payload = self._request(
                "GET", "/messages", params=params, quota=QUOTA["list"]
            ).json()
            for entry in payload.get("messages") or ():
                yield entry["id"]
            token = payload.get("nextPageToken")
            if not token:
                return

    def get_message(self, message_id: str, *, full: bool = False) -> MessageRow:
        params: dict = {"format": "full" if full else "metadata"}
        if not full:
            params["metadataHeaders"] = ["From", "Subject", "List-Id"]
        payload = self._request(
            "GET", f"/messages/{message_id}", params=params, quota=QUOTA["get"]
        ).json()
        return parse_message(
            payload, full=full, excerpt_chars=self._excerpt_chars
        )

    def list_labels(self) -> list[Label]:
        payload = self._request("GET", "/labels", quota=QUOTA["labels_list"]).json()
        return [
            Label(
                id=entry["id"],
                name=entry.get("name", entry["id"]),
                type=entry.get("type", "user"),
            )
            for entry in payload.get("labels") or ()
        ]

    def create_label(self, name: str) -> Label:
        payload = self._request(
            "POST",
            "/labels",
            json={
                "name": name,
                "labelListVisibility": "labelShow",
                "messageListVisibility": "show",
            },
            quota=QUOTA["labels_create"],
        ).json()
        return Label(
            id=payload["id"],
            name=payload.get("name", name),
            type=payload.get("type", "user"),
        )

    def batch_modify(
        self,
        message_ids: Sequence[str],
        *,
        add: Sequence[str] = (),
        remove: Sequence[str] = (),
    ) -> None:
        ids = list(message_ids)
        if not ids:
            return
        for chunk in chunked(ids, BATCH_SIZE):
            body: dict = {"ids": list(chunk)}
            if add:
                body["addLabelIds"] = list(add)
            if remove:
                body["removeLabelIds"] = list(remove)
            self.calls.append(("batch_modify", (), body))
            self._request(
                "POST", "/messages/batchModify", json=body, quota=QUOTA["batch_modify"]
            )

    # internals --------------------------------------------------------------

    def _request(self, method: str, path: str, *, params=None, json=None, quota: int):
        attempt = 0
        refreshed = False
        while True:
            attempt += 1
            self._bucket.take(quota)
            response = self._http.request(
                method,
                API_ROOT + path,
                params=params,
                json=json,
                headers={"Authorization": f"Bearer {self._token}"},
            )
            if response.status_code == 401 and self._token_refresher and not refreshed:
                refreshed = True
                self._token = self._token_refresher()
                attempt -= 1
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= MAX_ATTEMPTS:
                    response.raise_for_status()
                self._sleep(min(MAX_BACKOFF_SECONDS, 2.0 ** (attempt - 1)))
                continue
            response.raise_for_status()
            return response


def parse_message(payload: dict, *, full: bool, excerpt_chars: int) -> MessageRow:
    """Turn one Gmail message resource into a `MessageRow`."""
    headers = {
        header["name"].lower(): header.get("value", "")
        for header in (payload.get("payload") or {}).get("headers") or ()
    }
    from_addr = _address_of(headers.get("from", ""))
    label_ids = tuple(payload.get("labelIds") or ())
    return MessageRow(
        id=payload["id"],
        thread_id=payload.get("threadId", ""),
        from_addr=from_addr,
        from_domain=_domain_of(from_addr),
        subject=headers.get("subject", ""),
        list_id=headers.get("list-id") or None,
        category=next(
            (label for label in label_ids if label.startswith("CATEGORY_")), None
        ),
        size_estimate=int(payload.get("sizeEstimate") or 0),
        label_ids=label_ids,
        excerpt=_excerpt(payload.get("payload") or {}, excerpt_chars) if full else None,
        fetched_at=now(),
    )


def _address_of(header: str) -> str:
    """`Alice <Alice@Example.COM>` becomes `Alice@Example.COM`.

    Stripping the display name keeps `from_re` rules matching the address, which
    is the thing a rule actually wants to match on.
    """
    value = (header or "").strip()
    if "<" in value and ">" in value:
        return value[value.rindex("<") + 1 : value.rindex(">")].strip()
    return value


def _domain_of(address: str) -> str:
    _, _, tail = address.rpartition("@")
    if not tail:
        return ""
    return tail.split(">")[0].strip().lower()


def _excerpt(payload: dict, limit: int) -> str | None:
    data = _body_data(payload)
    if not data:
        return None
    try:
        raw = base64.b64decode(
            data + "=" * (-len(data) % 4), altchars=b"-_", validate=True
        )
    except Exception:
        # A message we cannot read is not a reason to fail the run.
        return None
    text = raw.decode("utf-8", errors="replace")
    return text[:limit] if limit else text


def _body_data(payload: dict) -> str | None:
    parts = payload.get("parts") or []
    if parts:
        for mime in ("text/plain", "text/html"):
            for part in parts:
                if part.get("mimeType") == mime:
                    found = _body_data(part)
                    if found:
                        return found
        for part in parts:
            found = _body_data(part)
            if found:
                return found
        return None
    if payload.get("mimeType", "") in ("text/plain", "text/html", ""):
        return (payload.get("body") or {}).get("data") or None
    return None
