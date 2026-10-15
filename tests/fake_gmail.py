"""In-memory GmailClient so classify/apply/undo run end-to-end without network.

Both this and the real client chunk `batch_modify` at 1000 internally, so the
caller cannot tell them apart.
"""

from __future__ import annotations

import re
from typing import Iterator, Sequence

from gmail_sort.gmail import Label, chunked
from gmail_sort.store import MessageRow


class FakeGmailClient:
    def __init__(
        self,
        messages: dict[str, MessageRow],
        *,
        page_size: int = 500,
        label_ids: Sequence[Label | dict] | None = None,
        fail_on: set[str] | None = None,
    ):
        self.messages = dict(messages)
        self.page_size = page_size
        self.label_ids = [_as_label(label) for label in (label_ids or ())]
        self.fail_on = set(fail_on or ())
        self.calls: list[tuple] = []
        self.modified: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    # GmailClient ------------------------------------------------------------

    def iter_message_ids(self, query: str) -> Iterator[str]:
        # One recorded call per page, mirroring the real client's pageToken loop.
        # An empty mailbox still costs one call.
        ids = list(self.messages)
        pages = [
            ids[start : start + self.page_size]
            for start in range(0, len(ids), self.page_size)
        ] or [[]]
        for number, page in enumerate(pages):
            self.calls.append(
                ("iter_message_ids", (query,), {"pageToken": number})
            )
            for message_id in page:
                yield message_id

    def get_message(self, message_id: str, *, full: bool = False) -> MessageRow:
        self.calls.append(("get_message", (message_id,), {"full": full}))
        if message_id in self.fail_on:
            raise KeyError(f"no such message: {message_id}")
        return self.messages[message_id]

    def list_labels(self) -> list[Label]:
        self.calls.append(("list_labels", (), {}))
        return list(self.label_ids)

    def create_label(self, name: str) -> Label:
        self.calls.append(("create_label", (name,), {}))
        for label in self.label_ids:
            if label.name == name:
                return label
        label = Label(id=self._free_label_id(), name=name, type="user")
        self.label_ids.append(label)
        return label

    def batch_modify(
        self,
        message_ids: Sequence[str],
        *,
        add: Sequence[str] = (),
        remove: Sequence[str] = (),
    ) -> None:
        for chunk in chunked(message_ids, 1000):
            self.calls.append(
                ("batch_modify", (), {"ids": list(chunk), "add": list(add),
                                      "remove": list(remove)})
            )
            for message_id in chunk:
                if message_id in self.fail_on:
                    raise RuntimeError(f"message no longer exists: {message_id}")
            for message_id in chunk:
                self.modified.append((message_id, tuple(add), tuple(remove)))

    # internals --------------------------------------------------------------

    def _free_label_id(self) -> str:
        taken = {
            int(match.group(1))
            for label in self.label_ids
            if (match := re.fullmatch(r"Label_(\d+)", label.id))
        }
        number = 1
        while number in taken:
            number += 1
        return f"Label_{number}"


def _as_label(value: Label | dict) -> Label:
    if isinstance(value, Label):
        return value
    return Label(id=value["id"], name=value.get("name", value["id"]),
                 type=value.get("type", "user"))
