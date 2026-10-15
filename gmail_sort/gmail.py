"""The Gmail seam: one small protocol, two implementations.

The real client (`RestGmailClient`, added in a later task) and the test fake are
indistinguishable to `classify`, `apply` and `undo`, including the internal
1000-id chunking of `batch_modify`. Only five REST calls are ever made, so this
protocol stays small enough to reason about and to fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence

from gmail_sort.store import MessageRow


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
