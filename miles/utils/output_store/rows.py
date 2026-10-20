"""Rows of one field of a sample's store plan: segments of SGLang's stored objects, and how events edit them.

What a row of an object means is SGLang's definition:

- forward fields (``SeqRows``: R3, indexer top-k): row ``i`` is the forward output at the object's
  ``i``-th position; an object may start at ``routed_experts_start_len`` (additional R3);
- sampler fields (``RespRows``: the sampling mask): row ``i`` is the request's ``i``-th output token.

A sample's fields count rows like the inline path: forward fields ``len(tokens) - 1``, sampler
fields ``response_length``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

from miles.utils.output_store.materialize import gather_csr_rows, gather_dense_rows
from miles.utils.retry_utils import NonRetryableError

if TYPE_CHECKING:
    from radix_transfer_plugins import StoreRef


class StorePlanError(NonRetryableError, ValueError):
    """A store plan edit or a response's ref breaks the plan's contract.

    Retrying cannot fix it. A ``ValueError``, so the session server answers it with 422.
    """


class Seg(NamedTuple):
    """Rows ``[start, stop)`` of one object; ``ref`` None means forced rows."""

    ref: StoreRef | None
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class Rows:
    """One field's segments: refs and row numbers only."""

    segs: tuple[Seg, ...] = ()

    def num_rows(self) -> int:
        return sum(seg.stop - seg.start for seg in self.segs)

    def prefix(self, n: int) -> Rows:
        if not 0 <= n <= self.num_rows():
            raise StorePlanError(f"cannot keep {n} rows of {self.num_rows()}")
        kept, left = [], n
        for seg in self.segs:
            if left == 0:
                break
            take = min(left, seg.stop - seg.start)
            kept.append(seg._replace(stop=seg.start + take))
            left -= take
        return type(self)(tuple(kept))

    def concat(self, other: Rows) -> Rows:
        if type(other) is not type(self):
            raise StorePlanError(f"cannot concatenate {type(other).__name__} to {type(self).__name__}")
        return type(self)(self.segs + other.segs)

    def append(self, new: Rows, *, response_length: int, seq_start: int) -> Rows:
        raise NotImplementedError

    def insert(self, n: int, *, response_length: int) -> Rows:
        raise NotImplementedError

    @staticmethod
    def expected_rows(*, num_tokens: int, response_length: int) -> int:
        raise NotImplementedError


class SeqRows(Rows):
    """Forward outputs (R3, indexer top-k): a row per sequence position but the last."""

    __slots__ = ()
    gather = staticmethod(gather_dense_rows)

    def append(self, new: Rows, *, response_length: int, seq_start: int) -> Rows:
        # A forward's object covers the sequence from seq_start on; seq_start 0 replaces every row.
        out = self.prefix(seq_start).concat(new)
        if out.num_rows() < self.num_rows():
            raise StorePlanError(f"a forward covers {out.num_rows()} rows, fewer than the {self.num_rows()} before it")
        return out

    def insert(self, n: int, *, response_length: int) -> Rows:
        # The next forward covers the inserted tokens.
        return self

    @staticmethod
    def expected_rows(*, num_tokens: int, response_length: int) -> int:
        return num_tokens - 1


class RespRows(Rows):
    """Sampler outputs (the sampling mask): a row per response token."""

    __slots__ = ()
    gather = staticmethod(gather_csr_rows)

    def append(self, new: Rows, *, response_length: int, seq_start: int) -> Rows:
        if seq_start:
            raise StorePlanError("sampler rows have no sequence start")
        self._check_aligned(response_length)
        return self.concat(new)

    def insert(self, n: int, *, response_length: int) -> Rows:
        self._check_aligned(response_length)
        return self.concat(RespRows((Seg(None, 0, n),))) if n else self

    @staticmethod
    def expected_rows(*, num_tokens: int, response_length: int) -> int:
        return response_length

    def _check_aligned(self, response_length: int) -> None:
        if self.num_rows() != response_length:
            raise StorePlanError(
                f"sampling mask rows {self.num_rows()} are not aligned with response_length {response_length}"
            )
