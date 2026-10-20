"""Rollout side of the output store path: a sample's merge plan over SGLang's stored objects.

With ``--use-output-store``, SGLang writes the R3, indexer top-k and sampling mask of a training
response to one object of the output store (``radix_transfer_plugins``) and returns its
``StoreRef`` in ``meta_info["output_store_ref"]``. The rollout never reads an object: every event
on a sample (a response, a tool observation, a session merge, a truncation) only edits the
sample's ``StorePlan``, the rows of which object each field takes and where forced rows go. The
trainer executes the plans (``miles.backends.training_utils.data.store_columns``) and writes the
training columns of the inline path, which this module does not touch.

What a row of an object means is SGLang's definition:

- forward fields (R3, indexer top-k): row ``i`` is the forward output at the object's ``i``-th
  position; an object may start at ``routed_experts_start_len`` (additional R3);
- sampler fields (the sampling mask): row ``i`` is the request's ``i``-th output token.

A sample's fields count rows like the inline path: forward fields ``len(tokens) - 1``, sampler
fields ``response_length``.
"""

from __future__ import annotations

import threading
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple

import msgspec
import numpy as np

from miles.utils.retry_utils import NonRetryableError
from miles.utils.store_exec import gather_csr_rows, gather_dense_rows

if TYPE_CHECKING:
    from radix_transfer_plugins import StoreRef

ROUTED_EXPERTS = "routed_experts"
INDEXER_TOPK = "indexer_topk"
SAMPLING_MASK = "sampling_mask"

# Stored field names, as radix_transfer_plugins' TokenOutputStash names them.
MASK_LENGTHS = "output_token_sampling_mask_lengths"
MASK_TOKEN_IDS = "output_token_sampling_mask_token_ids"
# Support log-prob mode only: a log-prob per support entry.
MASK_LOGPROBS = "output_token_sampling_logprobs"

OUTPUT_STORE_REF_KEY = "output_store_ref"
VIA_STORE_REQUEST_KEY = "return_outputs_via_store"
# Support log-prob mode: each sampled token's entry of its support row, picked by SGLang.
SELECTED_LOGPROBS_KEY = "output_token_sampling_selected_logprobs"
# The only backend of phase 1: every engine Miles launches writes to the run's own Mooncake.
SUPPORTED_STORE_BACKEND = "mooncake"
_INLINE_REPLAY_KEYS = (ROUTED_EXPERTS, INDEXER_TOPK, "output_token_sampling_mask")


class StorePlanError(NonRetryableError, ValueError):
    """A store plan edit or a response's ref breaks the plan's contract.

    Retrying cannot fix it. A ``ValueError``, so the session server answers it with 422.
    """


# ============================== requests and responses ==============================


def set_outputs_via_store(request: dict[str, Any], *, use_output_store: bool, evaluation: bool) -> None:
    """Ask SGLang to return a training request's replay outputs through the output store.

    Miles owns the field: a caller's own value is replaced, and evaluation requests never set it.
    """
    if use_output_store and not evaluation and requested_fields(request):
        request[VIA_STORE_REQUEST_KEY] = True
    else:
        request.pop(VIA_STORE_REQUEST_KEY, None)


def outputs_via_store(request: Mapping[str, Any]) -> bool:
    return bool(request.get(VIA_STORE_REQUEST_KEY))


def requested_fields(request: Mapping[str, Any]) -> frozenset[str]:
    """Names of the fields a request asks SGLang to return."""
    return frozenset(field.name for field in FIELDS if request.get(field.request_key))


def parse_store_ref(meta_info: Mapping[str, Any]) -> StoreRef | None:
    """The response's ref; a request that set ``return_outputs_via_store`` gets no inline replay data."""
    inline = [key for key in _INLINE_REPLAY_KEYS if key in meta_info]
    if inline:
        raise StorePlanError(
            f"request set {VIA_STORE_REQUEST_KEY} but the response carries inline {inline}; "
            f"a router between Miles and SGLang may have dropped {VIA_STORE_REQUEST_KEY}"
        )
    value = meta_info.get(OUTPUT_STORE_REF_KEY)
    return None if value is None else ref_from_builtins(value)


def ref_from_builtins(value: Any) -> StoreRef:
    # The package is needed only when a ref is present: runs without the output store do not install it.
    from radix_transfer_plugins import StoreRef

    ref = StoreRef.from_builtins(value)
    if ref.backend != SUPPORTED_STORE_BACKEND:
        raise StorePlanError(
            f"output store backend {ref.backend!r} is not supported; Miles reads only {SUPPORTED_STORE_BACKEND!r} refs"
        )
    return ref


def ref_to_builtins(ref: StoreRef) -> dict[str, Any]:
    """JSON types only: the train dict's msgpack codec rejects the tuples of ``to_builtins``."""
    return msgspec.json.decode(msgspec.json.encode(ref))


def store_sampled_logprobs(
    meta_info: Mapping[str, Any], *, num_output_tokens: int, support_logprobs: bool, aborted: bool
) -> list[float]:
    """The sampled tokens' log-probs of a store response, one per output token.

    They stay in ``meta_info``: ``output_token_sampling_logprobs`` in selected mode, SGLang's pick
    from each support row, ``output_token_sampling_selected_logprobs``, in support mode.
    """
    key = SELECTED_LOGPROBS_KEY if support_logprobs else "output_token_sampling_logprobs"
    log_probs = meta_info.get(key)
    if log_probs is None:
        if is_no_output_abort(meta_info, aborted=aborted, num_output_tokens=num_output_tokens):
            return []
        raise StorePlanError(f"SGLang store response is missing {key}")
    if len(log_probs) != num_output_tokens:
        raise StorePlanError(f"sampling log-prob length {len(log_probs)} != output token length {num_output_tokens}")
    if any(isinstance(value, (list, tuple)) for value in log_probs):
        raise StorePlanError(f"SGLang {key} must be scalar")
    return [float(value) for value in log_probs]


def is_no_output_abort(meta_info: Mapping[str, Any], *, aborted: bool, num_output_tokens: int) -> bool:
    finish_reason = meta_info.get("finish_reason") or {}
    return (aborted or finish_reason.get("type") == "abort") and num_output_tokens == 0


# ============================== rows of one field ==============================


class Seg(NamedTuple):
    """Rows ``[start, stop)`` of one object; ``ref`` None means forced rows."""

    ref: StoreRef | None
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class _Rows:
    """One field's segments: refs and row numbers only."""

    segs: tuple[Seg, ...] = ()

    def num_rows(self) -> int:
        return sum(seg.stop - seg.start for seg in self.segs)

    def prefix(self, n: int) -> _Rows:
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

    def concat(self, other: _Rows) -> _Rows:
        if type(other) is not type(self):
            raise StorePlanError(f"cannot concatenate {type(other).__name__} to {type(self).__name__}")
        return type(self)(self.segs + other.segs)

    def append(self, new: _Rows, *, response_length: int, seq_start: int) -> _Rows:
        raise NotImplementedError

    def insert(self, n: int, *, response_length: int) -> _Rows:
        raise NotImplementedError

    @staticmethod
    def expected_rows(*, num_tokens: int, response_length: int) -> int:
        raise NotImplementedError


class SeqRows(_Rows):
    """Forward outputs (R3, indexer top-k): a row per sequence position but the last."""

    __slots__ = ()
    gather = staticmethod(gather_dense_rows)

    def append(self, new: _Rows, *, response_length: int, seq_start: int) -> _Rows:
        # A forward's object covers the sequence from seq_start on; seq_start 0 replaces every row.
        out = self.prefix(seq_start).concat(new)
        if out.num_rows() < self.num_rows():
            raise StorePlanError(f"a forward covers {out.num_rows()} rows, fewer than the {self.num_rows()} before it")
        return out

    def insert(self, n: int, *, response_length: int) -> _Rows:
        # The next forward covers the inserted tokens.
        return self

    @staticmethod
    def expected_rows(*, num_tokens: int, response_length: int) -> int:
        return num_tokens - 1


class RespRows(_Rows):
    """Sampler outputs (the sampling mask): a row per response token."""

    __slots__ = ()
    gather = staticmethod(gather_csr_rows)

    def append(self, new: _Rows, *, response_length: int, seq_start: int) -> _Rows:
        if seq_start:
            raise StorePlanError("sampler rows have no sequence start")
        self._check_aligned(response_length)
        return self.concat(new)

    def insert(self, n: int, *, response_length: int) -> _Rows:
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


# ============================== the field table ==============================


@dataclass(frozen=True)
class Stored:
    """One stored array of a field: the dtypes SGLang may write and the training column's dtype."""

    name: str
    store_dtypes: tuple[str, ...]
    train_dtype: type[np.generic]
    # Stored only in some modes (support log-probs); the others are required.
    optional: bool = False


@dataclass(frozen=True)
class FieldDef:
    name: str
    kind: type[_Rows]
    # The first one's leading dimension counts the object's rows.
    stored: tuple[Stored, ...]
    request_key: str
    # R3 of a multimodal prompt may have a row per media token (legacy rollout).
    allow_media_rows: bool = False
    # An all-zero object means the engine did not capture the field.
    check_nonzero: bool = False
    # The launch flag that makes the field required on every training sample.
    required_by: str | None = None


# Adding a stored dtype = adding it to store_dtypes; train_dtype is the training column's contract.
FIELDS: tuple[FieldDef, ...] = (
    FieldDef(
        ROUTED_EXPERTS,
        SeqRows,
        stored=(Stored(ROUTED_EXPERTS, ("int32",), np.int32),),
        request_key="return_routed_experts",
        allow_media_rows=True,
        check_nonzero=True,
        required_by="use_rollout_routing_replay",
    ),
    FieldDef(
        INDEXER_TOPK,
        SeqRows,
        stored=(Stored(INDEXER_TOPK, ("int32",), np.int32),),
        request_key="return_indexer_topk",
        required_by="use_rollout_indexer_replay",
    ),
    FieldDef(
        SAMPLING_MASK,
        RespRows,
        stored=(
            Stored(MASK_LENGTHS, ("int32",), np.int32),
            Stored(MASK_TOKEN_IDS, ("int32",), np.int32),
            Stored(MASK_LOGPROBS, ("float32",), np.float32, optional=True),
        ),
        request_key="return_sampling_mask",
    ),
)
_FIELD_INDEX = {field.name: i for i, field in enumerate(FIELDS)}


# ============================== one sample's plan ==============================


@dataclass(frozen=True)
class StorePlan:
    """One sample's merge plan; ``Sample.rollout_store_plan``.

    ``fields`` follows ``FIELDS``; None means the field was not requested. ``owned_refs`` holds
    every ref the sample received, used by its segments or not: edits never drop one and merges
    take the union, so the rollout executor can delete every object a sample brought in.
    """

    fields: tuple[_Rows | None, ...]
    owned_refs: tuple[StoreRef, ...]

    def field(self, name: str) -> _Rows | None:
        return self.fields[_FIELD_INDEX[name]]

    def with_field(self, name: str, rows: _Rows | None) -> StorePlan:
        fields = list(self.fields)
        fields[_FIELD_INDEX[name]] = rows
        return replace(self, fields=tuple(fields))

    @classmethod
    def from_response(
        cls,
        ref: StoreRef | None,
        *,
        requested: Collection[str],
        num_tokens: int,
        response_length: int,
        media_rows: int = 0,
        seq_start: int = 0,
        allow_extra_seq_row: bool = False,
        no_output_abort: bool = False,
    ) -> StorePlan:
        """The plan of one response; reads only the ref.

        ``num_tokens`` is the length of the sequence the response's forward covered (prompt plus
        output), ``response_length`` its output length. A requested field must be in the ref,
        except an empty sampling mask of a no-output abort. ``allow_extra_seq_row`` drops the
        stop-edge row the legacy rollout's forward fields may carry.
        """
        specs = {spec.name: spec for spec in ref.fields} if ref is not None else {}
        fields = tuple(
            (
                _field_from_response(
                    field,
                    ref,
                    specs,
                    expected=field.kind.expected_rows(
                        num_tokens=num_tokens - seq_start, response_length=response_length
                    )
                    + (media_rows if field.allow_media_rows else 0),
                    allow_extra_row=allow_extra_seq_row and issubclass(field.kind, SeqRows),
                    no_output_abort=no_output_abort,
                )
                if field.name in requested
                else None
            )
            for field in FIELDS
        )
        return cls(fields=fields, owned_refs=() if ref is None else (ref,))

    def append_response(self, new: StorePlan, *, response_length: int, seq_start: int = 0) -> StorePlan:
        """Add a later response of the same sample; ``response_length`` is the sample's before it."""
        return StorePlan(
            fields=_zip_fields(
                self, new, lambda a, b: a.append(b, response_length=response_length, seq_start=seq_start)
            ),
            owned_refs=_union_refs(self.owned_refs, new.owned_refs),
        )

    def adopt_refs(self, refs: Iterable[StoreRef]) -> StorePlan:
        """Own more objects without using their rows, so they are deleted with the sample."""
        return replace(self, owned_refs=_union_refs(self.owned_refs, refs))

    def insert_tokens(self, n: int, *, response_length: int) -> StorePlan:
        """Tokens Miles inserted into the response, such as a tool observation."""
        fields = tuple(
            None if rows is None else rows.insert(n, response_length=response_length) for rows in self.fields
        )
        return replace(self, fields=fields)

    def merge_turn(self, later: StorePlan, *, observation_tokens: int, response_length: int) -> StorePlan:
        """A session's next turn after ``observation_tokens`` forced tokens; ``response_length`` is this turn's."""
        gap = self.insert_tokens(observation_tokens, response_length=response_length)
        return gap.append_response(later, response_length=response_length + observation_tokens)

    def strip_tail(self, n: int) -> StorePlan:
        """Drop the rows of the last ``n`` tokens."""
        fields = tuple(None if rows is None else rows.prefix(rows.num_rows() - n) for rows in self.fields)
        return replace(self, fields=fields)

    def validate(self, *, num_tokens: int, response_length: int, media_rows: int = 0) -> None:
        owned = {ref.key for ref in self.owned_refs}
        for field, rows in zip(FIELDS, self.fields, strict=True):
            if rows is None:
                continue
            expected = rows.expected_rows(num_tokens=num_tokens, response_length=response_length)
            allowed = (expected, expected + media_rows) if field.allow_media_rows else (expected,)
            if rows.num_rows() not in allowed:
                raise StorePlanError(f"{field.name} has {rows.num_rows()} rows; expected {allowed}")
            if any(seg.ref is not None and seg.ref.key not in owned for seg in rows.segs):
                raise StorePlanError(f"{field.name} uses an object the sample does not own")

    def to_table(self) -> tuple[np.ndarray, list[StoreRef]]:
        """int32 ``[n, 4]`` rows of ``(field, obj, start, stop)``; ``obj`` indexes ``owned_refs``.

        Forced rows have ``obj = -1``; a field present with no rows gets one forced ``(0, 0)`` row.
        """
        index = {ref.key: i for i, ref in enumerate(self.owned_refs)}
        table = []
        for f, rows in enumerate(self.fields):
            if rows is None:
                continue
            if not rows.segs:
                table.append((f, -1, 0, 0))
            for seg in rows.segs:
                if seg.ref is not None and seg.ref.key not in index:
                    raise StorePlanError(f"{FIELDS[f].name} uses an object the sample does not own")
                table.append((f, -1 if seg.ref is None else index[seg.ref.key], seg.start, seg.stop))
        return np.asarray(table, dtype=np.int32).reshape(-1, 4), list(self.owned_refs)

    @classmethod
    def from_table(cls, table: np.ndarray, refs: Sequence[StoreRef]) -> StorePlan:
        if table.ndim != 2 or table.shape[1] != 4:
            raise StorePlanError(f"store plan table must be [n, 4], got {table.shape}")
        segs: dict[int, list[Seg]] = {}
        for f, obj, start, stop in table.tolist():
            if not 0 <= f < len(FIELDS) or not -1 <= obj < len(refs):
                raise StorePlanError(f"store plan row {(f, obj, start, stop)} is out of range")
            rows = segs.setdefault(f, [])
            if obj >= 0 or start != stop:
                rows.append(Seg(None if obj < 0 else refs[obj], start, stop))
        fields = tuple(field.kind(tuple(segs[f])) if f in segs else None for f, field in enumerate(FIELDS))
        return cls(fields=fields, owned_refs=tuple(refs))


def _field_from_response(
    field: FieldDef,
    ref: StoreRef | None,
    specs: Mapping[str, Any],
    *,
    expected: int,
    allow_extra_row: bool,
    no_output_abort: bool,
) -> _Rows:
    rows_spec = specs.get(field.stored[0].name)
    if rows_spec is None:
        if no_output_abort and expected == 0:
            return field.kind()
        raise StorePlanError(f"the request asked for {field.name}, but the response's output store ref has none")
    for stored in field.stored:
        spec = specs.get(stored.name)
        if spec is None and not stored.optional:
            raise StorePlanError(f"output store ref carries {rows_spec.name} without {stored.name}")
        if spec is not None and spec.dtype not in stored.store_dtypes:
            raise StorePlanError(f"stored {stored.name} has dtype {spec.dtype}; Miles accepts {stored.store_dtypes}")
    if (logprobs := specs.get(MASK_LOGPROBS)) is not None and field.name == SAMPLING_MASK:
        if logprobs.shape[0] != specs[MASK_TOKEN_IDS].shape[0]:
            raise StorePlanError(
                f"stored support log-probs have {logprobs.shape[0]} values for "
                f"{specs[MASK_TOKEN_IDS].shape[0]} support entries"
            )
    rows = rows_spec.shape[0]
    if rows != expected and not (allow_extra_row and rows == expected + 1):
        raise StorePlanError(f"stored {rows_spec.name} has {rows} rows; expected {expected}")
    return field.kind((Seg(ref, 0, expected),) if expected else ())


def _zip_fields(a: StorePlan, b: StorePlan, combine) -> tuple[_Rows | None, ...]:
    out = []
    for field, x, y in zip(FIELDS, a.fields, b.fields, strict=True):
        if (x is None) != (y is None):
            raise StorePlanError(f"cannot combine a sample with and without {field.name}")
        out.append(None if x is None else combine(x, y))
    return tuple(out)


def _union_refs(a: Iterable[StoreRef], b: Iterable[StoreRef]) -> tuple[StoreRef, ...]:
    return tuple({ref.key: ref for ref in (*a, *b)}.values())


def combine_plans(a: StorePlan | None, b: StorePlan | None, combine) -> StorePlan | None:
    """Apply a two-plan edit under the strict None rule: both or neither sample has a plan."""
    if a is None and b is None:
        return None
    if a is None or b is None:
        raise StorePlanError("cannot combine a sample with output store data and one without")
    return combine(a, b)


# ============================== objects dropped before conversion ==============================

# Refs of samples the rollout dropped before conversion (filters, oversampling, aborts); the rollout
# executor deletes them with the current rollout's batch. Process-local: drops happen in its process.
_dropped_refs: dict[str, StoreRef] = {}
_dropped_lock = threading.Lock()


def register_dropped_samples(before: Iterable[Any], after: Iterable[Any] = ()) -> None:
    """Register the refs of samples in ``before`` (nested lists) that are not in ``after``; no read, no delete."""
    kept = {ref.key for ref in _owned_refs_of(after)}
    dropped = {ref.key: ref for ref in _owned_refs_of(before) if ref.key not in kept}
    if dropped:
        with _dropped_lock:
            _dropped_refs.update(dropped)


def take_dropped_refs() -> list[StoreRef]:
    with _dropped_lock:
        refs = list(_dropped_refs.values())
        _dropped_refs.clear()
    return refs


def owned_refs_of_samples(samples: Iterable[Any]) -> list[StoreRef]:
    return list({ref.key: ref for ref in _owned_refs_of(samples)}.values())


def _owned_refs_of(node: Iterable[Any]) -> Iterable[StoreRef]:
    for item in node:
        if isinstance(item, (list, tuple)):
            yield from _owned_refs_of(item)
        elif (plan := getattr(item, "rollout_store_plan", None)) is not None:
            yield from plan.owned_refs
