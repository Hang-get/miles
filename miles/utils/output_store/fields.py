"""The fields SGLang stores, and how the trainer turns a field's rows into training columns.

A field combines two independent parts with what only it knows:

- rows (``miles.utils.output_store.rows``): how sample events edit its segments; the trainer expands
  them into each row's source row (``expand_rows``);
- layout: how its stored arrays hold a row, dense or CSR; ``gather_dense_rows`` /
  ``gather_csr_rows`` copy the listed rows;
- the field: what a forced row holds, what to check, which training columns to write.

Each field's ``materialize`` runs the same steps on the shard: stack its objects, expand the rows,
fill the forced rows, gather, write the columns. ``plan.FIELDS`` lists the fields.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import torch

from miles.utils.output_store.materialize import (
    StoreExecError,
    check_csr_objects,
    check_support,
    compact_candidates,
    expand_rows,
    gather_csr_rows,
    gather_dense_rows,
    redirect_forced_rows,
    stack_objects,
    support_topk,
)
from miles.utils.output_store.rows import RespRows, Rows, SeqRows, StorePlanError

# The shard's objects a field uses: store key -> stored name -> array; segments index this order.
Objects = Mapping[str, Mapping[str, np.ndarray]]
# (seg_obj, seg_start, seg_stop, sample_seg_offsets) of the shard's samples, as expand_rows takes them.
Segments = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Stored:
    """One stored array of a field: the dtypes SGLang may write and the training column's dtype."""

    name: str
    store_dtypes: tuple[str, ...]
    train_dtype: type[np.generic]
    # Stored only in some modes (support log-probs); the others are required.
    optional: bool = False


@dataclass(frozen=True, kw_only=True)
class FieldDef:
    """One stored field: what its plan needs (rows kind, stored arrays, request key) and its columns."""

    kind: ClassVar[type[Rows]]
    name: str
    # The first one's leading dimension counts the object's rows.
    stored: tuple[Stored, ...]
    request_key: str
    # The launch flag that makes the field required on every training sample.
    required_by: str | None = None

    def check_specs(self, specs: Mapping[str, Any]) -> None:
        """Checks of a response ref's specs beyond each stored array's presence, dtype and rows."""

    def materialize(
        self, args: Namespace, rollout_data: Mapping[str, Any], objects: Objects, segments: Segments
    ) -> dict[str, list]:
        """The field's training columns, a value per sample of the shard."""
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class DenseForwardField(FieldDef):
    """A forward field stored dense, a row per forward position; never forced."""

    kind: ClassVar[type[Rows]] = SeqRows
    column: str

    def materialize(
        self, args: Namespace, rollout_data: Mapping[str, Any], objects: Objects, segments: Segments
    ) -> dict[str, list]:
        (stored,) = self.stored
        for key, arrays in objects.items():
            self._check_object(args, key, arrays[stored.name])
        values, obj_row_offsets = _stack(objects, stored)
        index = expand_rows(*segments, obj_row_offsets)
        # No fill: a forced row has no source, which the gather rejects.
        rows = gather_dense_rows(values, index.source, out_dtype=np.dtype(stored.train_dtype))
        return {self.column: _split(rows, index.sample_row_offsets)}

    def _check_object(self, args: Namespace, key: str, values: np.ndarray) -> None:
        """The field's checks of one used object's array."""


@dataclass(frozen=True, kw_only=True)
class RoutedExpertsField(DenseForwardField):
    """R3: ``[rows, layers, topk]`` routed experts."""

    def _check_object(self, args: Namespace, key: str, values: np.ndarray) -> None:
        # As on the inline path: an all-zero object means the engine skipped routed-expert capture.
        if len(values) and not values.any():
            raise StoreExecError(
                f"routed_experts payload is all zeros: the sglang engine did not capture routed experts "
                f"(topk-bypassing --moe-runner-backend such as flashinfer_trtllm?); output store object {key}"
            )


@dataclass(frozen=True, kw_only=True)
class IndexerTopkField(DenseForwardField):
    """Indexer top-k: ``[rows, streams, topk]``, a stream per DSA indexer layer."""

    def _check_object(self, args: Namespace, key: str, values: np.ndarray) -> None:
        # As on the inline path: streams map to the model's indexer layers by position.
        expected = getattr(args, "rollout_indexer_topk_num_streams", None)
        if expected is not None and values.shape[1] != expected:
            raise ValueError(
                f"output store object {key} has {self.name} with {values.shape[1]} streams but the model has "
                f"{expected} indexer layers; replaying it would map streams to the wrong layers."
            )


@dataclass(frozen=True, kw_only=True)
class SamplingMaskField(FieldDef):
    """The sampling mask, a CSR support per response token: stored as (lengths, token ids, log-probs),
    the log-probs in support mode only. A forced row was not sampled: its support is its token alone."""

    kind: ClassVar[type[Rows]] = RespRows

    def check_specs(self, specs: Mapping[str, Any]) -> None:
        _, ids, logprobs = self.stored
        if (spec := specs.get(logprobs.name)) is not None and spec.shape[0] != specs[ids.name].shape[0]:
            raise StorePlanError(
                f"stored support log-probs have {spec.shape[0]} values for {specs[ids.name].shape[0]} support entries"
            )

    def materialize(
        self, args: Namespace, rollout_data: Mapping[str, Any], objects: Objects, segments: Segments
    ) -> dict[str, list]:
        support_mode = args.rollout_sampling_logprobs_mode == "support"
        lengths_stored, ids_stored, logprobs_stored = self.stored
        value_stored = (ids_stored, logprobs_stored) if support_mode else (ids_stored,)
        if support_mode and (
            missing := [key for key, arrays in objects.items() if logprobs_stored.name not in arrays]
        ):
            raise ValueError(
                f"support log-prob mode needs {logprobs_stored.name}; output store objects {missing[:4]} lack it"
            )
        lengths, obj_row_offsets = _stack(objects, lengths_stored)
        values, value_offsets = zip(*(_stack(objects, stored) for stored in value_stored), strict=True)
        check_csr_objects(lengths, obj_row_offsets, value_offsets)

        index = expand_rows(*segments, obj_row_offsets)
        tokens, token_offsets = _response_tokens(rollout_data)
        if not np.array_equal(index.sample_row_offsets, token_offsets):
            raise ValueError("sampling mask rows do not match the response lengths")

        # Fill rows after the objects: a forced row's support is its response token, with log-prob 0,
        # which no column reads (top-k and compaction skip forced rows).
        forced = index.source < 0
        fill = (tokens[forced], np.zeros(int(forced.sum())))[: len(values)]
        lengths = np.concatenate((lengths, np.ones(int(forced.sum()), dtype=lengths.dtype)))
        values = [np.concatenate((array, part.astype(array.dtype))) for array, part in zip(values, fill, strict=True)]
        row_offsets, gathered = gather_csr_rows(
            lengths,
            values,
            redirect_forced_rows(index.source, fill_start=int(obj_row_offsets[-1])),
            out_dtypes=[np.dtype(stored.train_dtype) for stored in value_stored],
        )
        ids, logprobs = gathered[0], (gathered[1] if support_mode else None)
        check_support(ids, row_offsets, tokens, exactly_once=support_mode)
        return self._columns(args, rollout_data, index.sample_row_offsets, row_offsets, ids, logprobs, forced)

    def _columns(
        self,
        args: Namespace,
        rollout_data: Mapping[str, Any],
        sample_row_offsets: np.ndarray,
        row_offsets: np.ndarray,
        ids: np.ndarray,
        logprobs: np.ndarray | None,
        forced: np.ndarray,
    ) -> dict[str, list]:
        """The inline path's columns: the CSR mask and, in support mode (``logprobs`` given), the
        top-k candidates built from it."""
        pairs = _pairs(sample_row_offsets)
        columns = {
            "rollout_sampling_mask_ids": [torch.from_numpy(ids[row_offsets[a] : row_offsets[b]]) for a, b in pairs],
            "rollout_sampling_mask_offsets": [
                torch.from_numpy(row_offsets[a : b + 1] - row_offsets[a]) for a, b in pairs
            ],
        }
        support_mode = logprobs is not None
        if (k := args.rollout_top_logprobs_num) and support_mode:
            topk_ids, topk_logprobs = support_topk(ids, row_offsets, logprobs, forced, k)
            columns["rollout_topk_token_ids"] = _split(topk_ids, sample_row_offsets)
            columns["rollout_topk_log_probs"] = _split(topk_logprobs, sample_row_offsets)
        if args.ci_test and k:
            _validate_topk_candidates({**rollout_data, **columns}, k)
        if k and support_mode and args.loss_type == "score_centering":
            lengths = compact_candidates(row_offsets, forced)
            columns["rollout_topk_lengths"] = [torch.from_numpy(x) for x in _split(lengths, sample_row_offsets)]
            del columns["rollout_topk_token_ids"]
        return columns


# ============================== helpers ==============================


def _stack(objects: Objects, stored: Stored) -> tuple[np.ndarray, np.ndarray]:
    return stack_objects([arrays[stored.name] for arrays in objects.values()], empty_dtype=stored.train_dtype)


def _split(values: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [values[a:b] for a, b in _pairs(offsets)]


def _pairs(offsets: np.ndarray) -> list[tuple[int, int]]:
    return list(zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True))


def _response_tokens(rollout_data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lengths = [int(n) for n in rollout_data["response_lengths"]]
    parts = [
        np.asarray(tokens[len(tokens) - n :], dtype=np.int64)
        for tokens, n in zip(rollout_data["tokens"], lengths, strict=True)
    ]
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    return (np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)), offsets


def _validate_topk_candidates(rollout_data: Mapping[str, Any], k: int) -> None:
    """--ci-test: the inline path's per-sample candidate check, now that the sampling mask is assembled."""
    # Imported here: miles.utils.types imports this module through the plan.
    from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample
    from miles.utils.sampling_mask import RolloutSamplingMask
    from miles.utils.types import Sample

    for i, tokens in enumerate(rollout_data["tokens"]):
        sample = Sample(
            tokens=list(tokens),
            response_length=int(rollout_data["response_lengths"][i]),
            loss_mask=list(rollout_data["loss_masks"][i]),
            rollout_log_probs=list(rollout_data["rollout_log_probs"][i]),
            rollout_topk_token_ids=rollout_data["rollout_topk_token_ids"][i],
            rollout_topk_log_probs=rollout_data["rollout_topk_log_probs"][i],
            rollout_sampling_mask=RolloutSamplingMask(
                ids=rollout_data["rollout_sampling_mask_ids"][i],
                offsets=rollout_data["rollout_sampling_mask_offsets"][i],
            ),
        )
        validate_rollout_topk_logprobs_sample(sample, k)
