import json
import os

from radixscale.bridge.sglang_adapter import to_sglang_named_tensors
from radixscale.bridge.weight_sync import iter_full_weights_by_module

from miles.backends.training_utils.weight_update.hf_weight_iterator import HfWeightIteratorBase
from miles.backends.training_utils.weight_update.hf_weight_iterator.atomic_groups import get_hf_atomic_update_groups
from miles.backends.training_utils.weight_update.hf_weight_iterator.checkpoint_towers import (
    iter_checkpoint_tower_units,
)


def _vlm_prefixed(hf_checkpoint: str) -> bool:
    with open(os.path.join(hf_checkpoint, "config.json")) as f:
        return "text_config" in json.load(f)


class RadixscaleHfWeightIterator(HfWeightIteratorBase):
    """One HF-named unit per gathered radixscale parameter."""

    forced_placement = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vlm_prefixed = _vlm_prefixed(self.args.hf_checkpoint)

    def _iter_hf_param_units(self, weights, *, materialize):
        assert weights is None, "the radixscale backend streams live parameters only"
        for bucket in iter_full_weights_by_module(self.model):
            if materialize:
                for name, tensor in bucket:
                    yield list(to_sglang_named_tensors([(name, tensor)], vlm_prefixed=self._vlm_prefixed))
        yield from iter_checkpoint_tower_units(self.args.hf_checkpoint, materialize=materialize)

    def _hf_atomic_update_groups(self):
        return get_hf_atomic_update_groups(self.model_name)

    def _iter_hf_adapter_units(self, adapter, *, materialize):
        raise NotImplementedError("the radixscale backend has no LoRA")
