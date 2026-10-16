"""Step5 native LoRA over the head-wise-gated ``linear_qkv``: shapes, names, merge math, targets — no GPU."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file
from tests.fast.backends.megatron_utils.lora.native.naming_stub import named

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named, load_lora_adapter_hf
from miles.backends.megatron_utils.lora.native.merge import merge_lora_into_weights
from miles.backends.megatron_utils.lora.native.naming import ConverterNaming
from miles.backends.megatron_utils.lora.native.registry import (
    MODEL_SPECS,
    resolve_adapter_targets,
    resolve_config_spec,
)
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles.utils.lora.hf_lora_targets import resolve_hf_lora_targets
from miles_plugins.models.step5.qkv_layout import fuse_qkv_and_gate, split_qkv_and_gate

HIDDEN, HEADS, GROUPS, HEAD_DIM, RANK, ALPHA = 16, 8, 4, 3, 2, 6
PREFIX = "model.layers.1.self_attn."
STEP5 = MODEL_SPECS["step5"]
SCALE = ALPHA / RANK


class _Linear(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.clone())

    def forward(self, x):
        return x @ self.weight.t(), None


@pytest.fixture(autouse=True)
def _single_process_tp(monkeypatch):
    """Each TP rank is built in this process; the column branch's TP-region copy is the identity forward."""
    from megatron.core.tensor_parallel import mappings

    monkeypatch.setattr(mappings, "copy_to_tensor_model_parallel_region", lambda x, group=None: x)


def _base_weights():
    torch.manual_seed(0)
    return {
        "q": torch.randn(HEADS * HEAD_DIM, HIDDEN),
        "k": torch.randn(GROUPS * HEAD_DIM, HIDDEN),
        "v": torch.randn(GROUPS * HEAD_DIM, HIDDEN),
        "g": torch.randn(HEADS, HIDDEN),
        "o": torch.randn(HIDDEN, HEADS * HEAD_DIM),
    }


def _hf_adapter():
    """Full HF adapter tensors as SGLang receives them: q/k/v/o only, B nonzero."""
    torch.manual_seed(1)
    rows = {"q_proj": HEADS * HEAD_DIM, "k_proj": GROUPS * HEAD_DIM, "v_proj": GROUPS * HEAD_DIM, "o_proj": HIDDEN}
    tensors = {}
    for leaf, out_rows in rows.items():
        in_features = HEADS * HEAD_DIM if leaf == "o_proj" else HIDDEN
        tensors[f"{PREFIX}{leaf}.lora_A.weight"] = torch.randn(RANK, in_features)
        tensors[f"{PREFIX}{leaf}.lora_B.weight"] = torch.randn(out_rows, RANK)
    return tensors


def _delta(adapter, leaf):
    return SCALE * adapter[f"{PREFIX}{leaf}.lora_B.weight"] @ adapter[f"{PREFIX}{leaf}.lora_A.weight"]


def _context(tp_size, tp_rank):
    return AttachContext(
        lora=LoRAConfig(rank=RANK, alpha=ALPHA, dropout=0.0, target_modules=None),
        transformer_config=SimpleNamespace(
            hidden_size=HIDDEN,
            sequence_parallel=False,
            num_query_groups=GROUPS,
            gated_linear_unit=True,
            head_wise_attn_gate=True,
            attention_output_gate=False,
        ),
        tp_size=tp_size,
        tp_rank=tp_rank,
        naming=None,
    )


def _rank_attention(base, tp_size, tp_rank):
    """One TP rank's Step5 attention: its block of the fused [q;k;v;gate] and its o_proj input columns."""
    fused = fuse_qkv_and_gate(
        query=base["q"], key=base["k"], value=base["v"], gate=base["g"], num_query_groups=GROUPS, tp_size=tp_size
    )
    attention = nn.Module()
    attention.linear_qkv = _Linear(fused.chunk(tp_size)[tp_rank])
    attention.linear_proj = _Linear(base["o"].chunk(tp_size, dim=1)[tp_rank])
    attention.num_attention_heads_per_partition = HEADS // tp_size
    attention.num_query_groups_per_partition = GROUPS // tp_size
    attention.hidden_size_per_attention_head = HEAD_DIM
    model = nn.Module()
    model.attn = attention
    context = _context(tp_size, tp_rank)
    STEP5.validate(context)
    assert STEP5.attention.attach(attention, None, named(context, PREFIX)) == 2
    return model


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_adapters_cover_qkv_and_leave_the_gate_frozen(tmp_path, tp_size):
    base, adapter = _base_weights(), _hf_adapter()
    save_file(adapter, str(tmp_path / "adapter_model.safetensors"))
    ranks = [_rank_attention(base, tp_size, rank) for rank in range(tp_size)]

    merged_qkv, merged_o = [], []
    for rank, model in enumerate(ranks):
        qkv_adapter = model.attn.lora_qkv_adapter
        assert {name for name, _ in qkv_adapter.named_parameters()} == {
            f"{p}_{m}" for p in "qkv" for m in "AB"
        }, "the gate rows must carry no adapter"
        assert qkv_adapter.q_B.shape == (HEADS // tp_size * HEAD_DIM, RANK)
        assert qkv_adapter.k_B.shape == qkv_adapter.v_B.shape == (GROUPS // tp_size * HEAD_DIM, RANK)
        assert model.attn.lora_o_adapter.o_A.shape == (RANK, HEADS // tp_size * HEAD_DIM)
        assert load_lora_adapter_hf([model], str(tmp_path)) == 8

        weights = {name: param for name, param in model.named_parameters() if "lora" not in name}
        merged = merge_lora_into_weights([model], weights)
        merged_qkv.append(merged["attn.linear_qkv.weight"])
        merged_o.append(merged["attn.linear_proj.weight"])

        x = torch.randn(5, HIDDEN)
        with torch.no_grad():
            torch.testing.assert_close(
                model.attn.linear_qkv(x)[0], x @ merged_qkv[-1].t(), rtol=1e-5, atol=1e-5, msg=f"rank {rank}"
            )

    # The converter's split of the gathered merged weight is base + delta under each HF name.
    query, key, value, gate = split_qkv_and_gate(
        torch.cat(merged_qkv),
        num_query_groups=GROUPS,
        num_attention_heads=HEADS,
        head_dim=HEAD_DIM,
        tp_size=tp_size,
    )
    torch.testing.assert_close(query, base["q"] + _delta(adapter, "q_proj"))
    torch.testing.assert_close(key, base["k"] + _delta(adapter, "k_proj"))
    torch.testing.assert_close(value, base["v"] + _delta(adapter, "v_proj"))
    assert torch.equal(gate, base["g"])
    torch.testing.assert_close(torch.cat(merged_o, dim=1), base["o"] + _delta(adapter, "o_proj"))


def test_export_names_match_the_hf_projections(tmp_path):
    base, adapter = _base_weights(), _hf_adapter()
    save_file(adapter, str(tmp_path / "adapter_model.safetensors"))
    model = _rank_attention(base, tp_size=1, tp_rank=0)
    load_lora_adapter_hf([model], str(tmp_path))

    exported = dict(export_lora_hf_named([model]))
    assert set(exported) == set(adapter)
    for name, tensor in exported.items():
        torch.testing.assert_close(tensor.float(), adapter[name], rtol=1e-2, atol=1e-2)


def test_the_host_layout_is_checked():
    """A linear_qkv without the gate rows (an ungated GQA) is not Step5's layout."""
    attention = nn.Module()
    attention.linear_qkv = _Linear(torch.randn((HEADS + 2 * GROUPS) * HEAD_DIM, HIDDEN))
    attention.linear_proj = _Linear(torch.randn(HIDDEN, HEADS * HEAD_DIM))
    attention.num_attention_heads_per_partition = HEADS
    attention.num_query_groups_per_partition = GROUPS
    attention.hidden_size_per_attention_head = HEAD_DIM
    with pytest.raises(AssertionError, match="local rows"):
        STEP5.attention.attach(attention, None, named(_context(1, 0), PREFIX))


def _layer(layer_number, **weights):
    layer = nn.Module()
    layer.layer_number = layer_number
    for path, (shape, partition_dim) in weights.items():
        *modules, leaf = path.split("__")
        owner = layer
        for name in modules:
            if not hasattr(owner, name):
                owner.add_module(name, nn.Module())
            owner = getattr(owner, name)
        weight = nn.Parameter(torch.empty(shape))
        weight.tensor_model_parallel = partition_dim is not None
        weight.partition_dim = -1 if partition_dim is None else partition_dim
        owner.register_parameter(leaf, weight)
    return layer


def test_converter_names_qkv_gate_dense_and_shared_expert_hosts():
    args = SimpleNamespace(
        num_query_groups=GROUPS, num_attention_heads=HEADS, kv_channels=HEAD_DIM, tensor_model_parallel_size=2
    )
    qkv_rows = (HEADS + 2 * GROUPS) * HEAD_DIM + HEADS
    layer = _layer(
        2,
        self_attention__linear_qkv__weight=((qkv_rows // 2, HIDDEN), 0),
        self_attention__linear_proj__weight=((HIDDEN, HEADS * HEAD_DIM // 2), 1),
        mlp__shared_experts__linear_fc1__weight=((6, HIDDEN), 0),
        mlp__shared_experts__linear_fc2__weight=((HIDDEN, 3), 1),
    )
    dense = _layer(1, mlp__linear_fc1__weight=((10, HIDDEN), 0))
    naming = ConverterNaming(args, "step5", tp_size=2, is_hf_name=lambda _name: True)

    def prefix(layer, path, leaves):
        owner = layer
        for name in path.split("__"):
            owner = getattr(owner, name)
        return naming.block_prefix(layer, owner, leaves)

    leaves = [p.hf_leaf for p in STEP5.attention.layout.fused[0].projections]
    qkv_leaves = [*leaves, *STEP5.attention.layout.fused[0].unadapted_leaves]
    assert prefix(layer, "self_attention__linear_qkv__weight", qkv_leaves) == "model.layers.1.self_attn."
    with pytest.raises(AssertionError, match="g_proj"):
        prefix(layer, "self_attention__linear_qkv__weight", leaves)
    assert prefix(layer, "self_attention__linear_proj__weight", ["o_proj"]) == "model.layers.1.self_attn."
    assert prefix(layer, "mlp__shared_experts__linear_fc1__weight", ["gate_proj", "up_proj"]) == (
        "model.layers.1.share_expert."
    )
    assert prefix(layer, "mlp__shared_experts__linear_fc2__weight", ["down_proj"]) == "model.layers.1.share_expert."
    assert prefix(dense, "mlp__linear_fc1__weight", ["gate_proj", "up_proj"]) == "model.layers.0.mlp."


def _step5_vl_config(moe_layer_list=(1, 2, 3)):
    return {
        "model_type": "step5_vl",
        "text_config": {"model_type": "step5", "num_hidden_layers": 5, "moe_layer_list": list(moe_layer_list)},
    }


def _resolve(hf_config, targets):
    return resolve_adapter_targets(
        hf_config, targets, hf_modules=[], lora_type="lora", experts_shared_outer_loras=False
    )


def test_all_linear_is_attention_dense_mlp_and_shared_expert():
    config = _step5_vl_config()
    assert resolve_config_spec(config) == ("step5", STEP5)
    targets = resolve_hf_lora_targets(config, target_modules=["all-linear"])
    assert targets == [
        *(f"model.layers.*.self_attn.{leaf}" for leaf in ("q_proj", "k_proj", "v_proj", "o_proj")),
        *(
            f"model.layers.*.{block}.{leaf}"
            for block in ("mlp", "share_expert")
            for leaf in ("gate_proj", "up_proj", "down_proj")
        ),
    ]
    assert _resolve(config, targets) == targets
    assert resolve_hf_lora_targets(_step5_vl_config(moe_layer_list=range(5)), target_modules=["mlp"]) == [
        f"model.layers.*.share_expert.{leaf}" for leaf in ("gate_proj", "up_proj", "down_proj")
    ]


@pytest.mark.parametrize(
    "target", ["model.layers.*.self_attn.g_proj", "model.layers.*.moe.experts.*.gate_proj"], ids=["gate", "routed"]
)
def test_the_gate_and_routed_experts_are_rejected(target):
    with pytest.raises(AssertionError, match="does not implement adapters"):
        _resolve(_step5_vl_config(), [target])
