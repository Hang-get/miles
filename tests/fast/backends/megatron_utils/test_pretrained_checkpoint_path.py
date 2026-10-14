from types import SimpleNamespace
from unittest.mock import patch

from miles.backends.megatron_utils import checkpoint as checkpoint_module


def test_hf_pretrained_fallback(tmp_path) -> None:
    pretrained = tmp_path / "pretrained"
    pretrained.mkdir()
    (pretrained / "model.safetensors").touch()
    args = SimpleNamespace(load=None, pretrained_checkpoint=str(pretrained))
    with (
        patch.object(checkpoint_module, "get_args", return_value=args),
        patch.object(checkpoint_module, "is_lora_enabled", return_value=False),
        patch.object(checkpoint_module, "_load_checkpoint_hf", return_value=(0, 0)) as load_hf,
    ):
        assert checkpoint_module.load_checkpoint([], None, None, None, False) == (0, 0, False)
        assert load_hf.call_args.kwargs["load_path"] == str(pretrained)
        assert args.load is None


def test_explicit_load_has_priority(tmp_path) -> None:
    requested = tmp_path / "requested"
    requested.mkdir()
    (requested / "model.safetensors").touch()
    args = SimpleNamespace(load=str(requested), pretrained_checkpoint="unused")
    with (
        patch.object(checkpoint_module, "get_args", return_value=args),
        patch.object(checkpoint_module, "is_lora_enabled", return_value=False),
        patch.object(checkpoint_module, "_load_checkpoint_hf", return_value=(0, 0)) as load_hf,
    ):
        checkpoint_module.load_checkpoint([], None, None, None, False)
        assert load_hf.call_args.kwargs["load_path"] == str(requested)
