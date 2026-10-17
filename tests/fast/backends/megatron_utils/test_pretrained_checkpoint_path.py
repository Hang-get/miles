"""Exercise real checkpoint path validation and dispatch without loading weights."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from miles.backends.megatron_utils import checkpoint


class PretrainedCheckpointPathTest(unittest.TestCase):
    def check_path(self, resume: bool) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base, existing = root / "base", root / "resume"
            for directory in (base, existing):
                directory.mkdir()
                (directory / "config.json").write_text("{}")
            args = SimpleNamespace(load=str(existing) if resume else None,
                                   pretrained_checkpoint=str(base), finetune=True)
            expected = existing if resume else base
            with patch.object(checkpoint, "get_args", return_value=args), patch.object(
                checkpoint, "_load_checkpoint_hf", return_value=(0, 0)
            ) as loader, patch.object(checkpoint.ModelCompanionSampleConsumptionUtils, "clear_for_finetune"):
                result = checkpoint.load_checkpoint([], None, None, None, False)
            self.assertEqual(loader.call_args.kwargs["load_path"], str(expected))
            self.assertEqual(result.iteration, 0)
            self.assertFalse(result.restored_trained_iteration)

    def test_fresh_pretrained_fallback(self) -> None:
        self.check_path(resume=False)

    def test_existing_resume_has_priority(self) -> None:
        self.check_path(resume=True)


if __name__ == "__main__":
    unittest.main()
