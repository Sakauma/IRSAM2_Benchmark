import tempfile
import unittest
from pathlib import Path

from irsam2_benchmark.models.learned_auto_prompt import (
    AutoPromptModelConfig,
    _require_torch,
    build_ir_prompt_net,
    load_auto_prompt_model,
    save_auto_prompt_checkpoint,
)


def _parameter_count(model) -> int:
    return sum(int(param.numel()) for param in model.parameters())


class LearnedAutoPromptModelTests(unittest.TestCase):
    def test_ir_prompt_v2_keeps_output_contract_and_is_larger_than_small(self):
        torch, _, _ = _require_torch()
        x = torch.randn(2, 3, 16, 16)
        small = build_ir_prompt_net(AutoPromptModelConfig(hidden_channels=8))
        v2 = build_ir_prompt_net(AutoPromptModelConfig(architecture="ir_prompt_v2", hidden_channels=8))

        outputs = v2(x)

        self.assertEqual(getattr(v2, "model_name"), "IRPromptNetV2")
        self.assertEqual(tuple(outputs["objectness_logits"].shape), (2, 1, 16, 16))
        self.assertEqual(tuple(outputs["box_size"].shape), (2, 2, 16, 16))
        self.assertEqual(tuple(outputs["confidence_logits"].shape), (2, 1))
        self.assertGreater(_parameter_count(v2), _parameter_count(small) * 3)

    def test_ir_prompt_v2_checkpoint_round_trip_preserves_outputs(self):
        torch, _, _ = _require_torch()
        torch.manual_seed(7)
        config = AutoPromptModelConfig(architecture="ir_prompt_v2", hidden_channels=8)
        model = build_ir_prompt_net(config)
        model.eval()
        x = torch.randn(1, 3, 12, 12)
        with torch.no_grad():
            expected = model(x)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "checkpoint.pt"
            save_auto_prompt_checkpoint(checkpoint_path, model, config=config, metadata={"unit": True})
            loaded, info = load_auto_prompt_model(checkpoint_path)
            with torch.no_grad():
                actual = loaded(x)

        self.assertEqual(getattr(loaded, "model_name"), "IRPromptNetV2")
        self.assertEqual(info["config"]["architecture"], "ir_prompt_v2")
        self.assertTrue(info["metadata"]["unit"])
        self.assertTrue(torch.allclose(expected["objectness_logits"], actual["objectness_logits"], atol=1e-6))
        self.assertTrue(torch.allclose(expected["box_size"], actual["box_size"], atol=1e-6))
        self.assertTrue(torch.allclose(expected["confidence_logits"], actual["confidence_logits"], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
