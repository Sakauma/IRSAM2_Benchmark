import unittest

import torch

from irsam2_benchmark.adapters import AdapterFactory
from irsam2_benchmark.decoders import DecoderFactory
from irsam2_benchmark.distillers import DistillerFactory
from irsam2_benchmark.losses import LossFactory
from irsam2_benchmark.priors import PriorFactory
from irsam2_benchmark.prompts import PromptFactory
from irsam2_benchmark.quantizers import QuantizerFactory


class ModularComponentTests(unittest.TestCase):
    def test_factories_build_p0_modules(self):
        image = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
        image[:, :, 8, 8] = 1.0
        prior = PriorFactory.build({"name": "prior_fusion", "enabled": ["local_contrast"], "scales": [3]})
        prompt = PromptFactory.build({"name": "heuristic_physics", "percentile": 99.0, "min_component_area": 1})
        maps = prior(image)
        result = prompt(image, maps)
        self.assertIn("fused", maps)
        self.assertIsNotNone(result["point"])
        self.assertIsNotNone(result["box"])

    def test_placeholder_factories_are_constructible(self):
        tensor = torch.ones((1, 1, 4, 4), dtype=torch.float32)
        self.assertTrue(torch.equal(AdapterFactory.build({"name": "identity"})(tensor), tensor))
        self.assertTrue(torch.equal(DecoderFactory.build({"name": "sam2_mask_decoder"})(tensor), tensor))
        self.assertEqual(float(LossFactory.build({"name": "none"})(tensor)), 0.0)
        self.assertEqual(float(DistillerFactory.build({"name": "none"})(tensor)), 0.0)
        self.assertIs(QuantizerFactory.build({"name": "none"})(tensor), tensor)


if __name__ == "__main__":
    unittest.main()

