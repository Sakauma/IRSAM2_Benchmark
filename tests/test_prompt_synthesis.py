import unittest

import numpy as np

from irsam2_benchmark.data.prompt_synthesis import connected_components, expand_box_xyxy, mask_to_point_prompt, mask_to_tight_box


class PromptSynthesisTests(unittest.TestCase):
    def test_mask_to_tight_box(self):
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[2:5, 3:7] = 1.0
        self.assertEqual(mask_to_tight_box(mask), [3.0, 2.0, 7.0, 5.0])

    def test_expand_box(self):
        loose = expand_box_xyxy([3, 2, 7, 5], width=10, height=10, pad_ratio=0.25, min_pad=1.0, min_side=4.0)
        self.assertLessEqual(loose[0], 3.0)
        self.assertLessEqual(loose[1], 2.0)
        self.assertGreaterEqual(loose[2], 7.0)
        self.assertGreaterEqual(loose[3], 5.0)

    def test_default_expand_box_caps_small_target_growth(self):
        loose = expand_box_xyxy([111, 178, 114, 181], width=256, height=256)
        self.assertEqual(loose, [109.5, 176.5, 115.5, 182.5])

    def test_connected_components(self):
        mask = np.zeros((5, 5), dtype=np.float32)
        mask[0, 0] = 1.0
        mask[4, 4] = 1.0
        components = connected_components(mask)
        self.assertEqual(len(components), 2)


if __name__ == "__main__":
    unittest.main()
