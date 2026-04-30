import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from irsam2_benchmark.evaluation.heatmaps import write_heatmap_artifact


class HeatmapArtifactTests(unittest.TestCase):
    def test_write_heatmap_artifact_saves_raw_overlay_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            heatmap = np.arange(16, dtype=np.float32).reshape(4, 4)
            image = np.zeros((4, 4, 3), dtype=np.uint8)

            paths = write_heatmap_artifact(
                root=root,
                experiment_id="exp",
                dataset="dataset",
                sample_id="sample/1",
                stage="teacher encoder",
                heatmap=heatmap,
                image=image,
                meta={"kind": "unit"},
            )

            self.assertTrue(Path(paths["raw"]).exists())
            self.assertTrue(Path(paths["heatmap"]).exists())
            self.assertTrue(Path(paths["overlay"]).exists())
            self.assertTrue(Path(paths["meta"]).exists())
            with np.load(paths["raw"]) as payload:
                self.assertEqual(payload["heatmap"].shape, (4, 4))
                self.assertEqual(payload["heatmap_normalized"].shape, (4, 4))
            with Image.open(paths["overlay"]) as overlay:
                self.assertEqual(overlay.size, (4, 4))
            meta = json.loads(Path(paths["meta"]).read_text(encoding="utf-8"))
            self.assertEqual(meta["kind"], "unit")
            self.assertEqual(meta["shape"], [4, 4])


if __name__ == "__main__":
    unittest.main()
