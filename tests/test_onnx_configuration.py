"""Loading compatibility; numerical ONNX integration is checked by --verify."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from onnx_example import OnnxLiteTracker


class OnnxConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        # Configuration checks do not need 100 MB of graph files. Real graph
        # loading and predictions are exercised by the export verifier.
        self.sessions = patch("onnxruntime.InferenceSession").start()
        self.addCleanup(patch.stopall)

    def test_legacy_folder_and_original_positional_arguments(self):
        tracker = OnnxLiteTracker(
            str(self.root),
            8,
            4,
            3,
            4,
            (256, 320),
            2,
            1110,
            128,
            ["CPUExecutionProvider"],
        )
        self.assertEqual(tracker.model_resolution, (256, 320))
        self.assertEqual(tracker.window_len, 8)
        self.assertEqual(tracker.iters, 2)
        self.assertEqual(tuple(tracker.time_emb.shape), (1, 8, 1110))
        self.assertEqual(self.sessions.call_count, 3)
        self.assertIsNone(tracker.coords_buffer)

    def test_legacy_defaults_require_no_metadata(self):
        tracker = OnnxLiteTracker(str(self.root), providers=["CPUExecutionProvider"])
        self.assertEqual(tracker.model_resolution, (384, 512))
        self.assertEqual(tracker.window_len, 16)
        self.assertEqual(tuple(tracker.time_emb.shape), (1, 16, 1110))

    def test_metadata_restores_settings_and_rejects_conflicting_overrides(self):
        config = {
            "format_version": 1,
            "window_len": 8,
            "stride": 4,
            "corr_radius": 3,
            "corr_levels": 4,
            "model_resolution": [256, 320],
            "iters": 2,
            "input_dim": 1110,
            "latent_dim": 128,
            "inv_sigmoid_true_val": 4.6,
        }
        (self.root / "tracker_config.json").write_text(json.dumps(config))
        embedding = np.ones((1, 8, 1110), dtype=np.float32)
        np.save(self.root / "time_emb.npy", embedding)
        tracker = OnnxLiteTracker(str(self.root), providers=["CPUExecutionProvider"])
        self.assertEqual(tracker.window_len, 8)
        self.assertEqual(tracker.model_resolution, (256, 320))
        torch.testing.assert_close(tracker.time_emb, torch.from_numpy(embedding))
        self.sessions.reset_mock()
        with self.assertRaisesRegex(ValueError, "window_len"):
            OnnxLiteTracker(str(self.root), window_len=16)
        self.sessions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
