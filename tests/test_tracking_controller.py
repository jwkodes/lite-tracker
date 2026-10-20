import unittest

import numpy as np
import torch

from managed_demo import TrackingController
from src.lite_tracker import LiteTracker
from src.managed_lite_tracker import ManagedLiteTracker
from src.model_utils import get_points_on_a_grid


class TrackingControllerTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.model = LiteTracker(model_resolution=(64, 64)).eval()
        self.tracker = ManagedLiteTracker(self.model, max_points=9)
        self.controller = TrackingController(
            target_points=9,
            grid_size=3,
            min_distance=8,
            min_keyframe_gap=3,
            max_keyframe_gap=8,
            patience=2,
        )
        self.frame = torch.zeros(1, 3, 64, 64)
        self.visibility = True
        self.positions = None
        original = self.model.forward

        def observations(frame, queries):
            local_frame = self.model.online_ind
            coords, vis, conf = original(frame, queries)
            if local_frame > 0:
                coords = (
                    queries[:, None, :, 1:].clone()
                    if self.positions is None
                    else torch.as_tensor(self.positions)[None, None].float()
                )
                vis = torch.as_tensor(
                    np.broadcast_to(
                        self.visibility, (1, 1, len(self.tracker.track_ids))
                    ).copy()
                )
            return coords, vis, conf

        self.model.forward = observations

    def step(self, visible=True, positions=None):
        self.visibility = visible
        self.positions = positions
        outputs = self.controller.process_frame(self.tracker, self.frame)
        return outputs, dict(self.controller.event)

    def test_initial_grid_and_ordinary_frames_preserve_session(self):
        _, event = self.step()
        self.assertEqual(event["added"], list(range(9)))
        torch.testing.assert_close(
            self.tracker.queries[..., 1:], get_points_on_a_grid(3, (64, 64))
        )
        for _ in range(2):
            _, event = self.step(visible=False)
            self.assertFalse(event["keyframe"])
            self.assertEqual(event["added"], [])
            self.assertEqual(event["dropped"], [])
            self.assertEqual(self.tracker.track_ids, list(range(9)))
        self.assertEqual(self.model.online_ind, 3)
        self.assertEqual(self.tracker.session_index, 0)

    def test_keyframe_carries_current_positions_and_replaces_lost_points(self):
        self.step()
        for _ in range(2):
            self.step(visible=False)
        current = get_points_on_a_grid(3, (64, 64))[0].numpy() * 0.9 + 3.2
        _, event = self.step(visible=[True] * 4 + [False] * 5, positions=current)
        self.assertTrue(event["keyframe"])
        self.assertIn("low_survivors", event["keyframe_reason"])
        self.assertEqual(event["carried"], [0, 1, 2, 3])
        self.assertEqual(event["dropped"], [4, 5, 6, 7, 8])
        self.assertEqual(event["added"], [9, 10, 11, 12, 13])
        self.assertEqual(self.tracker.track_ids, [0, 1, 2, 3, 9, 10, 11, 12, 13])
        torch.testing.assert_close(
            self.tracker.queries[0, :4, 1:], torch.tensor(current[:4])
        )
        self.assertEqual(self.tracker.frame_index, 4)
        self.assertEqual(self.model.online_ind, 1)
        self.assertEqual(self.model.coords_buffer.shape[1], 1)

    def test_brief_loss_recovers_without_boundary(self):
        self.step()
        self.step(visible=False)
        self.step()
        _, event = self.step()
        self.assertFalse(event["keyframe"])
        self.assertEqual(self.tracker.track_ids, list(range(9)))
        self.assertEqual(self.model.online_ind, 4)

    def test_coverage_trigger_preserves_visible_ids(self):
        self.step()
        clustered = np.full((9, 2), 20.0, dtype=np.float32)
        for _ in range(3):
            _, event = self.step(positions=clustered)
        self.assertIn("low_coverage", event["keyframe_reason"])
        self.assertNotIn("low_survivors", event["keyframe_reason"])
        self.assertEqual(event["dropped"], [])
        self.assertEqual(event["carried"], list(range(9)))
        self.assertEqual(self.model.online_ind, 1)

    def test_interval_restarts_even_when_no_additions_needed(self):
        self.step()
        for _ in range(7):
            _, event = self.step()
            self.assertFalse(event["keyframe"])
        _, event = self.step()
        self.assertEqual(event["keyframe_reason"], "interval")
        self.assertEqual(event["added"], [])
        self.assertEqual(event["carried"], list(range(9)))
        self.assertEqual(self.model.online_ind, 1)
        self.assertEqual(self.tracker.frame_index, 9)

    def test_total_loss_reseeds_at_boundary_and_never_reuses_ids(self):
        self.step()
        for boundary in (3, 6):
            for _ in range(3):
                _, event = self.step(visible=False)
            self.assertTrue(event["keyframe"])
            self.assertEqual(event["carried"], [])
            self.assertEqual(len(event["dropped"]), 9)
            self.assertEqual(len(event["added"]), 9)
            self.assertEqual(event["capacity_blocked"], 0)
            self.assertEqual(self.tracker.frame_index, boundary + 1)
            self.assertEqual(self.model.online_ind, 1)
        self.assertEqual(self.tracker.track_ids, list(range(18, 27)))

    def test_invalid_configuration_rejected(self):
        for kwargs in [
            {"patience": 0},
            {"survivor_ratio": 0},
            {"min_keyframe_gap": 60, "max_keyframe_gap": 15},
        ]:
            with self.assertRaises(ValueError):
                TrackingController(**kwargs)


if __name__ == "__main__":
    unittest.main()
