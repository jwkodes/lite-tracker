import copy
import unittest

import torch

from src.lite_tracker import LiteTracker
from src.managed_lite_tracker import ManagedLiteTracker


class ManagedLiteTrackerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)
        self.model = LiteTracker(model_resolution=(64, 64)).eval()
        self.tracker = ManagedLiteTracker(self.model, max_points=4)
        self.frame = torch.rand(1, 3, 64, 64) * 255
        self.queries = torch.tensor(
            [[[0.0, 16.0, 16.0], [0.0, 32.0, 32.0], [0.0, 48.0, 48.0]]]
        )

    def caches(self):
        return [
            getattr(self.model, n) for n in self.tracker._CACHES
        ] + self.model.track_feat_cache

    def test_addition_preserves_existing_state_and_bounds_history(self):
        self.tracker(self.frame, self.queries)
        self.tracker(self.frame)
        before = [v.clone() for v in self.caches()]
        self.assertEqual(
            self.tracker.add_queries(torch.tensor([[[2.0, 24.0, 24.0]]])), [3]
        )
        for old, new in zip(before, self.caches()):
            torch.testing.assert_close(new[:, :, :3], old, rtol=0, atol=0)
        coords, vis, _ = self.tracker(self.frame)
        torch.testing.assert_close(coords[0, 0, 3], torch.tensor([24.0, 24.0]))
        self.assertTrue(vis[0, 0, 3])
        for _ in range(20):
            coords, _, conf = self.tracker(self.frame)
            self.assertTrue(torch.isfinite(coords).all())
            self.assertTrue(torch.isfinite(conf).all())
            self.assertTrue(all(v.shape[2] == 4 for v in self.caches()))
        self.assertEqual(self.tracker.track_ids, [0, 1, 2, 3])
        self.assertEqual(self.model.coords_buffer.shape[1], 15)
        self.assertEqual(self.model.online_ind, 23)

    def test_invisible_queries_persist_and_can_recover(self):
        original = self.model.forward

        def visibility(frame, queries):
            step = self.model.online_ind
            coords, vis, conf = original(frame, queries)
            return coords, torch.full_like(vis, step >= 20), conf

        self.model.forward = visibility
        for i in range(21):
            _, visible, _ = self.tracker(self.frame, self.queries if i == 0 else None)
            self.assertEqual(self.tracker.track_ids, [0, 1, 2])
            self.assertEqual(visible.tolist(), [[[i >= 20] * 3]])
        self.assertFalse(hasattr(self.tracker, "remove_tracks"))

    def test_future_queries_are_hidden_until_their_start_frame(self):
        queries = self.queries.clone()
        queries[0, 2, 0] = 3
        original = self.model.forward

        def all_visible(frame, queries):
            coords, vis, conf = original(frame, queries)
            return coords, torch.ones_like(vis), conf

        self.model.forward = all_visible
        for i in range(4):
            _, visible, _ = self.tracker(self.frame, queries if i == 0 else None)
            self.assertEqual(visible[0, 0, 2].item(), i >= 3)

    def test_initial_empty_gap_then_addition(self):
        for _ in range(3):
            coords, vis, conf = self.tracker(self.frame)
            self.assertEqual(coords.shape, (1, 1, 0, 2))
            self.assertEqual(vis.shape, (1, 1, 0))
            self.assertEqual(conf.shape, (1, 1, 0))
        self.assertEqual(self.model.online_ind, 0)
        self.assertEqual(
            self.tracker.add_queries(torch.tensor([[[3.0, 24.0, 24.0]]])), [0]
        )
        coords, vis, _ = self.tracker(self.frame)
        torch.testing.assert_close(coords[0, 0, 0], torch.tensor([24.0, 24.0]))
        self.assertTrue(vis.all())
        self.assertEqual(self.tracker.frame_index, 4)
        self.assertEqual(self.model.online_ind, 1)

    def test_validation_is_atomic(self):
        self.tracker(self.frame, self.queries)
        before = [v.clone() for v in self.caches()]
        invalid = [
            torch.tensor([[[0.0, 2.0, 2.0]]]),
            torch.tensor([[[1.5, 2.0, 2.0]]]),
            torch.tensor([[[1.0, float("nan"), 2.0]]]),
            torch.tensor([[[1.0, 64.0, 2.0]]]),
            torch.tensor([[[1.0, 2.0, 2.0], [1.0, 3.0, 3.0]]]),
        ]
        for value in invalid:
            with self.assertRaises(ValueError):
                self.tracker.add_queries(value)
        with self.assertRaises(ValueError):
            self.tracker(self.frame, self.queries[:, :2])
        with self.assertRaises(ValueError):
            self.tracker(torch.zeros(2, 3, 64, 64))
        with self.assertRaises(ValueError):
            self.tracker(torch.zeros(1, 3, 32, 64))
        self.assertEqual(self.tracker.track_ids, [0, 1, 2])
        self.assertEqual(self.model.online_ind, 1)
        for old, new in zip(before, self.caches()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)

    def test_insertion_matches_native_scheduled_queries(self):
        reference = copy.deepcopy(self.model)
        queries = self.queries.clone()
        queries[0, 2, 0] = 3
        for step in range(8):
            if step == 3:
                self.tracker.add_queries(queries[:, 2:])
            actual = self.tracker(self.frame, queries[:, :2] if step == 0 else None)
            with torch.no_grad():
                expected = reference(self.frame, queries)
            count = 2 if step < 3 else 3
            for got, want in zip(actual, expected):
                torch.testing.assert_close(
                    got, want[:, :, :count], atol=1e-4, rtol=1e-4
                )

    def test_keyframe_restarts_memory_preserves_ids_and_absolute_clock(self):
        reference = copy.deepcopy(self.model)
        original = self.model.forward

        def known_boundary(frame, queries):
            step = self.model.online_ind
            outputs = original(frame, queries)
            if step == 2:
                coords = self.queries[:, None, :, 1:] + 2
                return coords, torch.tensor([[[True, False, True]]]), outputs[2]
            return outputs

        self.model.forward = known_boundary
        for i in range(3):
            self.tracker(self.frame, self.queries if i == 0 else None)
        outputs = self.tracker.start_keyframe([2, 0], torch.tensor([[[24.0, 24.0]]]))
        self.assertEqual(self.tracker.track_ids, [2, 0, 3])
        self.assertEqual(self.tracker.frame_index, 3)
        self.assertEqual(self.tracker.session_index, 1)
        self.assertEqual(self.model.online_ind, 1)
        self.assertEqual(self.model.coords_buffer.shape[1], 1)
        torch.testing.assert_close(
            self.tracker.queries[..., 0], torch.full((1, 3), 2.0)
        )
        local_queries = self.tracker.queries.clone()
        local_queries[..., 0] = 0
        with torch.no_grad():
            expected = reference(self.frame, local_queries)
        for actual, wanted in zip(outputs, expected):
            torch.testing.assert_close(actual, wanted)
        # Next input is frame 3, while the backend processes local frame 1.
        actual = self.tracker(self.frame)
        with torch.no_grad():
            expected = reference(self.frame, local_queries)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want)
        self.assertEqual(self.tracker.frame_index, 4)
        self.assertEqual(self.model.online_ind, 2)
        self.assertEqual(
            self.tracker.add_queries(torch.tensor([[[4.0, 8.0, 8.0]]])), [4]
        )

    def test_keyframe_validation_preserves_current_session(self):
        with self.assertRaises(ValueError):
            self.tracker.start_keyframe([])
        original = self.model.forward

        def last_invisible(frame, queries):
            coords, vis, conf = original(frame, queries)
            vis[..., -1] = False
            return coords, vis, conf

        self.model.forward = last_invisible
        self.tracker(self.frame, self.queries)
        before = [v.clone() for v in self.caches()]
        invalid = [
            ([99], None),
            ([0, 0], None),
            ([2], None),
            ([0], torch.tensor([[[64.0, 1.0]]])),
            ([0], torch.tensor([[[float("nan"), 1.0]]])),
            ([0, 1], torch.zeros(1, 3, 2)),
        ]
        for ids, points in invalid:
            with self.assertRaises(ValueError):
                self.tracker.start_keyframe(ids, points)
        self.assertEqual(self.tracker.track_ids, [0, 1, 2])
        self.assertEqual(self.tracker.session_index, 0)
        for old, new in zip(before, self.caches()):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
        self.tracker.start_keyframe([0])
        with self.assertRaises(ValueError):
            self.tracker.start_keyframe([0])

    def test_empty_keyframe_and_future_query_clock(self):
        self.tracker(self.frame, self.queries)
        self.tracker(self.frame)
        outputs = self.tracker.start_keyframe([])
        self.assertEqual(outputs[0].shape[2], 0)
        self.assertEqual(self.tracker.frame_index, 2)
        self.assertEqual(self.model.online_ind, 0)
        self.assertEqual(
            self.tracker.add_queries(torch.tensor([[[3.0, 24.0, 24.0]]])), [3]
        )
        _, visible, _ = self.tracker(self.frame)  # Frame 2; query not born yet.
        self.assertFalse(visible.any())
        coords, vis, _ = self.tracker(self.frame)  # Birth at absolute frame 3.
        torch.testing.assert_close(coords[0, 0, 0], torch.tensor([24.0, 24.0]))
        self.assertTrue(vis.all())
        self.assertEqual(self.tracker.frame_index, 4)
        self.assertEqual(self.model.online_ind, 2)

    def test_pending_additions_cannot_be_silently_lost_at_boundary(self):
        self.tracker(self.frame, self.queries)
        self.tracker.add_queries(torch.tensor([[[1.0, 24.0, 24.0]]]))
        with self.assertRaises(ValueError):
            self.tracker.start_keyframe([0])
        self.assertEqual(self.tracker.track_ids, [0, 1, 2, 3])

    def test_reset_begins_a_new_video(self):
        self.tracker(self.frame, self.queries)
        self.tracker.reset()
        self.assertEqual(self.tracker.frame_index, 0)
        self.assertEqual(self.model.online_ind, 0)
        self.assertEqual(self.tracker.track_ids, [])
        self.tracker(self.frame, self.queries)
        self.assertEqual(self.tracker.track_ids, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
