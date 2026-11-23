import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from demo_streamlit.player import player_html
from demo_streamlit.video import MAX_BYTES, analyze_video, events_for_trace, inspect_video, trace_for_track, window_features


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.poses = np.random.default_rng(7).uniform(.1, .9, (120, 33, 4)).astype(np.float32)
        self.poses[:, :, 3] = 1
        self.masks = np.ones(120, dtype=bool)

    def test_no_future_frames_or_interpolation_leakage(self):
        self.masks[85:89] = False
        self.poses[85:89] = np.nan
        first, quality = window_features(self.poses, self.masks, 30, 89, 3)
        self.poses[90:] = 10000
        second, _ = window_features(self.poses, self.masks, 30, 89, 3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (8, 32, 33))
        self.assertTrue(np.isfinite(first).all())
        self.assertAlmostEqual(quality, 86 / 90)

    def test_warmup_and_missing_pose_never_predict_adl(self):
        features, _ = window_features(self.poses, self.masks, 30, 88, 3)
        self.assertIsNone(features)
        self.masks[89] = False
        features, _ = window_features(self.poses, self.masks, 30, 89, 3)
        self.assertIsNone(features)
        self.masks[:100] = False
        features, quality = window_features(self.poses, self.masks, 30, 119, 3)
        self.assertIsNone(features)
        self.assertLess(quality, .5)
        calls = []
        rows = trace_for_track(self.poses, np.zeros(120, dtype=bool), 30, 3, lambda features: calls.append(features))
        self.assertFalse(calls)
        self.assertTrue(all(row["score"] is None for row in rows))

    def test_exact_window_boundary_and_no_future_endpoints(self):
        rows = trace_for_track(self.poses, self.masks, 30, 3, lambda features: .75)
        self.assertTrue(all(row["score"] is None for row in rows if row["time"] < 89/30))
        self.assertEqual(rows[-1]["time"], 119/30)
        self.assertEqual(rows[-1]["score"], .75)
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            trace_for_track(self.poses, self.masks, 30, 3, lambda features: float("nan"))

    def test_event_gaps_are_not_bridged(self):
        rows = [{"time": 0, "score": .8}, {"time": 1, "score": None},
                {"time": 2, "score": .5}, {"time": 3, "score": .1}]
        self.assertEqual(events_for_trace(rows, 4), [{"start": 0, "end": 1, "peak": .8},
                                                   {"start": 2, "end": 3, "peak": .5}])

    def test_size_and_invalid_media_fail_before_inference(self):
        with self.assertRaises(ValueError):
            analyze_video(b"", None, {})
        with patch("demo_streamlit.video.MAX_BYTES", 1):
            with self.assertRaises(ValueError):
                analyze_video(b"xx", None, {})
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "invalid.mp4"
            source.write_bytes(b"not a video")
            with self.assertRaises(ValueError):
                inspect_video(source)
        self.assertEqual(MAX_BYTES, 50 * 1024 * 1024)

    def test_player_payload_cannot_end_script_or_inject_html(self):
        malicious = '</script><img src=x onerror="alert(1)">'
        markup = player_html(model_name=malicious)
        self.assertNotIn(malicious, markup)
        self.assertIn(r"\u003c/script\u003e", markup)
        self.assertNotIn("__PAYLOAD__", markup)
        self.assertIn("video.currentTime", markup)


if __name__ == "__main__":
    unittest.main()
