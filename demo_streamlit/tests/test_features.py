import copy
import csv
import io
import json
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from contextlib import closing
from unittest.mock import Mock, patch

import numpy as np

from demo_streamlit.history import HistoryStore
from demo_streamlit.jobs import AnalysisJob
from demo_streamlit.reporting import readable_report, report_data, summary, trace_csv
from demo_streamlit.video import AnalysisCancelled, preview_video, trace_for_track


def sample_result():
    return {"info": {"duration": 4, "fps": 30, "frames": 120, "width": 320, "height": 240},
            "rows": [{"time": 0, "score": None, "quality": 1, "model_ms": None, "warmup": True},
                     {"time": 3, "score": .8, "quality": 1, "model_ms": 2, "warmup": False},
                     {"time": 3.5, "score": None, "quality": 0, "model_ms": None, "warmup": False}],
            "events": [{"start": 3, "end": 3.5, "peak": .8, "thumbnail": "private-image"}],
            "identity": {"variant": "A3_motion_attention", "seed": 42, "subject": 1, "sha256": "abc"},
            "poses": [None] * 120, "angles": [None] * 120, "processing_seconds": 5,
            "pose_seconds": 3, "pipeline": "causal-window-v1", "pose_asset_sha256": "def",
            "window_seconds": 3, "valid_windows": 1, "video_name": "example.mp4", "video_sha256": "video-hash"}


class ReportTests(unittest.TestCase):
    def test_summary_separates_warmup_gaps_and_events(self):
        totals = summary(sample_result())
        self.assertEqual(totals["warmup_seconds"], 3)
        self.assertEqual(totals["unavailable_seconds"], .5)
        self.assertEqual(totals["high_seconds"], .5)
        self.assertEqual(totals["valid_windows"], 1)
        self.assertEqual(totals["events"], 1)

    def test_exports_strip_images_and_escape_untrusted_names(self):
        result = sample_result()
        result["video_name"] = '<script>alert("test")</script>'
        original = copy.deepcopy(result)
        report = report_data(result)
        self.assertNotIn("poses", report)
        self.assertNotIn("angles", report)
        self.assertNotIn("thumbnail", report["events"][0])
        markup = readable_report(result, '<img src=x onerror="bad">').decode("utf-8")
        self.assertNotIn("<script>", markup)
        self.assertNotIn("<img", markup)
        self.assertNotIn("private-image", markup)
        self.assertIn("&lt;script&gt;", markup)
        records = list(csv.DictReader(io.StringIO(trace_csv(result).decode("utf-8-sig"))))
        self.assertEqual(records[0]["score"], "")
        self.assertEqual(records[1]["score"], "0.8")
        self.assertEqual(result, original)


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = HistoryStore(Path(self.temporary.name) / "history.sqlite3")

    def tearDown(self):
        self.temporary.cleanup()

    def test_no_disk_write_until_explicit_save(self):
        self.assertEqual(self.store.entries(), [])
        self.assertFalse(self.store.path.exists())

    def test_report_only_default_and_opt_in_video(self):
        result = sample_result()
        identity = self.store.save(result, "ME-STGCN-TA · A3", b"private-video")
        preview, stored, model = self.store.load(identity)
        self.assertIsNone(preview)
        self.assertNotIn("poses", stored)
        self.assertNotIn("thumbnail", stored["events"][0])
        self.assertEqual(model, "ME-STGCN-TA · A3")
        with closing(sqlite3.connect(self.store.path)) as connection:
            serialized, media = connection.execute("SELECT result, preview FROM sessions").fetchone()
        self.assertNotIn("private-image", serialized)
        self.assertIsNone(media)
        with_video = self.store.save(result, "ME-STGCN-TA · A3", b"private-video", True)
        self.assertEqual(self.store.load(with_video)[:2], (b"private-video", result))
        self.assertEqual(len(self.store.entries()), 2)
        self.store.delete(identity)
        self.assertEqual(len(self.store.entries()), 1)
        with self.assertRaises(ValueError):
            self.store.load(identity)
        self.assertEqual(self.store.load(with_video)[0], b"private-video")

    def test_storage_bound_and_path_traversal(self):
        with patch("demo_streamlit.history.MAX_ENTRIES", 1):
            self.store.save(sample_result(), "Model")
            with self.assertRaisesRegex(ValueError, "20 phiên"):
                self.store.save(sample_result(), "Model")
        with self.assertRaises(ValueError):
            self.store.load("../history.sqlite3")
        with self.assertRaises(ValueError):
            self.store.save(sample_result(), "Model", include_video=True)

    def test_tampered_saved_result_is_rejected(self):
        identity = self.store.save(sample_result(), "Model")
        with closing(sqlite3.connect(self.store.path)) as connection, connection:
            connection.execute("UPDATE sessions SET result=? WHERE id=?", (json.dumps({}), identity))
        with self.assertRaisesRegex(ValueError, "toàn vẹn"):
            self.store.load(identity)


class JobTests(unittest.TestCase):
    signature = ("video-hash", "accepted", "A3_motion_attention", 42, 1, 3, "causal-window-v1")

    def test_completes_without_streamlit_api_or_automatic_persistence(self):
        def analyzer(video, model, identity, window, progress, check):
            progress(.5, "Processing")
            check()
            return b"preview", sample_result()

        job = AnalysisJob(b"input", self.signature, "clip.mp4", lambda *args: (None, {}), analyzer)
        job._thread.join(3)
        self.assertEqual(job.snapshot()[0], "completed")
        result = job.take_result()
        self.assertEqual(result[1]["video_sha256"], "video-hash")
        self.assertEqual(result[1]["video_name"], "clip.mp4")
        self.assertIsNone(job.take_result())

    def test_cancel_drops_partial_output_and_finishes(self):
        entered, release = threading.Event(), threading.Event()

        def analyzer(video, model, identity, window, progress, check):
            entered.set()
            release.wait(3)
            check()
            return b"wrong", sample_result()

        job = AnalysisJob(b"input", self.signature, "clip", lambda *args: (None, {}), analyzer)
        self.assertTrue(entered.wait(3))
        job.cancel()
        release.set()
        job._thread.join(3)
        self.assertEqual(job.snapshot()[0], "cancelled")
        self.assertIsNone(job.take_result())

    def test_cancel_completed_job_discards_result(self):
        job = AnalysisJob(b"input", self.signature, "clip", lambda *args: (None, {}),
                          lambda *args: (b"preview", sample_result()))
        job._thread.join(3)
        job.cancel()
        self.assertEqual(job.snapshot()[0], "cancelled")
        self.assertIsNone(job.take_result())

    def test_failed_loader_does_not_publish_results(self):
        def loader(*args):
            raise ValueError("Checkpoint mismatch")

        job = AnalysisJob(b"input", self.signature, "clip", loader)
        job._thread.join(3)
        self.assertEqual(job.snapshot()[0], "failed")
        self.assertIsNone(job.take_result())

    def test_deadline_stops_at_next_cancellation_boundary(self):
        entered, release = threading.Event(), threading.Event()

        def loader(*args):
            entered.set()
            release.wait(3)
            return None, {}

        analyzer = Mock()
        job = AnalysisJob(b"input", self.signature, "clip", loader, analyzer)
        self.assertTrue(entered.wait(3))
        job._started -= 241
        release.set()
        job._thread.join(3)
        self.assertEqual(job.snapshot()[0], "cancelled")
        self.assertIn("4 phút", job.snapshot()[2])
        analyzer.assert_not_called()

    def test_cancellation_reaches_window_loop_and_ffmpeg_cleanup(self):
        check = Mock(side_effect=AnalysisCancelled("stop"))
        with self.assertRaises(AnalysisCancelled):
            trace_for_track(np.zeros((120, 33, 4)), np.ones(120, dtype=bool), 30, 3, Mock(), check)
        process = Mock()
        process.poll.return_value = None
        with patch("demo_streamlit.video.subprocess.Popen", return_value=process), \
                patch("imageio_ffmpeg.get_ffmpeg_exe", return_value="ffmpeg"):
            with self.assertRaises(AnalysisCancelled):
                preview_video(Path("input"), Path("output"), check)
        process.kill.assert_called_once()
        process.wait.assert_called_once()


if __name__ == "__main__":
    unittest.main()
