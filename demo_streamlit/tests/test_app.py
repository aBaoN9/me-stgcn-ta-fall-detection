import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

from demo_streamlit.data import BASELINE_NAMES, MODEL_CHOICES, MODEL_NAMES, ROOT
from demo_streamlit.history import HistoryStore
from demo_streamlit.tests.test_features import sample_result


class AppTests(unittest.TestCase):
    def test_opt_in_video_history_reopens_player_without_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = HistoryStore(Path(temporary) / "test.sqlite3")
            with patch("demo_streamlit.history.HistoryStore", return_value=store), \
                    patch("demo_streamlit.jobs.load_checkpoint_model") as loader:
                app = AppTest.from_file(str(ROOT / "demo_streamlit/app.py"), default_timeout=60).run()
                app.session_state["analysis"] = (b"preview", sample_result())
                app.run()
                app.checkbox(key="include_video").check().run()
                app.button(key="save_current").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(store.entries()[0]["has_video"])
                app.button(key="open_history").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(app.session_state["history_view"][1][0], b"preview")
                loader.assert_not_called()

    def test_report_save_open_and_delete_use_explicit_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = HistoryStore(Path(temporary) / "test.sqlite3")
            with patch("demo_streamlit.history.HistoryStore", return_value=store):
                app = AppTest.from_file(str(ROOT / "demo_streamlit/app.py"), default_timeout=60).run()
                app.session_state["analysis"] = (b"preview", sample_result())
                app.run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(store.entries(), [])
                self.assertFalse(app.checkbox(key="include_video").value)
                app.button(key="save_current").click().run()
                self.assertEqual(len(app.exception), 0)
                entries = store.entries()
                self.assertEqual(len(entries), 1)
                self.assertFalse(entries[0]["has_video"])
                app.button(key="open_history").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertIn("history_view", app.session_state)
                self.assertTrue(app.button(key="delete_history").disabled)
                app.checkbox(key=f'delete_{entries[0]["id"]}').check().run()
                app.button(key="delete_history").click().run()
                self.assertEqual(len(app.exception), 0)
                self.assertEqual(store.entries(), [])
                self.assertNotIn("history_view", app.session_state)

    def test_cancel_running_job_and_clear_result_on_model_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("demo_streamlit.history.HistoryStore", return_value=HistoryStore(Path(temporary) / "test.sqlite3")):
                app = AppTest.from_file(str(ROOT / "demo_streamlit/app.py"), default_timeout=60).run()
                job = Mock()
                job.snapshot.return_value = ("running", .3, "Processing")
                app.session_state["job"] = job
                app.run()
                self.assertEqual(len(app.exception), 0)
                app.button(key="cancel_job").click().run()
                job.cancel.assert_called_once()
                app.selectbox(key="variant").select("TCN_v2_small").run()
                self.assertEqual(job.cancel.call_count, 2)
                self.assertNotIn("analysis", app.session_state)

    @patch("demo_streamlit.history.HistoryStore")
    def test_model_only_interface_switches_and_reset(self, store_type):
        store_type.return_value.entries.return_value = []
        app = AppTest.from_file(str(ROOT / "demo_streamlit/app.py"), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(app.button[0].disabled)
        self.assertEqual([selector.key for selector in app.selectbox], ["variant", "seed", "subject"])
        self.assertEqual(app.selectbox(key="variant").options, list(MODEL_CHOICES.values()))
        self.assertFalse(app.title)
        self.assertFalse(any("Nhìn chuyển động" in block.value or "Video bên trái" in block.value
                             for block in app.markdown))
        expected = {"A3_motion_attention": "89.38%", "TCN_v2_small": "91.04%",
                    "GRU_v2_small": "89.58%", "CTR_GCN_adapted": "88.54%",
                    "SkateFormer_adapted_small": "84.17%"}
        self.assertEqual(set(MODEL_NAMES), set(expected))
        for variant, accuracy in expected.items():
            app.selectbox(key="variant").select(variant).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.metric[0].value, accuracy)
        for baseline in BASELINE_NAMES:
            app.selectbox(key="variant").select(baseline).run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.button(key="analyze").disabled)
            self.assertTrue(any("chưa có checkpoint" in message.value for message in app.info))
            self.assertFalse(app.metric)
        app.session_state["analysis"] = "obsolete"
        app.selectbox(key="seed").select(142).run()
        self.assertNotIn("analysis", app.session_state)
        self.assertTrue(app.button[0].disabled)
        app.button(key="reset").click().run()
        self.assertEqual(app.session_state["upload_generation"], 1)
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
