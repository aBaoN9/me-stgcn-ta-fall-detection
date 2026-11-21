import ctypes
import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class MonitorStartupTests(unittest.TestCase):
    def run_monitor(self, runtime_exists=False, temperature=60):
        script = Path(__file__).with_name("monitor_windows.py").resolve()
        original_directory = Path.cwd()
        memory_api = SimpleNamespace(kernel32=SimpleNamespace(
            GlobalMemoryStatusEx=lambda state: 1))
        process = SimpleNamespace(poll=lambda: 0, returncode=0)
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                if runtime_exists:
                    Path(".runtime").mkdir()
                with (
                    patch.object(ctypes, "windll", memory_api, create=True),
                    patch("sys.argv", [str(script), "smoke", ".runtime/smoke"]),
                    patch("time.strftime", return_value="tung-gru-ctr-rtx3050-20261121-000000"),
                    patch("time.sleep"),
                    patch("shutil.which", return_value="nvidia-smi"),
                    patch("shutil.disk_usage", return_value=SimpleNamespace(free=2**31)),
                    patch("subprocess.check_output", return_value=f"{temperature}, 0, 0"),
                    patch("subprocess.Popen", return_value=process) as launch,
                    patch("builtins.print"),
                ):
                    with self.assertRaises(SystemExit) as stopped:
                        runpy.run_path(str(script), run_name="__main__")
                self.assertEqual(stopped.exception.code, 1 if temperature >= 87 else 0)
                self.assertEqual(launch.call_count, 0 if temperature >= 87 else 1)
                reports = list(Path(".runtime").glob("*-measurement.json"))
                self.assertEqual(len(reports), 1)
                report = json.loads(reports[0].read_text(encoding="utf-8"))
                self.assertTrue(Path(report["backup"]).is_dir())
                self.assertTrue(Path(report["telemetry"]).is_file())
                self.assertEqual(report["samples"], 1)
                self.assertEqual(report["exit_code"], None if temperature >= 87 else 0)
                self.assertEqual(report["thermal_stop"],
                                 "Reached NVIDIA target temperature (87 C)" if temperature >= 87 else None)
            finally:
                os.chdir(original_directory)

    def test_fresh_checkout_creates_runtime_and_backup(self):
        self.run_monitor()

    def test_existing_runtime_still_starts(self):
        self.run_monitor(runtime_exists=True)

    def test_fresh_checkout_keeps_thermal_stop(self):
        self.run_monitor(temperature=87)


if __name__ == "__main__":
    unittest.main()
