import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from training.component_ablation.features import make_features_v2
from training.component_ablation.models import SkeletonModel
from training.component_ablation.training import metrics

from .models import A3, TCN, build_model, flatten_joints
from . import runner
from .runner import load_verified, run, splits


def fixture():
    rng = np.random.default_rng(42)
    return dict(poses=rng.random((16, 32, 33, 4), dtype=np.float32),
                masks=np.ones((16, 32), dtype=bool), durations=np.ones(16),
                labels=np.tile([0, 1], 8), subjects=np.repeat([1, 2, 3, 4], 4),
                video_ids=np.array([f"synthetic-{i}" for i in range(16)]))


class BenchmarkTests(unittest.TestCase):
    def test_feature_order_shape_and_mask(self):
        values = torch.arange(8 * 32 * 33).reshape(1, 8, 32, 33)
        flat = flatten_joints(values)
        for joint in range(33):
            for channel in range(8):
                torch.testing.assert_close(flat[0, joint * 8 + channel], values[0, channel, :, joint])
        with self.assertRaises(ValueError):
            flatten_joints(values.transpose(2, 3))
        data = fixture()
        data["masks"][0, 5] = False
        feature = make_features_v2(data["poses"][0], data["masks"][0], 1.0)
        self.assertEqual(feature.shape, (8, 32, 33))
        self.assertTrue((feature[2, 5] == 0).all())
        self.assertTrue(np.isfinite(feature).all())
        self.assertEqual(build_model(TCN).eval()(torch.from_numpy(feature[None])).shape, (1, 2))

    def test_temporal_stack_is_causal(self):
        model = build_model(TCN).eval()
        x = torch.randn(2, 264, 32)
        altered = x.clone()
        altered[:, :, 21:] += 10
        with torch.no_grad():
            torch.testing.assert_close(model.blocks(x)[:, :, :21], model.blocks(altered)[:, :, :21])

    def test_a3_matches_original(self):
        torch.manual_seed(42)
        reference = SkeletonModel(8, True, 0.3).eval()
        torch.manual_seed(42)
        actual = build_model(A3).eval()
        x = torch.randn(2, 8, 32, 33)
        with torch.no_grad():
            for expected, observed in zip(reference(x), actual(x)):
                torch.testing.assert_close(expected, observed, rtol=0, atol=0)

    def test_hash_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_verified(folder)
            (Path(folder) / "gmdcsa24_pose_32.npz").write_bytes(b"bad input")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_verified(folder)

    def test_nested_subjects(self):
        data = fixture()
        all_test = []
        for outer in range(1, 5):
            train, test, inner = splits(data, outer)
            all_test.extend(test)
            self.assertEqual(set(data["subjects"][test]), {outer})
            self.assertFalse(set(train) & set(test))
            self.assertEqual(sorted(i for _, _, val in inner for i in val), sorted(train))
            for person, fit, val in inner:
                self.assertEqual(set(data["subjects"][val]), {person})
                self.assertEqual(set(data["subjects"][fit]), {1, 2, 3, 4} - {outer, person})
        self.assertEqual(sorted(all_test), list(range(16)))
        data["labels"][data["subjects"] == 1] = 0
        with self.assertRaisesRegex(ValueError, "Missing class"):
            splits(data, 1)

    def test_metrics(self):
        result = metrics([1, 1, 1, 0, 0], [1, 0, 0, 0, 1])
        self.assertEqual([result[k] for k in ("tp", "tn", "fp", "fn")], [1, 1, 1, 2])
        self.assertAlmostEqual(result["accuracy"], 0.4)
        self.assertAlmostEqual(result["f1"], 0.4)
        self.assertIsNone(metrics([0, 1], [0, 0])["precision"])

    def test_synthetic_pipeline_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "synthetic"
            state = run(fixture(), "smoke", torch.device("cpu"), output, synthetic=True)
            self.assertEqual(state["actual_fits"], 8)
            self.assertEqual(state["reload"], "PASS")
            self.assertFalse(state["scientific_result"])
            self.assertTrue(state["synthetic"])
            self.assertEqual(len(list((output / "folds").glob("*/model.pt"))), 2)
            self.assertEqual(json.loads((output / "completion.json").read_text()), state)
            self.assertEqual(len((output / "predictions.csv").read_text().splitlines()), 25)
            with self.assertRaisesRegex(ValueError, "limited"):
                run(fixture(), "full", torch.device("cpu"), output, synthetic=True)

    def test_a3_fit_and_scores_match_original(self):
        from training.component_ablation import training as original
        data = fixture()
        _, _, inner = splits(data, 1)
        _, train, val = inner[0]
        cfg = original.default_config("smoke")
        device = torch.device("cpu")
        reference, epoch, history = original.fit_model(
            data, train, val, original.VARIANTS[A3], cfg, 1062, device)
        actual, actual_epoch, actual_history = runner.fit(A3, data, train, val, cfg, 1062, device)
        self.assertEqual((actual_epoch, actual_history), (epoch, history))
        loader = original.make_loader(data, val, original.VARIANTS[A3], cfg, False, 1062)
        labels, scores, _ = original.predict(reference, loader, device)
        observed = runner.scores_for(actual, data, val, cfg, device)
        np.testing.assert_array_equal(observed, scores)
        self.assertEqual(original.select_thresholds(labels, scores, cfg),
                         original.select_thresholds(data["labels"][val], observed, cfg))

    def test_interruption_resume_corruption_and_incompatible_lock(self):
        data, device = fixture(), torch.device("cpu")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "run"
            original_fold = runner.run_fold

            def interrupt_second(variant, *args):
                if variant == TCN:
                    (args[-2] / "model.pt").write_bytes(b"partial")
                    raise InterruptedError("simulated session interruption")
                return original_fold(variant, *args)

            with patch.object(runner, "run_fold", side_effect=interrupt_second):
                with self.assertRaises(InterruptedError):
                    run(data, "smoke", device, output, synthetic=True)
            a3 = output / "folds" / f"{A3}_seed42_subject1" / "model.pt"
            original_hash = runner.sha256(a3)
            stale = output / "folds" / ".pending-old"
            stale.mkdir()
            (stale / "partial").write_bytes(b"interrupted")
            lock_file = output / "locked_protocol.json"
            lock = runner.read_json(lock_file)
            for key, value in (("code", {}), ("config", {}), ("device", "cuda")):
                runner.original.write_json(lock_file, dict(lock, **{key: value}))
                with self.assertRaisesRegex(ValueError, "Resume.*mismatch"):
                    run(data, "smoke", device, output, synthetic=True, resume=True)
            runner.original.write_json(lock_file, lock)
            with patch.object(runner, "run_fold", wraps=original_fold) as executed:
                done = run(data, "smoke", device, output, synthetic=True, resume=True)
                self.assertEqual([call.args[0] for call in executed.call_args_list], [TCN])
            self.assertEqual(runner.sha256(a3), original_hash)
            self.assertFalse(done["scientific_result"])
            self.assertEqual(run(data, "smoke", device, output, synthetic=True, resume=True), done)
            missing = output / "folds" / f"{TCN}_seed42_subject1"
            moved = output / "missing"
            missing.rename(moved)
            with self.assertRaisesRegex(ValueError, "missing seed/fold"):
                run(data, "smoke", device, output, synthetic=True, resume=True)
            moved.rename(missing)
            extra = output / "folds" / f"{TCN}_seed142_subject1"
            extra.mkdir()
            with self.assertRaisesRegex(ValueError, "Unexpected"):
                run(data, "smoke", device, output, synthetic=True, resume=True)
            extra.rmdir()
            a3.write_bytes(b"corrupt checkpoint")
            with self.assertRaisesRegex(ValueError, "Checkpoint SHA-256 mismatch"):
                run(data, "smoke", device, output, synthetic=True, resume=True)

    def test_full_gate_and_smoke_cross_checks(self):
        data, device = fixture(), torch.device("cpu")
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "synthetic"
            target = Path(folder) / "full"
            run(data, "smoke", device, output, synthetic=True)
            with self.assertRaisesRegex(ValueError, "verified real inputs"):
                run(data, "full", device, target, smoke_evidence=output)
            self.assertFalse(target.exists())
            with self.assertRaisesRegex(ValueError, "matching CUDA"):
                runner.verify_smoke(data, device, output)
            # Exercise the gate without a CUDA device or verified real inputs.
            fake_hashes = {"pose": "fixture"}
            data["provenance"] = {"pose": {"sha256": "fixture"}}
            cuda = torch.device("cuda")
            with patch.object(runner, "expected_inputs", return_value=fake_hashes), patch.object(
                torch.cuda, "get_device_name", return_value="mock GPU"
            ):
                with self.assertRaisesRegex(ValueError, "Full requires completed"):
                    run(data, "full", cuda, target)
                with self.assertRaisesRegex(ValueError, "Smoke data/code/config/runtime mismatch"):
                    run(data, "full", cuda, target, smoke_evidence=output)
            self.assertFalse(target.exists())
            del data["provenance"]
            lock = runner.read_json(output / "locked_protocol.json")
            verified = runner.collect(data, lock, device, output)
            record = output / "folds" / f"{TCN}_seed42_subject1" / "fold.json"
            saved_record = runner.read_json(record)
            for mutation, error in (
                (dict(saved_record, inner=saved_record["inner"][:-1]), "inner-fit count"),
                (dict(saved_record, predictions=saved_record["predictions"][:-1]), "coverage"),
                (dict(saved_record, metadata=dict(saved_record["metadata"], seed=142)), "metadata"),
            ):
                runner.original.write_json(record, mutation)
                with self.assertRaisesRegex(ValueError, error):
                    runner.collect(data, lock, device, output)
            runner.original.write_json(record, saved_record)
            done_file = output / "completion.json"
            done = runner.read_json(done_file)
            runner.original.write_json(done_file, dict(done, actual_fits=7))
            with self.assertRaisesRegex(ValueError, "Completion metadata"):
                runner.validate_summary(data, lock, output, verified)
            runner.original.write_json(done_file, done)
            predictions = output / "predictions.csv"
            predictions.write_text(predictions.read_text().splitlines()[0] + "\n", encoding="utf-8")
            with self.assertRaises(AssertionError):
                runner.validate_summary(data, lock, output, verified)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
