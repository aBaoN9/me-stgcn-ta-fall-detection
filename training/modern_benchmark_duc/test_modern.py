"""Tests for Duc's modern benchmark. Positive and negative checks required by GB-20261124-001 section 9.

    python -B -m training.modern_benchmark_duc.test_modern --report .runtime/modern_benchmark_duc/tests.json

Synthetic fixtures are used for software checks only; they never reach a full run.
"""
import argparse
import copy
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from training.component_ablation import training as original
from training.component_ablation.models import make_loader
from training.modern_benchmark_duc import runner
from training.modern_benchmark_duc.graph import SKATE_PARTS, hop_matrix, shortest_paths, skate_layout, spatial_graph
from training.modern_benchmark_duc.models import (A3, ASSIGNED, BLOCKGCN, DEGCN, DS_GCN, JT_GRAPHFORMER, MODEL_IDS,
                                                  SKATEFORMER, BlockedModel, build_model, logits_of)

RUNNABLE = [A3, SKATEFORMER, BLOCKGCN, DEGCN, JT_GRAPHFORMER]
ROOT = Path(__file__).resolve().parents[2]


def synthetic_data(per_subject=8, seed=0):
    rng = np.random.default_rng(seed)
    count = 4 * per_subject
    poses = rng.uniform(0.3, 0.7, (count, 32, 33, 4)).astype(np.float32)
    poses[..., 3] = rng.uniform(0.2, 1.0, (count, 32, 33))
    labels = np.tile(np.repeat([0, 1], per_subject // 2), 4)
    poses[labels == 1, 16:, :, 1] += 0.2  # falls drop in y in the second half
    masks = np.ones((count, 32), dtype=bool)
    masks[3, 5:9] = False
    poses[3, 5:9] = 0
    return dict(poses=poses, masks=masks, durations=np.full(count, 2.0, np.float32), labels=labels,
                subjects=np.repeat([1, 2, 3, 4], per_subject),
                video_ids=np.asarray([f"SYN_{i:03d}" for i in range(count)]), synthetic=True)


def batch(size, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(size, 8, 32, 33, generator=generator), torch.randint(0, 2, (size,), generator=generator)


class ModelContract(unittest.TestCase):
    def test_shape_finite_backward_batch_1_and_16(self):
        for model_id in RUNNABLE:
            for size in (1, 16):
                with self.subTest(model=model_id, batch=size):
                    torch.manual_seed(0)
                    model = build_model(model_id)
                    x, y = batch(size)
                    if model_id == BLOCKGCN and size == 1:
                        # Author TopoTrans uses BatchNorm1d on (B, C): undefined for one sample in train mode.
                        model.train()
                        with self.assertRaises(ValueError):
                            logits_of(model, x)
                        model.eval()
                        with torch.no_grad():
                            out = logits_of(model, x)
                        self.assertEqual(tuple(out.shape), (1, 2))
                        continue
                    model.train()
                    out = logits_of(model, x)
                    self.assertEqual(tuple(out.shape), (size, 2))
                    self.assertTrue(torch.isfinite(out).all())
                    torch.nn.functional.cross_entropy(out, y).backward()
                    grads = [p.grad for p in model.parameters() if p.requires_grad]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
                    model.eval()
                    with torch.no_grad():
                        self.assertTrue(torch.isfinite(logits_of(model, x)).all())

    def test_eval_outputs_do_not_depend_on_other_clips(self):
        for model_id in RUNNABLE:
            with self.subTest(model=model_id):
                torch.manual_seed(0)
                model = build_model(model_id).eval()
                x, _ = batch(8)
                changed = x.clone()
                changed[4:] = torch.randn_like(changed[4:])
                with torch.no_grad():
                    torch.testing.assert_close(logits_of(model, x)[:4], logits_of(model, changed)[:4], rtol=1e-5, atol=1e-5)

    def test_missing_pose_visibility_and_dropped_joints_stay_finite(self):
        x, _ = batch(4)
        x[:, :, 10:14] = 0          # missing frames
        x[:, 2] = 0                 # zero visibility
        x[1] = 0                    # an all-zero clip
        x[2, :5, :, [3, 17]] = 0    # augmentation-style dropped joints
        for model_id in RUNNABLE:
            with self.subTest(model=model_id):
                torch.manual_seed(0)
                model = build_model(model_id).eval()
                with torch.no_grad():
                    self.assertTrue(torch.isfinite(logits_of(model, x).softmax(1)).all())

    def test_wrong_input_shape_is_rejected(self):
        for model_id in ASSIGNED[:-1]:
            with self.subTest(model=model_id), self.assertRaises(ValueError):
                build_model(model_id)(torch.zeros(2, 8, 32, 25))

    def test_unknown_and_blocked_models_fail_without_fallback(self):
        with self.assertRaises(ValueError):
            build_model("not_a_model")
        with self.assertRaises(BlockedModel):
            build_model(DS_GCN)
        self.assertEqual(set(MODEL_IDS), {A3, *ASSIGNED})


class GraphAndPadding(unittest.TestCase):
    def test_graph_is_mediapipe_33(self):
        graph = spatial_graph()
        self.assertEqual(graph.shape, (3, 33, 33))
        np.testing.assert_array_equal(graph[0], np.eye(33))
        distance = shortest_paths()
        self.assertTrue(np.isinf(distance[0, 11]))  # face is its own component in the A3 edge list
        self.assertEqual(distance[11, 15], 2)
        hops = hop_matrix()
        self.assertEqual(hops[0, 11], hops[np.isfinite(distance)].max() + 1)

    def test_skateformer_layout_and_masks(self):
        layout = skate_layout()
        self.assertEqual(len(layout), 36)
        self.assertEqual(int((layout < 0).sum()), 3)
        model = build_model(SKATEFORMER)
        block = model.stages[0].blocks[0]
        for index in range(4):
            for frames in (32, 16, 8, 4):
                mask = block.key_padding(index, frames, 2)
                self.assertFalse(mask.all(dim=1).any())
        joints = torch.arange(33.).view(1, 1, 1, 33)
        padded = joints[..., block.gather_index] * block.slot_real
        np.testing.assert_array_equal(padded[..., block.slot_of_joint].flatten().numpy(), np.arange(33.))
        self.assertEqual(sorted(j for part in SKATE_PARTS for j in part if j >= 0), list(range(33)))

    def test_skateformer_padded_slots_get_no_attention(self):
        model = build_model(SKATEFORMER).eval()
        attention = model.stages[0].blocks[0].attention[1]
        captured = {}

        def hook(module, inputs, output):
            captured["mask"] = inputs[1]
        attention.register_forward_hook(hook)
        x, _ = batch(2)
        with torch.no_grad():
            model(x)
        parts = torch.randn(captured["mask"].shape[0], captured["mask"].shape[1], 12 * 3)
        out_a = attention(parts, captured["mask"])
        parts_b = parts.clone()
        parts_b[captured["mask"]] = 1000.0      # change only padded keys/values
        out_b = attention(parts_b, captured["mask"])
        real = ~captured["mask"]
        torch.testing.assert_close(out_a[real], out_b[real])


class DataProtocol(unittest.TestCase):
    def setUp(self):
        self.data = synthetic_data()
        self.cfg = original.default_config("smoke")

    def test_no_split_leakage(self):
        for outer in (1, 2, 3, 4):
            train, test, inner = runner.splits(self.data, outer)
            self.assertFalse(set(train) & set(test))
            self.assertEqual(sorted(p for p, _, _ in inner), sorted({1, 2, 3, 4} - {outer}))
            for _, fit_indices, val in inner:
                self.assertFalse(set(fit_indices) & set(val))
                self.assertFalse(set(fit_indices) & set(test) or set(val) & set(test))
        broken = dict(self.data, subjects=self.data["subjects"].copy())
        broken["subjects"][:] = 1
        with self.assertRaises(ValueError):
            runner.splits(broken, 1)

    def test_tensor_parity_and_deterministic_augmentation(self):
        indices = np.arange(24)
        a = [x for x, _ in make_loader(self.data, indices, runner.RECIPE, self.cfg, True, 42)]
        b = [x for x, _ in make_loader(self.data, indices, runner.RECIPE, self.cfg, True, 42)]
        c = [x for x, _ in make_loader(self.data, indices, runner.RECIPE, self.cfg, True, 43)]
        for left, right in zip(a, b):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertFalse(all(torch.equal(left, right) for left, right in zip(a, c)))
        self.assertEqual(tuple(a[0].shape[1:]), (8, 32, 33))
        self.assertEqual(runner.RECIPE, original.VARIANTS[A3])

    def test_no_training_batch_of_one_on_real_split_sizes(self):
        sizes = {1: 32, 2: 48, 3: 43, 4: 37}  # GMDCSA-24 clips per subject
        for outer in sizes:
            train = sum(v for k, v in sizes.items() if k != outer)
            self.assertNotEqual(train % 16, 1)
            for inner in sizes:
                if inner != outer:
                    self.assertNotEqual((train - sizes[inner]) % 16, 1)

    def test_class_weights(self):
        train = np.flatnonzero(self.data["subjects"] != 1)
        counts = np.bincount(self.data["labels"][train], minlength=2)
        expected = len(train) / (2 * counts.astype(np.float32))
        np.testing.assert_allclose(expected, [1.0, 1.0])

    def test_generic_loop_matches_original_a3(self):
        from training.component_ablation.models import SkeletonModel
        data, cfg = self.data, dict(self.cfg, max_epochs=3, patience=2)
        train, val = np.flatnonzero(data["subjects"] > 2), np.flatnonzero(data["subjects"] == 2)
        device = torch.device("cpu")
        for validation, epochs in ((val, None), (None, 2)):
            reference, ref_epoch, ref_history = original.fit_model(data, train, validation, runner.RECIPE, cfg, 7,
                                                                   device, epochs)
            mirror, epoch, history = runner.fit_generic(lambda: SkeletonModel(**runner.ARCHITECTURE[A3]), data, train,
                                                        validation, cfg, 7, device, epochs)
            self.assertEqual(ref_epoch, epoch)
            self.assertEqual(ref_history, history)
            for key, value in reference.state_dict().items():
                torch.testing.assert_close(mirror.state_dict()[key], value, rtol=0, atol=0)

    def test_metric_edge_cases(self):
        values = original.metrics(np.array([0, 0, 0]), np.array([0, 0, 0]))
        self.assertIsNone(values["sensitivity"])
        self.assertIsNone(values["precision"])
        self.assertEqual(values["accuracy"], 1.0)
        values = original.metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
        self.assertAlmostEqual(values["f1"], 0.5)
        self.assertEqual((values["tp"], values["fp"], values["fn"], values["tn"]), (1, 1, 1, 1))


class RunnerEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="duc-modern-test-"))
        cls.data = synthetic_data()
        cls.device = torch.device("cpu")
        cls.threads = torch.get_num_threads()
        cls.done = {}
        for model_id in (A3, SKATEFORMER):
            output = cls.tmp / model_id
            cls.done[model_id] = runner.run(cls.data, model_id, "smoke", cls.device, output, cls.threads, synthetic=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def lock(self, model_id):
        return runner.protocol_lock(self.data, model_id, "smoke", self.device, self.threads, True)

    def copy_run(self, model_id, name):
        target = self.tmp / name
        shutil.copytree(self.tmp / model_id, target)
        return target

    def test_smoke_completion_reload_and_coverage(self):
        for model_id, done in self.done.items():
            with self.subTest(model=model_id):
                self.assertEqual((done["fits"], done["outer_runs"], done["inner_fits"], done["refits"]), (4, 1, 3, 1))
                self.assertEqual(done["prediction_rows"], 3 * 8)
                self.assertFalse(done["scientific_result"])
                self.assertTrue(done["synthetic"])
                predictions = pd.read_csv(self.tmp / model_id / "predictions.csv")
                self.assertEqual(list(predictions.columns), runner.PREDICTION_COLUMNS)
                inner = pd.read_csv(self.tmp / model_id / "inner_oof.csv")
                self.assertEqual(len(inner), 24)
                manifest = json.loads((self.tmp / model_id / "models/manifest.json").read_text())
                self.assertEqual(len(manifest), 1)
                checkpoint = self.tmp / model_id / manifest[0]["file"]
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), manifest[0]["sha256"])

    def test_resume_revalidates_and_signature_guard(self):
        output = self.copy_run(A3, "resume_ok")
        done = runner.run(self.data, A3, "smoke", self.device, output, self.threads, synthetic=True, resume=True)
        self.assertEqual(done["fits"], 4)
        partial = self.copy_run(A3, "resume_partial")
        (partial / "completion.json").unlink()
        done = runner.run(self.data, A3, "smoke", self.device, partial, self.threads, synthetic=True, resume=True)
        self.assertEqual(done["fits"], 4)
        changed = dict(self.data, durations=self.data["durations"] * 1.5)
        with self.assertRaises(ValueError):
            runner.run(changed, A3, "smoke", self.device, self.copy_run(A3, "resume_bad"), self.threads,
                       synthetic=True, resume=True)
        with self.assertRaises(FileExistsError):
            runner.run(self.data, A3, "smoke", self.device, self.tmp / A3, self.threads, synthetic=True)

    def test_validate_summary_detects_tampering(self):
        cases = {
            "metric_edit": lambda d: self._edit_csv(d / "pooled_by_seed.csv", "accuracy", 0.999),
            "score_edit": lambda d: self._edit_csv(d / "predictions.csv", "score", 0.123456),
            "drop_row": lambda d: self._drop_row(d / "predictions.csv"),
            "label_edit": lambda d: self._edit_csv(d / "predictions.csv", "label", 1, row=0),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                output = self.copy_run(A3, f"tamper_{name}")
                mutate(output)
                lock = self.lock(A3)
                with self.assertRaises((ValueError, AssertionError)):
                    runner.validate_summary(self.data, lock, output, runner.collect(self.data, lock, self.device, output))

    def test_fold_tampering_is_caught(self):
        cases = {
            "checkpoint": lambda r, d: (d / "model.pt").write_bytes((d / "model.pt").read_bytes()[:-8] + b"tampered"),
            "nan_score": lambda r, d: r["predictions"][0].update(score=float("nan")),
            "policy_score": lambda r, d: r["predictions"][len(r["predictions"]) // 3].update(score=0.4242),
            "threshold": lambda r, d: r["choices"]["balanced"].update(threshold=0.123),
            "inner_label": lambda r, d: r["inner_oof"][0].update(label=1 - r["inner_oof"][0]["label"]),
            "duplicate_id": lambda r, d: r["predictions"].__setitem__(1, dict(r["predictions"][0])),
            "subject": lambda r, d: r["predictions"][0].update(subject=2),
            "epoch": lambda r, d: r["inner"][0].update(selected_epoch=r["inner"][0]["selected_epoch"] + 1),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                output = self.copy_run(A3, f"fold_{name}")
                folder = next((output / "folds").iterdir())
                record = json.loads((folder / "fold.json").read_text())
                mutate(record, folder)
                (folder / "fold.json").write_text(json.dumps(record))
                lock = self.lock(A3)
                with self.assertRaises(Exception):
                    runner.collect(self.data, lock, self.device, output)

    def test_missing_fold_and_empty_directory_fail(self):
        output = self.copy_run(A3, "missing_fold")
        shutil.rmtree(next((output / "folds").iterdir()))
        lock = self.lock(A3)
        with self.assertRaises(ValueError):
            runner.collect(self.data, lock, self.device, output)
        empty = self.tmp / "empty"
        (empty / "folds").mkdir(parents=True)
        with self.assertRaises(Exception):
            runner.validate_summary(self.data, lock, empty, {})

    def test_full_guards(self):
        with self.assertRaises(ValueError):
            runner.run(self.data, A3, "full", self.device, self.tmp / "full_synthetic", self.threads, synthetic=True)
        real_like = dict(self.data, synthetic=False, provenance={})
        with self.assertRaises(ValueError):
            runner.run(real_like, A3, "full", self.device, self.tmp / "full_noinputs", self.threads)
        with self.assertRaises(ValueError):
            runner.verify_test_report_dict({"status": "PASS", "code": {}, "failures": 0})
        with self.assertRaises(BlockedModel):
            runner.run(self.data, DS_GCN, "smoke", self.device, self.tmp / "ds", self.threads, synthetic=True)
        self.assertFalse((self.tmp / "ds").exists())
        with self.assertRaises(ValueError):
            runner.run(self.data, "unknown", "smoke", self.device, self.tmp / "unknown", self.threads, synthetic=True)

    def test_wrong_input_hash_is_rejected(self):
        source = ROOT / "data/gmdcsa24"
        if not (source / "gmdcsa24_cross_subject_folds.csv").exists():
            self.skipTest("real inputs not present")
        copy_dir = self.tmp / "bad_inputs"
        shutil.copytree(source, copy_dir)
        with (copy_dir / "gmdcsa24_cross_subject_folds.csv").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaises(ValueError):
            runner.load_verified(copy_dir)

    @staticmethod
    def _edit_csv(path, column, value, row=0):
        frame = pd.read_csv(path)
        frame.loc[row, column] = value
        frame.to_csv(path, index=False)

    @staticmethod
    def _drop_row(path):
        frame = pd.read_csv(path)
        frame.iloc[1:].to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, help="write a JSON PASS/FAIL report tied to the code hashes")
    args, rest = parser.parse_known_args()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    code = runner.code_hashes()
    report = dict(status="PASS" if result.wasSuccessful() else "FAIL", tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
                  code=code, code_sha256=hashlib.sha256(json.dumps(code, sort_keys=True).encode()).hexdigest(),
                  environment=runner.environment(torch.device("cpu"), torch.get_num_threads()))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "tests_run", "failures", "errors", "skipped")}))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
