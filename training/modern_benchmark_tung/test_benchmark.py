import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from training.backbone_benchmark.models import logits_of
from training.backbone_benchmark.test_benchmark import fixture
from training.component_ablation.features import make_features_v2
from training.component_ablation.models import make_loader

from . import runner
from .adapters.dha import hops, part_ids
from .adapters.hyper import PARTS, augmented_graph
from .models import A3, MODEL_IDS, build_model


class ModernTests(unittest.TestCase):
    def test_outer_inventory_rejects_corruption(self):
        data = fixture()
        seeds = [42]
        subject = 1
        valid = [
            dict(variant=A3, seed=42, policy=policy, sample_index=int(i),
                 video_id=data["video_ids"][i], subject=subject, label=int(data["labels"][i]),
                 score=0.75, prediction=1, threshold=0.5)
            for policy in runner.POLICIES for i in np.flatnonzero(data["subjects"] == subject)
        ]
        runner.summarize(valid, data, seeds, [subject], A3)
        for column, value in (("score", float("nan")), ("label", 9), ("subject", 4),
                              ("seed", 242), ("video_id", "wrong"),
                              ("score", 0.8), ("policy", "extra")):
            with self.subTest(column=column, value=value):
                rows = [dict(row) for row in valid]
                rows[0][column] = value
                with self.assertRaises(ValueError):
                    runner.summarize(rows, data, seeds, [subject], A3)
        for rows in (valid[:-1], [valid[0], *valid]):
            with self.assertRaises(ValueError):
                runner.summarize(rows, data, seeds, [subject], A3)

    def test_all_shapes_and_gradients(self):
        data = fixture()
        data["masks"][0, 4] = False
        features = torch.from_numpy(np.stack([make_features_v2(pose, mask, 1.0)
            for pose, mask in zip(data["poses"], data["masks"])]))
        self.assertEqual(features.shape, (16, 8, 32, 33))
        self.assertTrue((features[0, 2, 4] == 0).all())
        for model_id in MODEL_IDS:
            with self.subTest(model=model_id):
                model = build_model(model_id)
                for batch in (1, 16):
                    model.train()
                    model.zero_grad(set_to_none=True)
                    logits = logits_of(model, features[:batch])
                    self.assertEqual(logits.shape, (batch, 2))
                    self.assertTrue(logits.isfinite().all())
                    torch.nn.functional.cross_entropy(logits, torch.arange(batch) % 2).backward()
                    self.assertTrue(all(p.grad is not None and p.grad.isfinite().all()
                                        for p in model.parameters() if p.requires_grad))
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(logits_of(model, features[:1]),
                                               logits_of(model, features[:1]), rtol=0, atol=0)

    def test_graph_inventories(self):
        self.assertEqual(part_ids().shape, (33,))
        self.assertEqual(set(part_ids().tolist()), set(range(5)))
        self.assertEqual(int(hops()[11, 13]), 1)
        self.assertEqual(int(hops()[0, 0]), 0)
        graph = augmented_graph()
        self.assertEqual(graph.shape, (39, 39))
        torch.testing.assert_close(graph, graph.T)
        self.assertEqual(set().union(*map(set, PARTS)), set(range(33)))
        for index, part in enumerate(PARTS, 33):
            self.assertEqual(set(graph[index, :33].nonzero().flatten().tolist()), set(part))

    def test_canonical_loader_tensor_parity(self):
        data = fixture()
        recipe = runner.original.VARIANTS[A3]
        cfg = runner.original.default_config("full")
        train, test, inner = runner.splits(data, 1)
        self.assertFalse(set(train) & set(test))
        for _, fit, val in inner:
            self.assertFalse(set(fit) & set(val))
            self.assertFalse(set(fit) & set(test))
        seed = 42
        reference = next(iter(make_loader(data, train, recipe, cfg, True, seed)))
        for variant in MODEL_IDS:
            with self.subTest(variant=variant):
                candidate = next(iter(runner.make_loader(data, train, recipe, cfg, True, seed)))
                for x, y in zip(reference, candidate):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertEqual(runner.original.metrics([0, 1], [0, 0])["precision"], None)
        with self.assertRaisesRegex(ValueError, "Unknown model ID"):
            build_model("wrong-id")

    def test_resume_and_tamper_guards(self):
        data = fixture()
        device = torch.device("cpu")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "smoke"
            done = runner.run(data, "smoke", device, output, synthetic=True)
            self.assertEqual(done["actual_fits"], 4)
            self.assertFalse(done["scientific_result"])
            (output / "completion.json").unlink()
            with patch.object(runner, "run_fold", side_effect=AssertionError("Unexpected refit")):
                self.assertEqual(runner.run(data, "smoke", device, output, synthetic=True,
                                            resume=True)["coverage"], "PASS")
            with self.assertRaisesRegex(ValueError, "Resume"):
                runner.run(data, "smoke", device, output, synthetic=True, resume=True,
                           model="cascadeformer_v2_ce")
            path = output / "pooled_by_seed.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "accuracy"] = 5
            frame.to_csv(path, index=False)
            with self.assertRaises(AssertionError):
                runner.run(data, "smoke", device, output, synthetic=True, resume=True)
