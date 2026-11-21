import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ..test_benchmark import fixture
from training.component_ablation.models import EDGES
from . import runner
from .models import CTR, GRU, build_model, pose_graph


class SupplementTests(unittest.TestCase):
    def test_model_shapes_and_gradients(self):
        features = torch.randn(2, 8, 32, 33)
        features[:, 2, 5] = 0
        for variant in (GRU, CTR):
            with self.subTest(variant=variant):
                model = build_model(variant)
                result = model(features)
                self.assertEqual(result.shape, (2, 2))
                result.square().mean().backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(parameter.grad.isfinite().all())

    def test_graph_orientation_and_coverage(self):
        graph = pose_graph()
        torch.testing.assert_close(graph[0], torch.eye(33))
        self.assertEqual(set(map(tuple, graph[1].nonzero().tolist())),
                         {(target, source) for source, target in EDGES})
        self.assertEqual(set(map(tuple, graph[2].nonzero().tolist())), set(EDGES))
        self.assertEqual(int((graph[1:].sum((0, 1)) > 0).sum()), 33)

    def test_resume_and_evidence(self):
        data = fixture()
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "smoke"
            run_fold = runner.run_fold

            def interrupt(variant, *args):
                if variant == CTR:
                    raise InterruptedError
                return run_fold(variant, *args)

            with patch.object(runner, "run_fold", side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    runner.run(data, "smoke", device, output, synthetic=True)
            with patch.object(runner, "run_fold", wraps=run_fold) as resumed:
                result = runner.run(data, "smoke", device, output, synthetic=True, resume=True)
                self.assertEqual(resumed.call_count, 1)
                self.assertEqual(resumed.call_args.args[0], CTR)
            self.assertEqual(result["actual_fits"], 8)
            self.assertEqual(result["reload"], "PASS")
            self.assertEqual(result["coverage"], "PASS")
            self.assertFalse(result["scientific_result"])
            checkpoint = next((output / "folds").glob("*/model.pt"))
            checkpoint.write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "Checkpoint SHA-256"):
                runner.run(data, "smoke", device, output, synthetic=True, resume=True)
