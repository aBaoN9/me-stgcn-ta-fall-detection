import tempfile
import unittest
from pathlib import Path

import torch

from ..test_benchmark import fixture
from . import runner
from .models import SKATE, SkateBlock, build_model, partition_indices


class SkateTests(unittest.TestCase):
    def test_partition_dimensions_and_shuffle(self):
        for frames in (16, 32):
            groups = [partition_indices(frames, dt, dj) for dt in (False, True)
                      for dj in (False, True)]
            for group in groups:
                self.assertEqual(group.shape, (frames * 36 // 24, 24))
                torch.testing.assert_close(group.flatten().sort().values, torch.arange(frames * 36))
            self.assertEqual(len({tuple(g.flatten().tolist()) for g in groups}), 4)
            self.assertEqual((groups[0][0] // 36).unique().tolist(), [0, 1, 2, 3])
            self.assertEqual((groups[2][0] // 36).unique().tolist(), list(range(0, frames, frames // 4)))

    def test_padding_does_not_affect_real_joints(self):
        block = SkateBlock(16).eval()
        value = torch.randn(2, 16, 36, 32)
        changed = value.clone()
        changed[:, :, 33:] = 100
        with torch.no_grad():
            torch.testing.assert_close(block(value)[:, :, :33], block(changed)[:, :, :33])

    def test_shape_and_gradients(self):
        inputs = torch.randn(2, 8, 32, 33)
        inputs[:, 2, 5] = 0
        model = build_model(SKATE)
        output = model(inputs)
        self.assertEqual(output.shape, (2, 2))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
        with self.assertRaises(ValueError):
            model(inputs.transpose(2, 3))

    def test_resume_validates_checkpoint_and_coverage(self):
        data = fixture()
        device = torch.device("cpu")
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "smoke"
            result = runner.run(data, "smoke", device, output, synthetic=True)
            self.assertEqual(result["actual_fits"], 4)
            (output / "completion.json").unlink()
            from unittest.mock import patch
            with patch.object(runner, "run_fold", side_effect=AssertionError("unexpected refit")):
                self.assertEqual(runner.run(data, "smoke", device, output,
                                            synthetic=True, resume=True)["coverage"], "PASS")
            next((output / "folds").glob("*/model.pt")).write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "Checkpoint SHA-256"):
                runner.run(data, "smoke", device, output, synthetic=True, resume=True)
