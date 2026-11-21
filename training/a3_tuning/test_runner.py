import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from training.backbone_benchmark.test_benchmark import fixture
from training.component_ablation import training as original
from training.backbone_benchmark import runner as reference
from . import runner


class TuningTests(unittest.TestCase):
    def test_recipe_inventory_and_budget(self):
        self.assertEqual([value['candidate'] for value in runner.recipes()], ['C0', 'C1', 'C2', 'C3', 'C4', 'C5'])
        self.assertEqual(len(runner.tasks()), 36)
        for worker in (0, 1):
            assigned = [task[0] for index, task in enumerate(runner.tasks()) if index % 2 == worker]
            self.assertEqual([assigned.count(model) for model in runner.MODELS], [6, 6, 6])
        self.assertEqual(len(runner.tasks()) * (6 * 3 + 1), 684)
        self.assertEqual(runner.recipes()[5]['cosine_t_max'], 50)

    def test_original_a3_c0_parity(self):
        data = fixture()
        train, validation = np.arange(8), np.arange(8, 12)
        config = dict(runner.recipes()[0], max_epochs=2)
        expected, epoch, _ = original.fit_model(data, train, validation,
                                               original.VARIANTS[runner.A3], config, 1062, torch.device('cpu'))
        with tempfile.TemporaryDirectory() as folder:
            actual, record = runner.fit(runner.A3, data, train, validation, config, 1062,
                                         torch.device('cpu'), folder, 'unit-test')
        self.assertEqual(epoch, record['selected_epoch'])
        for key in expected.state_dict():
            torch.testing.assert_close(expected.state_dict()[key], actual.state_dict()[key], rtol=0, atol=0)

    def test_exact_resume_scheduler_rng_and_checkpoint_guard(self):
        data = fixture()
        train, validation = np.arange(8), np.arange(8, 12)
        for variant in runner.MODELS:
            for candidate in (0, 4, 5):
                with self.subTest(variant=variant, candidate=candidate), tempfile.TemporaryDirectory() as folder:
                    config = dict(runner.recipes()[candidate], max_epochs=3)
                    root = Path(folder)
                    expected, clean = runner.fit(variant, data, train, validation, config, 1062,
                                                 torch.device('cpu'), root / 'clean', 'unit-test')
                    with self.assertRaises(InterruptedError):
                        runner.fit(variant, data, train, validation, config, 1062,
                                   torch.device('cpu'), root / 'resume', 'unit-test', interrupt_after=1)
                    actual, resumed = runner.fit(variant, data, train, validation, config, 1062,
                                                 torch.device('cpu'), root / 'resume', 'unit-test')
                    self.assertEqual(clean['history'], resumed['history'])
                    self.assertEqual(clean['selected_epoch'], resumed['selected_epoch'])
                    for key in expected.state_dict():
                        torch.testing.assert_close(expected.state_dict()[key], actual.state_dict()[key], rtol=0, atol=0)
                    rates = [entry['learning_rate'] for entry in clean['history']]
                    self.assertAlmostEqual(rates[0], config['learning_rate'])
                    if candidate:
                        self.assertGreater(rates[0], rates[1])
                    with self.assertRaisesRegex(ValueError, 'signature'):
                        runner.fit(variant, data, train, validation, config, 999,
                                   torch.device('cpu'), root / 'resume', 'unit-test')
                    with (root / 'resume/state.pt').open('ab') as stream:
                        stream.write(b'corruption')
                    with self.assertRaisesRegex(ValueError, 'mismatch'):
                        runner.fit(variant, data, train, validation, config, 1062,
                                   torch.device('cpu'), root / 'resume', 'unit-test')

    def test_candidate_ranking_and_duplicate_guard(self):
        def candidate(name, scores):
            return dict(config={'candidate': name}, indices=[1, 2, 3, 4], labels=[0, 1, 0, 1],
                        scores=scores, log_probabilities=np.log(np.stack([1-np.array(scores), scores], 1)).tolist())
        first = candidate('C0', [.2, .8, .2, .8])
        second = candidate('C1', [.1, .9, .1, .9])
        selected, _ = runner.choose_candidate([first, second])
        self.assertEqual(selected['config']['candidate'], 'C1')
        selected, _ = runner.choose_candidate([second, second])
        self.assertIs(selected, second)
        second['indices'] = [1, 1, 3, 4]
        with self.assertRaisesRegex(ValueError, 'coverage'):
            runner.choose_candidate([second])

    def test_fixed_refit_checkpoint_roundtrip(self):
        data = fixture()
        for variant in runner.MODELS:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as folder:
                config = runner.recipes()[5]
                actual, first = runner.fit(variant, data, np.arange(12), None, config, 42,
                                           torch.device('cpu'), folder, 'unit-test', fixed_epochs=2)
                loaded, second = runner.fit(variant, data, np.arange(12), None, config, 42,
                                            torch.device('cpu'), folder, 'unit-test', fixed_epochs=2)
                self.assertEqual(len(first['history']), 2)
                self.assertEqual(first, second)
                for key in actual.state_dict():
                    torch.testing.assert_close(actual.state_dict()[key], loaded.state_dict()[key], rtol=0, atol=0)

    def test_outer_held_out_coverage_and_budget_guard(self):
        data = fixture()
        for outer in range(1, 5):
            train, test, inner = reference.splits(data, outer)
            self.assertFalse(set(train) & set(test))
            combined = [int(index) for _, _, validation in inner for index in validation]
            self.assertEqual(sorted(combined), train.tolist())
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Incomplete fit budget'):
                runner.summarize(folder, complete=True)

    def test_fold_selection_checkpoint_and_partial_are_consistent(self):
        data = fixture()
        configurations = [dict(value, max_epochs=1) for value in runner.recipes()]
        with tempfile.TemporaryDirectory() as folder, patch.object(runner, 'recipes', return_value=configurations):
            output = Path(folder)
            runner.run_fold(data, runner.A3, 42, 1, torch.device('cpu'), output, 'unit-test')
            target = output / 'folds' / runner.task_name(runner.A3, 42, 1)
            fold = runner.read_json(target / 'fold.json')
            checkpoint = torch.load(target / 'model.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['decision'], runner.read_json(target / 'selection.json'))
            self.assertEqual(len(fold['decision']['ranking']), 6)
            self.assertEqual({row['sample_index'] for row in fold['predictions']}, {0, 1, 2, 3})
            model = runner.model_for(runner.A3, checkpoint['dropout'])
            model.load_state_dict(checkpoint['state'])
            scores, _ = runner.predict(model, data, np.arange(4), configurations[0], torch.device('cpu'))
            np.testing.assert_allclose(scores, [row['score'] for row in fold['predictions']], rtol=0, atol=0)
            progress = runner.summarize(output)
            self.assertEqual(progress['completed_fits'], 19)
            self.assertEqual(progress['completed_outer'], 1)
            self.assertFalse((output / 'completion.json').exists())
            runner.run_fold(data, runner.A3, 42, 1, torch.device('cpu'), output, 'unit-test')
            with self.assertRaisesRegex(ValueError, 'signature/checkpoint mismatch'):
                runner.run_fold(data, runner.A3, 42, 1, torch.device('cpu'), output, 'changed-lock')

    def test_real_full_requires_cuda(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'requires CUDA'):
                runner.full(fixture(), torch.device('cpu'), folder, folder, 1)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
