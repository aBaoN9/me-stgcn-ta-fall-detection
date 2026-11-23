import io
import unittest
from unittest.mock import patch

import numpy as np

from demo_streamlit.data import (
    METRICS, MODEL_NAMES, RUNS, SEEDS, SKATE, SKATE_COMMIT, SKATE_RUN, checkpoint_for,
    load_canonical_data, load_checkpoint_model, load_metrics, load_predictions, metrics_from_counts, replay,
)


class EvidenceTests(unittest.TestCase):
    def test_all_metrics_match_confusion_counts(self):
        for run_key, run in RUNS.items():
            if not run.checkpoints:
                continue
            table = load_metrics(run_key)
            self.assertEqual(len(table), len(run.variants) * 3)
            for row in table.itertuples():
                self.assertEqual(row.tp + row.tn + row.fp + row.fn, 160)
                for metric in METRICS:
                    self.assertTrue(0 <= getattr(row, metric) <= 1)

    def test_optional_partial_metrics_use_confusion_counts(self):
        fixture = "variant,seed,tp,tn,fp,fn\n" + "".join(
            f"{variant},{seed},71,73,8,8\n"
            for variant in RUNS["tuning"].variants for seed in SEEDS
        )
        with patch("demo_streamlit.data.verified_file", return_value=io.StringIO(fixture)):
            table = load_metrics("tuning")
        self.assertEqual(len(table), 9)
        np.testing.assert_allclose(table.accuracy, .9)
        np.testing.assert_allclose(table.sensitivity, 71 / 79)

    def test_missing_optional_partial_archive_is_not_replaced(self):
        with patch("demo_streamlit.data.verified_file", side_effect=FileNotFoundError("No local archive")):
            with self.assertRaisesRegex(FileNotFoundError, "No local archive"):
                load_metrics("tuning")

    def test_research_primary_means(self):
        table = load_metrics("accepted")
        expected = {"A3_motion_attention": .89375, "TCN_v2_small": .9104166666666667,
                    "GRU_v2_small": .8958333333333334, "CTR_GCN_adapted": .8854166666666666,
                    SKATE: .8416666666666667}
        self.assertEqual(set(MODEL_NAMES), set(expected))
        for variant, accuracy in expected.items():
            self.assertAlmostEqual(table.loc[table.variant == variant, "accuracy"].mean(), accuracy)

    def test_skate_uses_own_verified_checkpoint_and_provenance(self):
        path = checkpoint_for("accepted", SKATE, 42, 1)
        self.assertTrue(path.is_relative_to(SKATE_RUN))
        model, identity = load_checkpoint_model("accepted", SKATE, 42, 1)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 22820)
        self.assertEqual(identity["source_commit"], SKATE_COMMIT)
        self.assertEqual(identity["review_status"], "accepted_pr4_merged")
        with patch("demo_streamlit.data.digest", return_value="bad-hash"):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                checkpoint_for("accepted", SKATE, 42, 1)

    def test_partial_never_falls_back_to_old_weights(self):
        with self.assertRaisesRegex(FileNotFoundError, "not recovered"):
            checkpoint_for("tuning", "A3_motion_attention", 42, 1)

    def test_bad_counts_and_identities_are_rejected(self):
        for counts in [(0, 0, 0, 0), (-1, 81, 0, 80), (78.5, 81, 0, .5)]:
            with self.assertRaises(ValueError):
                metrics_from_counts(*counts)
        with self.assertRaises(ValueError):
            checkpoint_for("accepted", "not-a-model", 42, 1)

    def test_tampered_artifact_is_rejected(self):
        with patch("demo_streamlit.data.digest", return_value="bad-hash"):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                load_predictions("TCN_v2_small", 42)

    def test_every_research_checkpoint_replays_held_out_sample(self):
        data = load_canonical_data()
        maximum_error = 0
        for variant in MODEL_NAMES:
            for seed in SEEDS:
                archived = load_predictions(variant, seed)
                for subject in (1, 2, 3, 4):
                    with self.subTest(variant=variant, seed=seed, subject=subject):
                        sample_index = int(np.flatnonzero(data["subjects"] == subject)[0])
                        result = replay("accepted", variant, seed, sample_index, data)
                        row = archived.loc[archived.sample_index == sample_index].iloc[0]
                        maximum_error = max(maximum_error, abs(result["score"] - row.score))
                        self.assertEqual(result["prediction"], row.prediction)
                        self.assertLess(abs(result["score"] - row.score), .002)
                        self.assertEqual(result["subject"], subject)
        count = len(MODEL_NAMES) * len(SEEDS) * 4
        print(f"{count} real held-out checkpoint replays; maximum CPU/GPU score delta: {maximum_error:.8f}")


if __name__ == "__main__":
    unittest.main()
