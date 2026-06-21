import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = BASE_DIR / "lab_utils.py"
SPEC = importlib.util.spec_from_file_location("lab04_utils", MODULE_PATH)
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class Lab04UtilsTestCase(unittest.TestCase):
    def test_compute_expected_cost_total_and_normalized(self):
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 0])

        # FP = 1 (index=2), FN = 2 (index=1,4)
        total = lab.compute_expected_cost(
            y_true=y_true,
            y_pred=y_pred,
            fp_cost=2.0,
            fn_cost=5.0,
            normalize=False,
        )
        normalized = lab.compute_expected_cost(
            y_true=y_true,
            y_pred=y_pred,
            fp_cost=2.0,
            fn_cost=5.0,
            normalize=True,
        )

        self.assertAlmostEqual(total, 12.0)
        self.assertAlmostEqual(normalized, 12.0 / 6.0)

    def test_sweep_thresholds_contract(self):
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_score = np.array([0.1, 0.2, 0.8, 0.9, 0.65, 0.4, 0.55, 0.3])

        grid = lab.sweep_thresholds(y_true=y_true, y_score=y_score, thresholds=[0.3, 0.5, 0.7])

        self.assertEqual(len(grid), 3)
        self.assertTrue({"threshold", "recall", "expected_cost"}.issubset(set(grid.columns)))
        self.assertTrue(np.all(grid["threshold"].between(0.0, 1.0)))

    def test_choose_threshold_policy_respects_recall_guardrail(self):
        grid = pd.DataFrame(
            [
                {"threshold": 0.3, "precision": 0.4, "recall": 0.90, "f1": 0.55, "fp_rate": 0.25, "fn_rate": 0.05, "expected_cost": 0.45},
                {"threshold": 0.5, "precision": 0.6, "recall": 0.65, "f1": 0.62, "fp_rate": 0.10, "fn_rate": 0.12, "expected_cost": 0.40},
                {"threshold": 0.7, "precision": 0.8, "recall": 0.40, "f1": 0.53, "fp_rate": 0.03, "fn_rate": 0.30, "expected_cost": 0.35},
            ]
        )

        winner = lab.choose_threshold_policy(grid, min_recall=0.60)

        self.assertAlmostEqual(float(winner["threshold"]), 0.5)
        self.assertTrue(bool(winner["guardrail_passed"]))
        self.assertIn("recall_ge", str(winner["policy_name"]))

    def test_choose_threshold_policy_fallback_when_guardrail_not_met(self):
        grid = pd.DataFrame(
            [
                {"threshold": 0.6, "precision": 0.7, "recall": 0.50, "f1": 0.58, "fp_rate": 0.08, "fn_rate": 0.20, "expected_cost": 0.35},
                {"threshold": 0.8, "precision": 0.9, "recall": 0.30, "f1": 0.45, "fp_rate": 0.02, "fn_rate": 0.35, "expected_cost": 0.33},
            ]
        )

        winner = lab.choose_threshold_policy(grid, min_recall=0.90)

        self.assertFalse(bool(winner["guardrail_passed"]))
        self.assertEqual(str(winner["policy_name"]), "min_cost_without_recall_guardrail")
        self.assertAlmostEqual(float(winner["threshold"]), 0.8)

    def test_compute_ece_in_range(self):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.1, 0.8, 0.3, 0.7, 0.9, 0.2, 0.65, 0.4, 0.55, 0.15])

        ece = lab.compute_ece(y_true=y_true, y_prob=y_prob, n_bins=5)

        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_choose_best_calibrated_variant_prefers_lower_brier(self):
        calibration_audit = pd.DataFrame(
            [
                {"dataset": "medical", "model": "RandomForest", "variant": "calibrated_sigmoid", "split": "validation", "brier": 0.190, "log_loss": 0.58, "roc_auc": 0.70, "pr_auc": 0.52, "ece": 0.05},
                {"dataset": "medical", "model": "RandomForest", "variant": "calibrated_isotonic", "split": "validation", "brier": 0.170, "log_loss": 0.60, "roc_auc": 0.69, "pr_auc": 0.51, "ece": 0.04},
            ]
        )

        winner = lab.choose_best_calibrated_variant(calibration_audit, dataset_name="medical")

        self.assertEqual(winner, "calibrated_isotonic")


if __name__ == "__main__":
    unittest.main()
