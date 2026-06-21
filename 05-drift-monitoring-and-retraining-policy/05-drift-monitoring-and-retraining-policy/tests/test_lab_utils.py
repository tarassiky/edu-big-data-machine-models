import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = BASE_DIR / "lab_utils.py"
SPEC = importlib.util.spec_from_file_location("lab05_utils", MODULE_PATH)
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


class Lab05UtilsTestCase(unittest.TestCase):
    """Проверяет базовые учебные инварианты утилит ЛР05.

    Тесты ориентированы на сценарии, которые студент должен уметь
    интерпретировать:
    - стабильный поток не должен массово флагать drift;
    - сдвинутый поток должен давать более сильный drift-сигнал;
    - policy обязана возвращать корректное действие по порогам;
    - сравнение после retrain должно включать обе фазы.
    """

    def make_synthetic_dataset(self, n: int = 900, seed: int = 42) -> pd.DataFrame:
        """Генерирует синтетический датасет с контролируемой нелинейностью.

        Args:
            n: Количество объектов в выборке.
            seed: Зерно генератора случайных чисел.

        Returns:
            DataFrame с признаками `age`, `income`, `segment` и `target`.
        """
        rng = np.random.default_rng(seed)

        age = rng.normal(45.0, 12.0, size=n).clip(18, 85)
        income = rng.normal(65000.0, 18000.0, size=n).clip(12000, 180000)
        segment = rng.choice(["A", "B", "C"], size=n, p=[0.55, 0.30, 0.15])

        logit = -3.4 + 0.05 * (age - 45.0) - 0.000015 * (income - 65000.0)
        logit += (segment == "C") * 0.55 + (segment == "B") * 0.20
        prob = 1.0 / (1.0 + np.exp(-logit))
        y = rng.binomial(1, prob).astype(int)

        return pd.DataFrame(
            {
                "age": age,
                "income": income,
                "segment": segment,
                "target": y,
            }
        )

    def test_stable_scenario_does_not_trigger_massive_drift(self):
        """Проверяет, что сценарий `stable` не создает массовых drift-флагов.

        Ожидаемый учебный результат:
            В стабильном окне доля drift-флагов заметно ниже критического уровня.
        """
        df = self.make_synthetic_dataset()
        x, y = lab.split_xy(df)

        x_ref, y_ref, _, _ = lab.prepare_reference_models(x=x, y=y, random_state=42)
        windows = lab.make_monitoring_windows(x_reference=x_ref, y_reference=y_ref, random_state=42)
        drift = lab.build_drift_detection_audit(dataset_name="demo", x_reference=x_ref, windows=windows)

        stable_share = float(
            drift[drift["scenario"] == "stable"]["drift_flag"].astype(float).mean()
        )
        self.assertLess(stable_share, 0.35)

    def test_shifted_scenario_has_more_drift_than_stable(self):
        """Проверяет, что `combined` дает больше drift-флагов, чем `stable`.

        Ожидаемый учебный результат:
            Сдвинутый сценарий должен быть статистически заметнее стабильного.
        """
        df = self.make_synthetic_dataset()
        x, y = lab.split_xy(df)

        x_ref, y_ref, _, _ = lab.prepare_reference_models(x=x, y=y, random_state=7)
        windows = lab.make_monitoring_windows(x_reference=x_ref, y_reference=y_ref, random_state=7)
        drift = lab.build_drift_detection_audit(dataset_name="demo", x_reference=x_ref, windows=windows)

        stable_share = float(
            drift[drift["scenario"] == "stable"]["drift_flag"].astype(float).mean()
        )
        combined_share = float(
            drift[drift["scenario"] == "combined"]["drift_flag"].astype(float).mean()
        )
        self.assertGreater(combined_share, stable_share)

    def test_policy_rule_returns_observe_and_retrain(self):
        """Проверяет корректное срабатывание policy-триггеров.

        Ожидаемый учебный результат:
            При слабых сигналах возвращается `observe`, при пороговом превышении
            хотя бы одного триггера возвращается `retrain`.
        """
        action_observe, reason_observe = lab.choose_retraining_action(
            drift_feature_share=0.10,
            delta_f1_vs_reference=-0.01,
            delta_cost_vs_reference=0.02,
        )
        self.assertEqual(action_observe, "observe")
        self.assertEqual(reason_observe, "no_trigger")

        action_retrain, reason_retrain = lab.choose_retraining_action(
            drift_feature_share=0.34,
            delta_f1_vs_reference=-0.02,
            delta_cost_vs_reference=0.03,
        )
        self.assertEqual(action_retrain, "retrain")
        self.assertIn("drift_share", reason_retrain)

    def test_post_retrain_comparison_contains_both_phases(self):
        """Проверяет структуру сравнения до/после переобучения.

        Ожидаемый учебный результат:
            В итоговой таблице присутствуют обе фазы и валидные метрики стоимости.
        """
        df = self.make_synthetic_dataset()
        x, y = lab.split_xy(df)

        x_ref, y_ref, models, _ = lab.prepare_reference_models(x=x, y=y, random_state=11)
        windows = lab.make_monitoring_windows(x_reference=x_ref, y_reference=y_ref, random_state=11)
        post = lab.build_post_retrain_comparison(
            dataset_name="demo",
            windows=windows,
            reference_models=models,
            model_variant="RandomForest",
            random_state=11,
        )

        self.assertEqual(set(post["phase"].unique()), {"before_retrain", "after_retrain"})
        self.assertTrue({"accuracy", "f1", "expected_cost"}.issubset(set(post.columns)))
        self.assertTrue((post["expected_cost"] >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
