"""Утилиты для ЛР 05: drift monitoring и retraining policy.

ЛР 05 автономна относительно предыдущих лабораторных: использует
исходные датасеты курса и воспроизводимый мониторинговый сценарий,
в котором сравниваются стабильное окно и окна со сдвигами.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
LAB01_DIR = BASE_DIR.parent / "01-feature-importance-and-selection"
DATA_DIR = LAB01_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

SEED = 42
DEFAULT_ALPHA = 0.05
DEFAULT_PSI_WARN = 0.10
DEFAULT_PSI_ALERT = 0.25
DEFAULT_RETRAIN_F1_DROP = 0.05
DEFAULT_RETRAIN_COST_INCREASE = 0.15
DEFAULT_DRIFT_FEATURE_SHARE = 0.30
DEFAULT_FP_COST = 1.0
DEFAULT_FN_COST = 5.0

DATASET_PATHS = {
    "medical": DATA_DIR / "medical_cardiovascular_risk.csv",
    "finance": DATA_DIR / "finance_credit_risk.csv",
}

DRIFT_SCENARIOS = ("stable", "covariate", "prior", "combined")

DRIFT_DETECTION_AUDIT_COLUMNS = [
    "dataset",
    "window_id",
    "scenario",
    "feature",
    "feature_type",
    "detector",
    "statistic",
    "p_value",
    "effect_size",
    "drift_flag",
]

MONITORING_QUALITY_AUDIT_COLUMNS = [
    "dataset",
    "window_id",
    "scenario",
    "model_variant",
    "accuracy",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "ece",
    "expected_cost",
    "delta_f1_vs_reference",
    "delta_cost_vs_reference",
]

RETRAINING_POLICY_DECISIONS_COLUMNS = [
    "dataset",
    "window_id",
    "scenario",
    "drift_feature_share",
    "delta_f1_vs_reference",
    "delta_cost_vs_reference",
    "policy_action",
    "trigger_reason",
]

POST_RETRAIN_COMPARISON_COLUMNS = [
    "dataset",
    "scenario",
    "phase",
    "accuracy",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier",
    "ece",
    "expected_cost",
]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Загружает датасет и валидирует минимальный контракт ЛР05.

    Учебный контекст:
        Используется в шаге подготовки данных (практика 1), чтобы гарантировать
        единый формат входа для всех дальнейших расчетов.

    Args:
        path: Путь к CSV-файлу с признаками и целевой колонкой.

    Returns:
        DataFrame с исходными данными, включая колонку `target`.

    Raises:
        ValueError: Если в файле отсутствует обязательная колонка `target`.
    """

    frame = pd.read_csv(path)
    if "target" not in frame.columns:
        raise ValueError(f"В датасете {path} отсутствует колонка 'target'.")
    return frame


def load_course_datasets() -> Dict[str, pd.DataFrame]:
    """Возвращает учебные датасеты курса в едином словаре.

    Учебный контекст:
        Применяется в практиках 1 и 2 как стандартная точка входа для сценариев
        мониторинга без зависимости от артефактов предыдущих лабораторных.

    Args:
        None.

    Returns:
        Словарь вида `{dataset_name: dataframe}` для `medical` и `finance`.
    """

    return {name: load_dataset(path) for name, path in DATASET_PATHS.items()}


def split_xy(df: pd.DataFrame, target: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    """Разделяет таблицу на признаки и целевую переменную.

    Учебный контекст:
        Используется перед обучением reference-моделей и перед генерацией
        мониторинговых окон.

    Args:
        df: Полная таблица с признаками и колонкой таргета.
        target: Имя колонки целевой переменной.

    Returns:
        Кортеж `(x, y)`, где `x` — таблица признаков, `y` — бинарный таргет.
    """

    x = df.drop(columns=[target]).copy()
    y = df[target].astype(int).copy()
    return x, y


def infer_feature_types(x: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Определяет типы признаков для последующей обработки и drift-анализа.

    Учебный контекст:
        Нужна для корректного выбора детектора (`KS` или `chi2`) и пайплайна
        препроцессинга.

    Args:
        x: Таблица признаков без целевой переменной.

    Returns:
        Кортеж `(numeric_features, categorical_features)`.
    """

    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in x.columns if col not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Создает единый препроцессор для числовых и категориальных признаков.

    Учебный контекст:
        Обеспечивает воспроизводимую обработку данных в обоих моделях ЛР05,
        чтобы сравнение качества было корректным.

    Args:
        x: Таблица признаков, по которой определяется тип каждого столбца.

    Returns:
        `ColumnTransformer` с обработкой:
        - числовые: `SimpleImputer(median)` + `StandardScaler`;
        - категориальные: `SimpleImputer(most_frequent)` + `OneHotEncoder`.
    """

    numeric_features, categorical_features = infer_feature_types(x)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def make_model(model_variant: str, random_state: int = SEED):
    """Создает модель выбранного семейства.

    Учебный контекст:
        В ЛР05 используются две базовые модели курса, чтобы показать, что
        мониторинг должен быть сопоставим между разными алгоритмами.

    Args:
        model_variant: Имя модели (`LogisticRegression` или `RandomForest`).
        random_state: Начальное зерно для воспроизводимости.

    Returns:
        Экземпляр sklearn-классификатора.

    Raises:
        ValueError: Если передано неподдерживаемое имя модели.
    """

    if model_variant == "LogisticRegression":
        return LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        )

    if model_variant == "RandomForest":
        return RandomForestClassifier(
            n_estimators=320,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    raise ValueError(
        "Неподдерживаемая модель. Ожидались: 'LogisticRegression' или 'RandomForest'."
    )


def make_model_pipeline(x_train: pd.DataFrame, model_variant: str, random_state: int = SEED) -> Pipeline:
    """Собирает обучающий pipeline из препроцессора и модели.

    Учебный контекст:
        Pipeline исключает рассинхронизацию между обучением и инференсом и
        делает сравнение окон мониторинга методологически корректным.

    Args:
        x_train: Обучающая таблица признаков для определения типов колонок.
        model_variant: Имя модели (`LogisticRegression` или `RandomForest`).
        random_state: Зерно генератора случайных чисел.

    Returns:
        Настроенный `Pipeline(preprocessor -> model)`.
    """

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(x_train)),
            ("model", make_model(model_variant=model_variant, random_state=random_state)),
        ]
    )


def get_binary_score_vector(model, x_data) -> np.ndarray:
    """Преобразует предсказания модели в вероятностный score-вектор.

    Учебный контекст:
        Единый формат score в диапазоне `[0, 1]` нужен для корректного расчета
        `ROC-AUC`, `PR-AUC`, `Brier`, `ECE` и последующей policy-оценки.

    Args:
        model: Обученная модель или pipeline sklearn.
        x_data: Матрица признаков для инференса.

    Returns:
        Вектор оценок положительного класса в диапазоне `[0, 1]`.
    """

    if hasattr(model, "predict_proba"):
        score = np.asarray(model.predict_proba(x_data)[:, 1], dtype=float)
        return np.clip(score, 0.0, 1.0)

    if hasattr(model, "decision_function"):
        margin = np.asarray(model.decision_function(x_data), dtype=float)
        margin = np.clip(margin, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-margin))

    pred = np.asarray(model.predict(x_data), dtype=float)
    return np.clip(pred, 0.0, 1.0)


def safe_roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Безопасно вычисляет `ROC-AUC` для бинарной задачи.

    Учебный контекст:
        В некоторых окнах мониторинга может встречаться один класс таргета;
        в таком случае метрика не определена и возвращается `NaN`.

    Args:
        y_true: Истинные бинарные метки.
        y_score: Вероятностные оценки положительного класса.

    Returns:
        Значение `ROC-AUC` или `NaN`, если метрика не определена.
    """

    y_true_arr = np.asarray(y_true).astype(int)
    if np.unique(y_true_arr).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true_arr, y_score))


def safe_pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Безопасно вычисляет `PR-AUC` для бинарной задачи.

    Учебный контекст:
        Аналогично `ROC-AUC`, метрика `PR-AUC` может быть не определена на
        вырожденных выборках, что важно корректно отражать в мониторинге.

    Args:
        y_true: Истинные бинарные метки.
        y_score: Вероятностные оценки положительного класса.

    Returns:
        Значение `PR-AUC` или `NaN`, если метрика не определена.
    """

    y_true_arr = np.asarray(y_true).astype(int)
    if np.unique(y_true_arr).size < 2:
        return float("nan")
    return float(average_precision_score(y_true_arr, y_score))


def compute_ece(y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10) -> float:
    """Вычисляет `Expected Calibration Error` (ECE).

    Учебный контекст:
        Показывает, насколько вероятности модели согласованы с реальными
        частотами событий в окнах мониторинга.

    Args:
        y_true: Истинные бинарные метки.
        y_prob: Вероятности положительного класса.
        n_bins: Количество корзин для калибровочной оценки.

    Returns:
        Значение ECE в диапазоне `[0, 1]`.

    Raises:
        ValueError: Если `n_bins <= 1`.
    """

    if n_bins <= 1:
        raise ValueError("n_bins должен быть больше 1.")

    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_prob_arr = np.clip(y_prob_arr, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, bins, right=True)

    total = len(y_true_arr)
    if total == 0:
        return 0.0

    ece = 0.0
    for bin_id in range(1, n_bins + 1):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        prob_mean = float(np.mean(y_prob_arr[mask]))
        target_mean = float(np.mean(y_true_arr[mask]))
        ece += (np.sum(mask) / total) * abs(prob_mean - target_mean)

    return float(ece)


def compute_expected_cost(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
    normalize: bool = True,
) -> float:
    """Считает ожидаемую стоимость ошибок на основе штрафов `FP/FN`.

    Учебный контекст:
        Метрика `expected_cost` используется в ЛР05 как один из триггеров
        policy-решения о переобучении.

    Args:
        y_true: Истинные бинарные метки.
        y_pred: Предсказанные бинарные метки.
        fp_cost: Стоимость ложноположительной ошибки.
        fn_cost: Стоимость ложноотрицательной ошибки.
        normalize: Если `True`, возвращается средняя стоимость на объект.

    Returns:
        Итоговая стоимость ошибок (нормированная или суммарная).

    Raises:
        ValueError: Если длины `y_true` и `y_pred` не совпадают.
    """

    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError("y_true и y_pred должны иметь одинаковую длину.")

    fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
    fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
    total_cost = float(fp_cost * fp + fn_cost * fn)

    if not normalize:
        return total_cost

    n = len(y_true_arr)
    return total_cost / n if n > 0 else 0.0


def compute_psi(reference_values: Sequence, current_values: Sequence, bins: int = 10) -> float:
    """Вычисляет `Population Stability Index` между двумя распределениями.

    Учебный контекст:
        `PSI` интерпретируется как сила сдвига распределения и служит
        эффект-размером в `drift_detection_audit`.

    Args:
        reference_values: Значения признака в референсном окне.
        current_values: Значения признака в текущем окне мониторинга.
        bins: Число корзин для числового признака.

    Returns:
        Значение `PSI` (чем больше, тем сильнее сдвиг).
    """

    eps = 1e-6
    reference = pd.Series(reference_values)
    current = pd.Series(current_values)

    if reference.empty or current.empty:
        return 0.0

    if pd.api.types.is_numeric_dtype(reference):
        ref_num = pd.to_numeric(reference, errors="coerce").dropna()
        cur_num = pd.to_numeric(current, errors="coerce").dropna()
        if ref_num.empty or cur_num.empty:
            return 0.0

        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.unique(np.quantile(ref_num, quantiles))
        # Если квантильные границы схлопнулись, переходим к равномерной сетке,
        # чтобы не терять оценку PSI на почти константных признаках.
        if len(edges) < 3:
            left = float(min(ref_num.min(), cur_num.min()))
            right = float(max(ref_num.max(), cur_num.max()))
            if np.isclose(left, right):
                return 0.0
            edges = np.linspace(left, right, bins + 1)

        ref_hist, _ = np.histogram(ref_num, bins=edges)
        cur_hist, _ = np.histogram(cur_num, bins=edges)
    else:
        ref_cat = reference.fillna("missing").astype(str)
        cur_cat = current.fillna("missing").astype(str)
        categories = sorted(set(ref_cat.unique()) | set(cur_cat.unique()))
        ref_hist = ref_cat.value_counts().reindex(categories, fill_value=0).to_numpy()
        cur_hist = cur_cat.value_counts().reindex(categories, fill_value=0).to_numpy()

    if ref_hist.sum() == 0 or cur_hist.sum() == 0:
        return 0.0

    ref_ratio = np.clip(ref_hist / ref_hist.sum(), eps, None)
    cur_ratio = np.clip(cur_hist / cur_hist.sum(), eps, None)
    return float(np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio)))


def _sample_indices(indices: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Выбирает индексы с безопасным fallback на sample with replacement."""

    if len(indices) == 0:
        raise ValueError("Невозможно построить окно: пустой набор индексов.")
    replace = len(indices) < size
    return rng.choice(indices, size=size, replace=replace)


def _apply_covariate_shift(x_window: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Вносит контролируемый covariate-shift в окно признаков."""

    shifted = x_window.copy()
    numeric_features, categorical_features = infer_feature_types(shifted)

    for col in numeric_features[:2]:
        col_values = pd.to_numeric(shifted[col], errors="coerce")
        std = float(col_values.std(skipna=True))
        shift = 0.8 * std if std > 1e-8 else 0.3
        shifted[col] = col_values.fillna(col_values.median()) + shift

    if categorical_features:
        col = categorical_features[0]
        categories = shifted[col].dropna().astype(str).unique().tolist()
        if len(categories) > 1:
            target = categories[-1]
            mask = rng.random(len(shifted)) < 0.40
            shifted.loc[mask, col] = target

    return shifted


def make_monitoring_windows(
    x_reference: pd.DataFrame,
    y_reference: pd.Series,
    window_size: int = 240,
    random_state: int = SEED,
) -> List[Dict[str, object]]:
    """Генерирует четыре мониторинговых окна для сценариев `drift`.

    Учебный контекст:
        Это основа практики 1: одно стабильное окно и три сдвинутых сценария
        (`covariate`, `prior`, `combined`) для сравнимого аудита.

    Args:
        x_reference: Референсные признаки.
        y_reference: Референсный таргет.
        window_size: Размер каждого окна мониторинга.
        random_state: Зерно генератора случайных чисел.

    Returns:
        Список словарей с полями `window_id`, `scenario`, `x_window`, `y_window`.

    Raises:
        ValueError: Если длины `x_reference` и `y_reference` различаются.
        ValueError: Если `window_size < 40`.
    """

    if len(x_reference) != len(y_reference):
        raise ValueError("x_reference и y_reference должны иметь одинаковую длину.")
    if window_size < 40:
        raise ValueError("window_size должен быть не меньше 40 для стабильных оценок.")

    x_ref = x_reference.reset_index(drop=True)
    y_ref = pd.Series(y_reference).astype(int).reset_index(drop=True)

    rng = np.random.default_rng(random_state)
    n = len(x_ref)
    size = min(window_size, n)

    all_idx = np.arange(n)
    stable_idx = _sample_indices(all_idx, size, rng)
    x_stable = x_ref.iloc[stable_idx].reset_index(drop=True)
    y_stable = y_ref.iloc[stable_idx].reset_index(drop=True)

    positive_idx = np.where(y_ref.to_numpy() == 1)[0]
    negative_idx = np.where(y_ref.to_numpy() == 0)[0]

    # В сценарии prior намеренно повышаем долю позитивного класса до ~75%,
    # чтобы смоделировать заметный сдвиг априорной вероятности события.
    n_pos = max(1, int(round(size * 0.75)))
    n_neg = max(1, size - n_pos)
    if n_pos + n_neg != size:
        n_neg = size - n_pos

    prior_pos_idx = _sample_indices(positive_idx, n_pos, rng)
    prior_neg_idx = _sample_indices(negative_idx, n_neg, rng)
    prior_idx = np.concatenate([prior_pos_idx, prior_neg_idx])
    rng.shuffle(prior_idx)

    x_prior = x_ref.iloc[prior_idx].reset_index(drop=True)
    y_prior = y_ref.iloc[prior_idx].reset_index(drop=True)

    x_covariate = _apply_covariate_shift(x_stable, rng)
    y_covariate = y_stable.copy()

    x_combined = _apply_covariate_shift(x_prior, rng)
    y_combined = y_prior.copy()

    windows = [
        {"window_id": 1, "scenario": "stable", "x_window": x_stable, "y_window": y_stable},
        {
            "window_id": 2,
            "scenario": "covariate",
            "x_window": x_covariate,
            "y_window": y_covariate,
        },
        {"window_id": 3, "scenario": "prior", "x_window": x_prior, "y_window": y_prior},
        {
            "window_id": 4,
            "scenario": "combined",
            "x_window": x_combined,
            "y_window": y_combined,
        },
    ]

    return windows


def detect_feature_drift(
    reference_feature: Sequence,
    current_feature: Sequence,
    feature_type: str,
    alpha: float = DEFAULT_ALPHA,
    psi_alert: float = DEFAULT_PSI_ALERT,
) -> Dict[str, float | bool | str]:
    """Оценивает `drift` одного признака статистическим тестом и `PSI`.

    Учебный контекст:
        Функция формирует элементарный сигнал для таблицы
        `drift_detection_audit.csv` на уровне отдельного признака.

    Args:
        reference_feature: Значения признака в референсном окне.
        current_feature: Значения признака в текущем окне.
        feature_type: Тип признака (`numeric` или `categorical`).
        alpha: Порог значимости для `p-value`.
        psi_alert: Порог тревоги по `PSI`.

    Returns:
        Словарь с полями `detector`, `statistic`, `p_value`, `effect_size`,
        `drift_flag`.

    Raises:
        ValueError: Если `feature_type` отличен от `numeric/categorical`.
    """

    if feature_type not in {"numeric", "categorical"}:
        raise ValueError("feature_type должен быть 'numeric' или 'categorical'.")

    if feature_type == "numeric":
        ref = pd.to_numeric(pd.Series(reference_feature), errors="coerce").dropna().to_numpy()
        cur = pd.to_numeric(pd.Series(current_feature), errors="coerce").dropna().to_numpy()

        if len(ref) == 0 or len(cur) == 0:
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.ks_2samp(ref, cur, alternative="two-sided", method="auto")

        detector = "ks"
    else:
        ref = pd.Series(reference_feature).fillna("missing").astype(str)
        cur = pd.Series(current_feature).fillna("missing").astype(str)
        categories = sorted(set(ref.unique()) | set(cur.unique()))
        ref_counts = ref.value_counts().reindex(categories, fill_value=0).to_numpy()
        cur_counts = cur.value_counts().reindex(categories, fill_value=0).to_numpy()

        try:
            statistic, p_value, _, _ = stats.chi2_contingency(np.vstack([ref_counts, cur_counts]))
        except ValueError:
            statistic, p_value = 0.0, 1.0

        detector = "chi2"

    effect_size = compute_psi(reference_feature, current_feature, bins=10)
    drift_flag = bool((float(p_value) < alpha) or (float(effect_size) >= psi_alert))

    return {
        "detector": detector,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "drift_flag": drift_flag,
    }


def build_drift_detection_audit(
    dataset_name: str,
    x_reference: pd.DataFrame,
    windows: Sequence[Dict[str, object]],
    alpha: float = DEFAULT_ALPHA,
    psi_alert: float = DEFAULT_PSI_ALERT,
) -> pd.DataFrame:
    """Строит аудит `drift` по всем окнам и признакам.

    Учебный контекст:
        Это первый обязательный артефакт практики 1, на который затем опирается
        policy-логика из практики 2.

    Args:
        dataset_name: Имя датасета (`medical` или `finance`).
        x_reference: Референсная таблица признаков.
        windows: Набор мониторинговых окон.
        alpha: Порог значимости для детекторов на `p-value`.
        psi_alert: Порог тревоги по `PSI`.

    Returns:
        DataFrame с колонками `DRIFT_DETECTION_AUDIT_COLUMNS`.
    """

    numeric_features, _ = infer_feature_types(x_reference)
    rows: List[Dict[str, object]] = []

    for window in windows:
        window_id = int(window["window_id"])
        scenario = str(window["scenario"])
        x_window = pd.DataFrame(window["x_window"])

        for feature in x_reference.columns.tolist():
            feature_type = "numeric" if feature in numeric_features else "categorical"
            stats_row = detect_feature_drift(
                reference_feature=x_reference[feature],
                current_feature=x_window[feature],
                feature_type=feature_type,
                alpha=alpha,
                psi_alert=psi_alert,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "window_id": window_id,
                    "scenario": scenario,
                    "feature": feature,
                    "feature_type": feature_type,
                    "detector": stats_row["detector"],
                    "statistic": stats_row["statistic"],
                    "p_value": stats_row["p_value"],
                    "effect_size": stats_row["effect_size"],
                    "drift_flag": bool(stats_row["drift_flag"]),
                }
            )

    frame = pd.DataFrame(rows)
    return frame.loc[:, DRIFT_DETECTION_AUDIT_COLUMNS]


def evaluate_binary_predictions(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float = 0.5,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
) -> Dict[str, float]:
    """Считает метрики качества, калибровки и стоимости бинарной модели.

    Учебный контекст:
        Метрики этой функции формируют основу таблиц
        `monitoring_quality_audit.csv` и `post_retrain_comparison.csv`.

    Args:
        y_true: Истинные бинарные метки.
        y_score: Вероятностные оценки положительного класса.
        threshold: Порог бинаризации вероятностей.
        fp_cost: Стоимость ложноположительной ошибки.
        fn_cost: Стоимость ложноотрицательной ошибки.

    Returns:
        Словарь метрик: `accuracy`, `f1`, `roc_auc`, `pr_auc`, `brier`,
        `ece`, `expected_cost`.
    """

    y_true_arr = np.asarray(y_true).astype(int)
    y_score_arr = np.clip(np.asarray(y_score, dtype=float), 0.0, 1.0)
    y_pred = (y_score_arr >= float(threshold)).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true_arr, y_score_arr),
        "pr_auc": safe_pr_auc(y_true_arr, y_score_arr),
        "brier": float(np.mean((y_score_arr - y_true_arr) ** 2)),
        "ece": float(compute_ece(y_true_arr, y_score_arr, n_bins=10)),
        "expected_cost": float(
            compute_expected_cost(
                y_true=y_true_arr,
                y_pred=y_pred,
                fp_cost=fp_cost,
                fn_cost=fn_cost,
                normalize=True,
            )
        ),
    }


def prepare_reference_models(
    x: pd.DataFrame,
    y: pd.Series,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
    """Обучает reference-модели и вычисляет базовые метрики.

    Учебный контекст:
        Это отправная точка для сравнения: все последующие дельты качества и
        стоимости считаются относительно `reference`.

    Args:
        x: Таблица признаков.
        y: Бинарная целевая переменная.
        random_state: Зерно генератора случайных чисел.

    Returns:
        Кортеж:
        - `x_reference`: референсные признаки;
        - `y_reference`: референсный таргет;
        - `models`: словарь обученных pipeline-моделей;
        - `reference_metrics`: базовые метрики по каждой модели.
    """

    x_train, x_reference, y_train, y_reference = train_test_split(
        x,
        y,
        test_size=0.35,
        random_state=random_state,
        stratify=y,
    )

    models: Dict[str, Pipeline] = {}
    reference_metrics: Dict[str, Dict[str, float]] = {}
    for model_variant in ("LogisticRegression", "RandomForest"):
        pipeline = make_model_pipeline(x_train=x_train, model_variant=model_variant, random_state=random_state)
        pipeline.fit(x_train, y_train)

        score = get_binary_score_vector(pipeline, x_reference)
        reference_metrics[model_variant] = evaluate_binary_predictions(y_reference, score)
        models[model_variant] = pipeline

    return x_reference.reset_index(drop=True), y_reference.reset_index(drop=True), models, reference_metrics


def build_monitoring_quality_audit(
    dataset_name: str,
    models: Dict[str, Pipeline],
    reference_metrics: Dict[str, Dict[str, float]],
    windows: Sequence[Dict[str, object]],
) -> pd.DataFrame:
    """Строит аудит качества и стоимости по всем окнам и моделям.

    Учебный контекст:
        Это второй обязательный артефакт практики 1, который связывает
        статистический сдвиг с прикладными метриками качества.

    Args:
        dataset_name: Имя датасета (`medical` или `finance`).
        models: Словарь reference-моделей.
        reference_metrics: Базовые метрики reference-среза.
        windows: Набор мониторинговых окон.

    Returns:
        DataFrame с колонками `MONITORING_QUALITY_AUDIT_COLUMNS`.
    """

    rows: List[Dict[str, object]] = []

    for window in windows:
        window_id = int(window["window_id"])
        scenario = str(window["scenario"])
        x_window = pd.DataFrame(window["x_window"])
        y_window = pd.Series(window["y_window"]).astype(int)

        for model_variant, model in models.items():
            score = get_binary_score_vector(model, x_window)
            metrics = evaluate_binary_predictions(y_window, score)

            baseline = reference_metrics[model_variant]
            row = {
                "dataset": dataset_name,
                "window_id": window_id,
                "scenario": scenario,
                "model_variant": model_variant,
                **metrics,
                "delta_f1_vs_reference": float(metrics["f1"] - baseline["f1"]),
                "delta_cost_vs_reference": float(
                    metrics["expected_cost"] - baseline["expected_cost"]
                ),
            }
            rows.append(row)

    frame = pd.DataFrame(rows)
    return frame.loc[:, MONITORING_QUALITY_AUDIT_COLUMNS]


def choose_retraining_action(
    drift_feature_share: float,
    delta_f1_vs_reference: float,
    delta_cost_vs_reference: float,
    drift_share_threshold: float = DEFAULT_DRIFT_FEATURE_SHARE,
    retrain_f1_drop: float = DEFAULT_RETRAIN_F1_DROP,
    retrain_cost_increase: float = DEFAULT_RETRAIN_COST_INCREASE,
) -> Tuple[str, str]:
    """Выбирает действие `observe/retrain` по фиксированным триггерам.

    Учебный контекст:
        Функция реализует ключевое policy-правило ЛР05 и используется при
        агрегировании решения на уровне окна мониторинга.

    Args:
        drift_feature_share: Доля признаков с `drift_flag=True`.
        delta_f1_vs_reference: Изменение `f1` относительно reference.
        delta_cost_vs_reference: Изменение `expected_cost` относительно reference.
        drift_share_threshold: Порог по доле drift-признаков.
        retrain_f1_drop: Допустимое падение `f1` перед retrain.
        retrain_cost_increase: Допустимый рост стоимости перед retrain.

    Returns:
        Кортеж `(policy_action, trigger_reason)`, где `policy_action` —
        `observe` или `retrain`.
    """

    reasons: List[str] = []
    if drift_feature_share >= drift_share_threshold:
        reasons.append("drift_share")
    if delta_f1_vs_reference <= -abs(retrain_f1_drop):
        reasons.append("f1_drop")
    if delta_cost_vs_reference >= abs(retrain_cost_increase):
        reasons.append("cost_increase")

    if reasons:
        return "retrain", ";".join(reasons)
    return "observe", "no_trigger"


def build_retraining_policy_decisions(
    drift_detection_audit: pd.DataFrame,
    monitoring_quality_audit: pd.DataFrame,
    drift_share_threshold: float = DEFAULT_DRIFT_FEATURE_SHARE,
    retrain_f1_drop: float = DEFAULT_RETRAIN_F1_DROP,
    retrain_cost_increase: float = DEFAULT_RETRAIN_COST_INCREASE,
) -> pd.DataFrame:
    """Строит policy-решения на уровне каждого мониторингового окна.

    Учебный контекст:
        Это центральный артефакт практики 2, где объединяются сигналы drift и
        деградация качества/стоимости.

    Args:
        drift_detection_audit: Таблица drift-сигналов по признакам.
        monitoring_quality_audit: Таблица quality/cost-сигналов.
        drift_share_threshold: Порог retrain по доле drift-признаков.
        retrain_f1_drop: Порог retrain по падению `f1`.
        retrain_cost_increase: Порог retrain по росту стоимости.

    Returns:
        DataFrame с колонками `RETRAINING_POLICY_DECISIONS_COLUMNS`.

    Raises:
        ValueError: Если входные таблицы не содержат обязательных колонок.
    """

    required_drift = {"dataset", "window_id", "scenario", "drift_flag"}
    required_quality = {
        "dataset",
        "window_id",
        "scenario",
        "delta_f1_vs_reference",
        "delta_cost_vs_reference",
    }

    if not required_drift.issubset(set(drift_detection_audit.columns)):
        raise ValueError("drift_detection_audit не содержит обязательные колонки.")
    if not required_quality.issubset(set(monitoring_quality_audit.columns)):
        raise ValueError("monitoring_quality_audit не содержит обязательные колонки.")

    drift_agg = (
        drift_detection_audit.groupby(["dataset", "window_id", "scenario"], as_index=False)[
            "drift_flag"
        ]
        .mean()
        .rename(columns={"drift_flag": "drift_feature_share"})
    )

    quality_agg = (
        monitoring_quality_audit.groupby(["dataset", "window_id", "scenario"], as_index=False)
        .agg(
            # Берем худший случай по качеству и стоимости, чтобы policy было
            # консервативным и не пропускало рискованные окна.
            delta_f1_vs_reference=("delta_f1_vs_reference", "min"),
            delta_cost_vs_reference=("delta_cost_vs_reference", "max"),
        )
        .reset_index(drop=True)
    )

    merged = drift_agg.merge(quality_agg, on=["dataset", "window_id", "scenario"], how="inner")

    actions: List[str] = []
    reasons: List[str] = []
    for row in merged.itertuples(index=False):
        action, reason = choose_retraining_action(
            drift_feature_share=float(row.drift_feature_share),
            delta_f1_vs_reference=float(row.delta_f1_vs_reference),
            delta_cost_vs_reference=float(row.delta_cost_vs_reference),
            drift_share_threshold=drift_share_threshold,
            retrain_f1_drop=retrain_f1_drop,
            retrain_cost_increase=retrain_cost_increase,
        )
        actions.append(action)
        reasons.append(reason)

    merged["policy_action"] = actions
    merged["trigger_reason"] = reasons
    return merged.loc[:, RETRAINING_POLICY_DECISIONS_COLUMNS]


def _safe_window_split(
    x_window: pd.DataFrame,
    y_window: pd.Series,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Делит окно на train/test для post-retrain сравнения."""

    y_unique = np.unique(np.asarray(y_window).astype(int))
    stratify = y_window if len(y_unique) > 1 else None

    return train_test_split(
        x_window,
        y_window,
        test_size=0.40,
        random_state=random_state,
        stratify=stratify,
    )


def build_post_retrain_comparison(
    dataset_name: str,
    windows: Sequence[Dict[str, object]],
    reference_models: Dict[str, Pipeline],
    model_variant: str = "RandomForest",
    random_state: int = SEED,
) -> pd.DataFrame:
    """Сравнивает метрики до и после переобучения на каждом сценарии.

    Учебный контекст:
        Это финальный артефакт практики 2, который проверяет, улучшает ли
        `retrain` качество и стоимость на сдвинутых окнах.

    Args:
        dataset_name: Имя датасета (`medical` или `finance`).
        windows: Набор мониторинговых окон.
        reference_models: Словарь reference-моделей.
        model_variant: Модель, которую сравниваем до/после retrain.
        random_state: Зерно генератора случайных чисел.

    Returns:
        DataFrame с колонками `POST_RETRAIN_COMPARISON_COLUMNS`.

    Raises:
        ValueError: Если `model_variant` отсутствует в `reference_models`.
    """

    if model_variant not in reference_models:
        raise ValueError(
            f"В reference_models отсутствует {model_variant}. Доступны: {sorted(reference_models)}"
        )

    reference_model = reference_models[model_variant]
    rows: List[Dict[str, object]] = []

    for window in windows:
        scenario = str(window["scenario"])
        x_window = pd.DataFrame(window["x_window"]).reset_index(drop=True)
        y_window = pd.Series(window["y_window"]).astype(int).reset_index(drop=True)

        # Оцениваем и before, и after на одном test-срезе окна, чтобы сравнение
        # отражало эффект retrain, а не разницу в выборках.
        x_train_w, x_test_w, y_train_w, y_test_w = _safe_window_split(
            x_window=x_window,
            y_window=y_window,
            random_state=random_state,
        )

        before_score = get_binary_score_vector(reference_model, x_test_w)
        before_metrics = evaluate_binary_predictions(y_test_w, before_score)
        rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario,
                "phase": "before_retrain",
                **before_metrics,
            }
        )

        retrained = make_model_pipeline(
            x_train=x_train_w,
            model_variant=model_variant,
            random_state=random_state,
        )
        retrained.fit(x_train_w, y_train_w)

        after_score = get_binary_score_vector(retrained, x_test_w)
        after_metrics = evaluate_binary_predictions(y_test_w, after_score)
        rows.append(
            {
                "dataset": dataset_name,
                "scenario": scenario,
                "phase": "after_retrain",
                **after_metrics,
            }
        )

    frame = pd.DataFrame(rows)
    return frame.loc[:, POST_RETRAIN_COMPARISON_COLUMNS]


def build_full_monitoring_cycle(
    dataset_name: str,
    df: pd.DataFrame,
    window_size: int = 240,
    random_state: int = SEED,
    alpha: float = DEFAULT_ALPHA,
    psi_alert: float = DEFAULT_PSI_ALERT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Запускает полный цикл ЛР05 и возвращает все обязательные артефакты.

    Учебный контекст:
        Используется в практических ноутбуках как основной orchestrator,
        чтобы студент видел полный workflow в одном вызове.

    Args:
        dataset_name: Имя датасета (`medical` или `finance`).
        df: Полный исходный датасет с колонкой `target`.
        window_size: Размер мониторингового окна.
        random_state: Зерно генератора случайных чисел.
        alpha: Порог `p-value` для drift-детекторов.
        psi_alert: Порог тревоги `PSI`.

    Returns:
        Кортеж из четырех DataFrame:
        - `drift_detection_audit`;
        - `monitoring_quality_audit`;
        - `retraining_policy_decisions`;
        - `post_retrain_comparison`.
    """

    x, y = split_xy(df)

    x_reference, y_reference, models, reference_metrics = prepare_reference_models(
        x=x,
        y=y,
        random_state=random_state,
    )

    windows = make_monitoring_windows(
        x_reference=x_reference,
        y_reference=y_reference,
        window_size=window_size,
        random_state=random_state,
    )

    drift_detection_audit = build_drift_detection_audit(
        dataset_name=dataset_name,
        x_reference=x_reference,
        windows=windows,
        alpha=alpha,
        psi_alert=psi_alert,
    )

    monitoring_quality_audit = build_monitoring_quality_audit(
        dataset_name=dataset_name,
        models=models,
        reference_metrics=reference_metrics,
        windows=windows,
    )

    retraining_policy_decisions = build_retraining_policy_decisions(
        drift_detection_audit=drift_detection_audit,
        monitoring_quality_audit=monitoring_quality_audit,
    )

    post_retrain_comparison = build_post_retrain_comparison(
        dataset_name=dataset_name,
        windows=windows,
        reference_models=models,
        model_variant="RandomForest",
        random_state=random_state,
    )

    return (
        drift_detection_audit,
        monitoring_quality_audit,
        retraining_policy_decisions,
        post_retrain_comparison,
    )
