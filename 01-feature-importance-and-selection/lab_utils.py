"""Утилиты для лабораторной работы по отбору признаков.

Файл используется всеми ноутбуками в теме `01-feature-importance-and-selection`.
Главная цель: дать единые и прозрачные строительные блоки, чтобы
фокус был на логике эксперимента, а не на технической рутине.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(path: str) -> pd.DataFrame:
    """Загружает CSV-датасет и проверяет наличие колонки `target`.

    Args:
        path: Путь к CSV-файлу.

    Returns:
        DataFrame с исходными колонками.

    Used in:
        Все ноутбуки (`todo` и `solutions`) как точка входа в данные.
    """

    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError(f"В датасете {path} отсутствует колонка 'target'.")
    return df


def split_xy(df: pd.DataFrame, target: str = "target") -> Tuple[pd.DataFrame, pd.Series]:
    """Разделяет признаки и таргет.

    Args:
        df: Исходный DataFrame.
        target: Имя целевой колонки.

    Returns:
        x: Таблица признаков.
        y: Целевая переменная (int).
    """

    x = df.drop(columns=[target]).copy()
    y = df[target].astype(int).copy()
    return x, y


def train_test_split_stratified(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Делит данные на train/test со стратификацией.

    Args:
        x: Признаки.
        y: Таргет.
        test_size: Доля test-части.
        random_state: Seed для воспроизводимости.

    Returns:
        x_train, x_test, y_train, y_test.
    """

    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def infer_feature_types(x: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Возвращает списки числовых и категориальных признаков.

    Args:
        x: Таблица признаков.

    Returns:
        numeric_features, categorical_features.
    """

    numeric_features = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in x.columns if col not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Строит единый препроцессор для числовых и категориальных признаков.

    Числовые признаки: median imputation + scaling.
    Категориальные признаки: most-frequent imputation + one-hot encoding.

    Args:
        x: Train-таблица признаков (нужна для определения типов колонок).

    Returns:
        ColumnTransformer.
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
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def transform_with_names(
    preprocessor: ColumnTransformer,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
):
    """Фитит preprocessor на train и трансформирует train/test.

    Args:
        preprocessor: ColumnTransformer из `build_preprocessor`.
        x_train: Train-признаки.
        x_test: Test-признаки.

    Returns:
        x_train_t: Трансформированный train.
        x_test_t: Трансформированный test.
        feature_names: Имена признаков после трансформаций.
    """

    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return x_train_t, x_test_t, feature_names


def to_dense(matrix):
    """Преобразует sparse-матрицу в dense (если требуется)."""

    if sparse.issparse(matrix):
        return matrix.toarray()
    return matrix


def get_binary_score_vector(model, x_data) -> Tuple[np.ndarray, str]:
    """Получает score-вектор для бинарной классификации в едином формате.

    Логика:
    - если есть `predict_proba` -> используем вероятность класса 1;
    - если есть `decision_function` -> применяем сигмоиду к margin;
    - иначе -> fallback на `predict` (0/1).

    Args:
        model: Обученная бинарная модель sklearn.
        x_data: Признаки для инференса.

    Returns:
        scores: np.ndarray формы (n_samples,) в диапазоне [0, 1].
        score_source: Источник score (`predict_proba`, `decision_function_sigmoid`, `predict`).

    Used in:
        Notebook 3 (threshold tuning, error analysis).
    """

    if hasattr(model, "predict_proba"):
        score = np.asarray(model.predict_proba(x_data)[:, 1], dtype=float)
        score = np.clip(score, 0.0, 1.0)
        return score, "predict_proba"

    if hasattr(model, "decision_function"):
        margin = np.asarray(model.decision_function(x_data), dtype=float)
        margin = np.clip(margin, -40.0, 40.0)
        score = 1.0 / (1.0 + np.exp(-margin))
        return score, "decision_function_sigmoid"

    fallback_pred = np.asarray(model.predict(x_data), dtype=float)
    fallback_pred = np.clip(fallback_pred, 0.0, 1.0)
    return fallback_pred, "predict"


def safe_roc_auc(model, x_test, y_test) -> float:
    """Считает ROC-AUC через unified score-вектор.

    Returns NaN, если ROC-AUC не может быть вычислен.
    """

    try:
        scores, _ = get_binary_score_vector(model, x_test)
        return float(roc_auc_score(y_test, scores))
    except Exception:
        return float("nan")


def compute_threshold_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    threshold: float,
) -> Dict[str, float]:
    """Считает метрики качества при произвольном пороге классификации.

    Args:
        y_true: Истинные метки (0/1).
        y_score: Score/вероятность класса 1.
        threshold: Порог отсечения (обычно 0..1).

    Returns:
        Dict с ключами: `threshold`, `accuracy`, `precision`, `recall`, `f1`.

    Used in:
        Notebook 3, самостоятельное задание по threshold tuning.
    """

    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred_arr = (y_score_arr >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }


def build_segment_error_table(
    dataset_name: str,
    segment_feature: str,
    segment_values: Sequence,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    n_bins: int = 4,
) -> pd.DataFrame:
    """Строит компактный сегментный анализ ошибок.

    Для числовых `segment_values` используется `qcut`, для категориальных —
    группировка по значениям.

    Args:
        dataset_name: Имя датасета (`medical`/`finance`).
        segment_feature: Название признака сегментации (`age`, `credit_score`, ...).
        segment_values: Значения признака сегментации для наблюдений.
        y_true: Истинные метки.
        y_pred: Предсказанные метки (0/1).
        n_bins: Число квантильных корзин для числового признака.

    Returns:
        DataFrame с колонками:
        `dataset`, `segment_feature`, `segment`, `n`,
        `error_rate`, `false_positive_rate`, `false_negative_rate`.

    Used in:
        Notebook 3, самостоятельное задание по error-by-segment.
    """

    segment_series = pd.Series(segment_values).reset_index(drop=True)
    y_true_s = pd.Series(y_true).astype(int).reset_index(drop=True)
    y_pred_s = pd.Series(y_pred).astype(int).reset_index(drop=True)

    if not (len(segment_series) == len(y_true_s) == len(y_pred_s)):
        raise ValueError("segment_values, y_true и y_pred должны иметь одинаковую длину.")

    if pd.api.types.is_numeric_dtype(segment_series):
        n_unique = int(segment_series.nunique(dropna=True))
        if n_unique >= 2:
            q = min(max(2, n_bins), n_unique)
            bins = pd.qcut(segment_series, q=q, duplicates="drop")
            segment_labels = bins.astype(str).fillna("missing")
        else:
            segment_labels = pd.Series(["all"] * len(segment_series))
    else:
        segment_labels = segment_series.astype(str).fillna("missing")

    frame = pd.DataFrame(
        {
            "dataset": dataset_name,
            "segment_feature": segment_feature,
            "segment": segment_labels,
            "y_true": y_true_s,
            "y_pred": y_pred_s,
        }
    )

    rows: List[Dict[str, object]] = []
    for segment_value, group in frame.groupby("segment", dropna=False):
        n = int(len(group))
        if n == 0:
            continue

        fp = int(((group["y_true"] == 0) & (group["y_pred"] == 1)).sum())
        fn = int(((group["y_true"] == 1) & (group["y_pred"] == 0)).sum())
        errors = int((group["y_true"] != group["y_pred"]).sum())

        rows.append(
            {
                "dataset": dataset_name,
                "segment_feature": segment_feature,
                "segment": str(segment_value),
                "n": n,
                "error_rate": float(errors / n),
                "false_positive_rate": float(fp / n),
                "false_negative_rate": float(fn / n),
            }
        )

    return pd.DataFrame(rows)


def evaluate_binary_model(model, x_train, y_train, x_test, y_test) -> Dict[str, float]:
    """Обучает бинарную модель и возвращает базовый набор метрик.

    Returns:
        Dict с ключами `accuracy`, `f1`, `roc_auc`, `fit_time_sec`.
    """

    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_time_sec = time.perf_counter() - start

    preds = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": safe_roc_auc(model, x_test, y_test),
        "fit_time_sec": float(fit_time_sec),
    }
    return metrics


def rank_desc(scores: np.ndarray) -> np.ndarray:
    """Возвращает ранги (1 = лучший) при сортировке по убыванию score."""

    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def append_ranking_rows(
    rows: List[Dict[str, object]],
    dataset_name: str,
    method_name: str,
    feature_names: List[str],
    scores: np.ndarray,
) -> None:
    """Добавляет строки в общий формат `feature_ranking`.

    Формат строки:
    `dataset`, `method`, `feature`, `score`, `rank`.
    """

    scores = np.asarray(scores, dtype=float)
    ranks = rank_desc(scores)
    for feature, score, rank in zip(feature_names, scores, ranks):
        rows.append(
            {
                "dataset": dataset_name,
                "method": method_name,
                "feature": feature,
                "score": float(score),
                "rank": int(rank),
            }
        )


def build_shortlist(
    feature_ranking: pd.DataFrame,
    dataset_name: str,
    methods: List[str],
    top_n: int = 12,
) -> List[str]:
    """Формирует shortlist по среднему рангу выбранных методов."""

    subset = feature_ranking[
        (feature_ranking["dataset"] == dataset_name)
        & (feature_ranking["method"].isin(methods))
    ].copy()
    agg = (
        subset.groupby("feature", as_index=False)["rank"]
        .mean()
        .sort_values("rank", ascending=True)
    )
    return agg.head(top_n)["feature"].tolist()


def metrics_to_long_rows(
    dataset_name: str,
    feature_set: str,
    model_name: str,
    metrics: Dict[str, float],
) -> List[Dict[str, object]]:
    """Преобразует словарь метрик в long-формат `model_results`."""

    return [
        {
            "dataset": dataset_name,
            "feature_set": feature_set,
            "model": model_name,
            "metric": "accuracy",
            "value": float(metrics["accuracy"]),
            "fit_time_sec": float(metrics["fit_time_sec"]),
        },
        {
            "dataset": dataset_name,
            "feature_set": feature_set,
            "model": model_name,
            "metric": "f1",
            "value": float(metrics["f1"]),
            "fit_time_sec": float(metrics["fit_time_sec"]),
        },
        {
            "dataset": dataset_name,
            "feature_set": feature_set,
            "model": model_name,
            "metric": "roc_auc",
            "value": float(metrics["roc_auc"]),
            "fit_time_sec": float(metrics["fit_time_sec"]),
        },
    ]