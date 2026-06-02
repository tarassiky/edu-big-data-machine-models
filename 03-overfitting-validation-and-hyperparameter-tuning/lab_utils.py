"""Утилиты для ЛР 03 – исправленная версия без внешнего JSON."""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "01-feature-importance-and-selection" / "data"
if not DATA_DIR.exists():
    DATA_DIR = BASE_DIR / "data"  # fallback
LAB01_OUTPUT_DIR = BASE_DIR.parent / "01-feature-importance-and-selection" / "outputs"
OUTPUT_DIR = BASE_DIR / "outputs"
SEED = 42

DATASET_PATHS = {
    "medical": DATA_DIR / "medical.csv",
    "finance": DATA_DIR / "finance.csv",
}

VALIDATION_CURVE_GRIDS = {
    "LogisticRegression": ("C", [0.01, 0.1, 1.0, 10.0, 100.0]),
    "RandomForest": ("max_depth", [2, 4, 6, 8, None]),
}
MODEL_FEATURE_SET_DECISION_COLUMNS = [
    "dataset", "model", "selected_feature_set", "train_f1", "validation_f1",
    "f1_gap", "abs_f1_gap", "tie_break_reason"
]

# ------------------- загрузка данных -------------------
def load_dataset(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError(f"В датасете {path} отсутствует колонка 'target'.")
    return df

def load_course_datasets() -> Dict[str, pd.DataFrame]:
    return {name: load_dataset(path) for name, path in DATASET_PATHS.items() if path.exists()}

def split_xy(df: pd.DataFrame, target: str = "target"):
    return df.drop(columns=[target]).copy(), df[target].astype(int).copy()

def train_valid_test_split_stratified(x, y, test_size=0.2, valid_size=0.2, random_state=SEED):
    x_train_valid, x_test, y_train_valid, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_valid_size = valid_size / (1.0 - test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid, y_train_valid, test_size=relative_valid_size,
        random_state=random_state, stratify=y_train_valid
    )
    return x_train, x_valid, x_test, y_train, y_valid, y_test

# ------------------- препроцессинг -------------------
def infer_feature_types(x: pd.DataFrame):
    numeric = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [col for col in x.columns if col not in numeric]
    return numeric, categorical

def build_preprocessor(x: pd.DataFrame):
    numeric, categorical = infer_feature_types(x)
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    return ColumnTransformer([("num", num_pipe, numeric), ("cat", cat_pipe, categorical)])

def transform_with_names(preprocessor, x_train, x_valid, x_test):
    x_train_t = preprocessor.fit_transform(x_train)
    x_valid_t = preprocessor.transform(x_valid)
    x_test_t = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return x_train_t, x_valid_t, x_test_t, feature_names

def to_dense(matrix):
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)

def select_features(X, feature_names, selected_features):
    indices = [feature_names.index(f) for f in selected_features if f in feature_names]
    if not indices:
        return X
    return X[:, indices] if not hasattr(X, 'tocsc') else X[:, indices]

# ------------------- candidate feature set из ranking -------------------
def load_feature_ranking() -> pd.DataFrame:
    path = LAB01_OUTPUT_DIR / "feature_ranking.csv"
    if not path.exists():
        path = OUTPUT_DIR / "feature_ranking.csv"
    if not path.exists():
        raise FileNotFoundError("Не найден feature_ranking.csv. Выполните сначала ЛР01.")
    return pd.read_csv(path)

def list_feature_set_names_from_ranking(dataset_name: str, ranking_df: pd.DataFrame = None) -> List[str]:
    if ranking_df is None:
        ranking_df = load_feature_ranking()
    methods = ranking_df[ranking_df['dataset']==dataset_name]['method'].unique()
    # определяем какие методы есть: фильтры и wrapper
    filter_methods = ['VarianceThreshold', 'Correlation', 'MutualInfo', 'ANOVA']
    wrapper_methods = ['RFE', 'SFS_forward', 'L1_logreg', 'RF_importance', 'Permutation']
    present_filters = [m for m in filter_methods if m in methods]
    present_wrappers = [m for m in wrapper_methods if m in methods]
    sets = ['full']
    if present_filters:
        sets.append('shortlist')
    if present_wrappers:
        sets.append('robust_D')
    return sets

def get_feature_set_features(dataset_name: str, feature_set_name: str,
                             feature_names: List[str], ranking_df: pd.DataFrame = None) -> List[str]:
    if feature_set_name == 'full':
        return None
    if ranking_df is None:
        ranking_df = load_feature_ranking()
    sub = ranking_df[ranking_df['dataset']==dataset_name]
    if feature_set_name == 'shortlist':
        filter_methods = ['VarianceThreshold', 'Correlation', 'MutualInfo', 'ANOVA']
        sub = sub[sub['method'].isin(filter_methods)]
        if sub.empty:
            return feature_names[:12]
        agg = sub.groupby('feature')['rank'].mean().sort_values()
        top = agg.head(12).index.tolist()
        return [f for f in top if f in feature_names]
    elif feature_set_name == 'robust_D':
        wrapper_methods = ['RFE', 'SFS_forward', 'L1_logreg', 'RF_importance', 'Permutation']
        sub = sub[sub['method'].isin(wrapper_methods)]
        if sub.empty:
            return feature_names[:10]
        method_sets = []
        for m in wrapper_methods:
            top = sub[sub['method']==m].nsmallest(10, 'rank')['feature'].tolist()
            if top:
                method_sets.append(set(top))
        if not method_sets:
            return feature_names[:10]
        intersection = set.intersection(*method_sets)
        if len(intersection) < 3:
            from collections import Counter
            all_feats = [f for s in method_sets for f in s]
            cnt = Counter(all_feats)
            intersection = set([f for f, c in cnt.most_common(10)])
        return [f for f in intersection if f in feature_names]
    else:
        raise ValueError(f"Неизвестный feature_set: {feature_set_name}")

# ---------- модели и метрики ----------
def make_default_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2500, class_weight="balanced", random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=350, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    }

def make_tuning_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2500, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=350, random_state=SEED, n_jobs=-1)
    }

def make_param_grids():
    return {
        "LogisticRegression": {"model__C": [0.01, 0.1, 1.0, 10.0], "model__class_weight": [None, "balanced"]},
        "RandomForest": {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 5, 10],
                         "model__class_weight": [None, "balanced_subsample"]}
    }

def get_binary_score_vector(model, x_data):
    if hasattr(model, "predict_proba"):
        score = np.asarray(model.predict_proba(x_data)[:, 1], dtype=float)
        return np.clip(score, 0.0, 1.0), "predict_proba"
    if hasattr(model, "decision_function"):
        margin = np.asarray(model.decision_function(x_data), dtype=float)
        margin = np.clip(margin, -40.0, 40.0)
        score = 1.0 / (1.0 + np.exp(-margin))
        return score, "decision_function_sigmoid"
    fallback = np.asarray(model.predict(x_data), dtype=float)
    return np.clip(fallback, 0.0, 1.0), "predict"

def evaluate_fitted_model(model, x_data, y_true):
    y_pred = model.predict(x_data)
    y_score, _ = get_binary_score_vector(model, x_data)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score) if len(np.unique(y_score)) > 1 else float('nan')
    }

def measure_fit_and_split_metrics(model, x_train, y_train, x_valid, y_valid):
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_time = time.perf_counter() - start
    train_metrics = evaluate_fitted_model(model, x_train, y_train)
    valid_metrics = evaluate_fitted_model(model, x_valid, y_valid)
    return model, fit_time, train_metrics, valid_metrics

def generalization_gap(train_val, valid_val):
    return float(train_val - valid_val)

# ---------- выбор feature set по правилам ----------
def _feature_set_summary_for_model(audit_df, dataset_name, model_name):
    sub = audit_df[(audit_df['dataset']==dataset_name) & (audit_df['model']==model_name)]
    if sub.empty:
        raise ValueError(f"Нет данных для {dataset_name}/{model_name}")
    pivot = sub.pivot_table(index='feature_set', columns='split', values=['accuracy','f1','roc_auc'], aggfunc='mean')
    pivot.columns = [f"{col[1]}_{col[0]}" for col in pivot.columns]
    pivot = pivot.reset_index()
    pivot['f1_gap'] = pivot['train_f1'] - pivot['validation_f1']
    pivot['abs_f1_gap'] = pivot['f1_gap'].abs()
    pivot['full_penalty'] = (pivot['feature_set'] == 'full').astype(int)
    return pivot

def _select_feature_set_winner_row(feature_rows):
    remaining = feature_rows.copy()
    top_val_f1 = remaining['validation_f1'].max()
    remaining = remaining[np.isclose(remaining['validation_f1'], top_val_f1)]
    best_gap = remaining['abs_f1_gap'].min()
    remaining = remaining[np.isclose(remaining['abs_f1_gap'], best_gap)]
    best_penalty = remaining['full_penalty'].min()
    remaining = remaining[remaining['full_penalty'] == best_penalty]
    ordered = remaining.sort_values('feature_set', ascending=True)
    return ordered.iloc[0]

def explain_feature_set_tie_break(feature_rows):
    top_f1 = feature_rows['validation_f1'].max()
    candidates = feature_rows[np.isclose(feature_rows['validation_f1'], top_f1)]
    if len(candidates) == 1:
        return "best validation_f1"
    best_gap = candidates['abs_f1_gap'].min()
    candidates = candidates[np.isclose(candidates['abs_f1_gap'], best_gap)]
    if len(candidates) == 1:
        return "tie on validation_f1 -> min abs_f1_gap"
    best_penalty = candidates['full_penalty'].min()
    candidates = candidates[candidates['full_penalty'] == best_penalty]
    if len(candidates) == 1:
        return "tie on f1 and gap -> prefer non-full"
    return "tie on all -> lexicographic"

def build_model_feature_set_decisions(audit_df):
    rows = []
    for ds in audit_df['dataset'].unique():
        for model in audit_df['model'].unique():
            summary = _feature_set_summary_for_model(audit_df, ds, model)
            winner = _select_feature_set_winner_row(summary)
            rows.append({
                'dataset': ds,
                'model': model,
                'selected_feature_set': winner['feature_set'],
                'train_f1': winner['train_f1'],
                'validation_f1': winner['validation_f1'],
                'f1_gap': winner['f1_gap'],
                'abs_f1_gap': winner['abs_f1_gap'],
                'tie_break_reason': explain_feature_set_tie_break(summary)
            })
    return pd.DataFrame(rows, columns=MODEL_FEATURE_SET_DECISION_COLUMNS)

def build_generalization_selection_summary(audit_df):
    rows = []
    for ds in audit_df['dataset'].unique():
        for model in audit_df['model'].unique():
            summary = _feature_set_summary_for_model(audit_df, ds, model)
            for _, row in summary.iterrows():
                rows.append({
                    'dataset': ds,
                    'model': model,
                    'feature_set': row['feature_set'],
                    'train_f1': row['train_f1'],
                    'validation_f1': row['validation_f1'],
                    'f1_gap': row['f1_gap'],
                    'abs_f1_gap': row['abs_f1_gap'],
                    'train_roc_auc': row['train_roc_auc'],
                    'validation_roc_auc': row['validation_roc_auc'],
                    'roc_auc_gap': row['train_roc_auc'] - row['validation_roc_auc']
                })
    return pd.DataFrame(rows)

# ---------- pipeline и GridSearch ----------
class PreprocessedFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features=None):
        self.selected_features = selected_features
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            x_df = X
        else:
            x_df = pd.DataFrame(X).copy()
        self.preprocessor_ = build_preprocessor(x_df)
        self.preprocessor_.fit(x_df, y)
        self.feature_names_ = self.preprocessor_.get_feature_names_out().tolist()
        if self.selected_features is None:
            self.selected_indices_ = list(range(len(self.feature_names_)))
        else:
            pos = {name: i for i, name in enumerate(self.feature_names_)}
            self.selected_indices_ = [pos.get(f, -1) for f in self.selected_features]
        return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            x_df = X
        else:
            x_df = pd.DataFrame(X).copy()
        dense = to_dense(self.preprocessor_.transform(x_df))
        if self.selected_features is None:
            return dense
        cols = []
        for idx in self.selected_indices_:
            if idx == -1:
                cols.append(np.zeros((dense.shape[0], 1), dtype=float))
            else:
                cols.append(dense[:, [idx]])
        return np.hstack(cols) if cols else np.empty((dense.shape[0], 0), dtype=float)
    def get_feature_names_out(self, input_features=None):
        if self.selected_features is None:
            return np.asarray(self.feature_names_)
        return np.asarray(self.selected_features)

def build_model_pipeline(model, selected_features=None):
    return Pipeline([("features", PreprocessedFeatureSelector(selected_features=selected_features)), ("model", model)])

def top_gridsearch_rows(cv_results, dataset_name, feature_set_name, model_name, top_n=5):
    df = cv_results.sort_values(['rank_test_f1', 'mean_test_f1'], ascending=[True, False]).head(top_n)
    rows = []
    for rank, (idx, row) in enumerate(df.iterrows(), 1):
        params = row['params']
        if isinstance(params, dict):
            params_json = json.dumps(params, sort_keys=True, default=str)
        else:
            params_json = str(params)
        rows.append({
            'dataset': dataset_name, 'feature_set': feature_set_name, 'model': model_name,
            'rank': rank, 'params_json': params_json,
            'mean_cv_f1': row['mean_test_f1'], 'std_cv_f1': row['std_test_f1'],
            'mean_cv_roc_auc': row.get('mean_test_roc_auc', np.nan),
            'mean_cv_accuracy': row.get('mean_test_accuracy', np.nan),
            'mean_fit_time_sec': row['mean_fit_time']
        })
    return pd.DataFrame(rows)

def choose_validation_winner(validation_summary, dataset_name):
    sub = validation_summary[validation_summary['dataset'] == dataset_name].copy()
    sub['model_priority'] = sub['model'].map({'LogisticRegression': 0, 'RandomForest': 1}).fillna(99)
    ordered = sub.sort_values(['validation_f1', 'validation_roc_auc', 'model_priority'], ascending=[False, False, True])
    return ordered.iloc[0]

def fit_and_evaluate_pipeline(estimator, x_train, y_train, x_eval, y_eval):
    start = time.perf_counter()
    estimator.fit(x_train, y_train)
    fit_time = time.perf_counter() - start
    metrics = evaluate_fitted_model(estimator, x_eval, y_eval)
    metrics['fit_time_sec'] = fit_time
    return estimator, metrics

# ---------- заглушки для совместимости (если что-то ещё вызовет) ----------
def load_feature_sets(*args, **kwargs):
    # больше не нужен, но оставим, чтобы не сломался старый импорт
    return {}
def list_feature_set_names(dataset_name, *args, **kwargs):
    return list_feature_set_names_from_ranking(dataset_name)
def get_feature_set_features(dataset_name, feature_set_name, feature_names=None, *args, **kwargs):
    return get_feature_set_features(dataset_name, feature_set_name, feature_names)
def load_model_feature_set_decisions(*args, **kwargs):
    # будет загружено из CSV, созданного первым ноутбуком
    path = OUTPUT_DIR / "model_feature_set_decisions.csv"
    if not path.exists():
        raise FileNotFoundError("Сначала выполните ноутбук 01")
    return pd.read_csv(path)
def get_model_feature_set_decision(decisions_df, dataset_name, model_name):
    row = decisions_df[(decisions_df['dataset']==dataset_name) & (decisions_df['model']==model_name)]
    if row.empty:
        raise ValueError(f"Нет решения для {dataset_name}/{model_name}")
    return row.iloc[0]