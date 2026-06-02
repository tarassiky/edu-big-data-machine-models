"""Утилиты для ЛР02: интерпретируемость и explainability."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ---------- Функции из ЛР01 (адаптированные) ----------
def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "target" not in df.columns:
        raise ValueError(f"В датасете {path} отсутствует колонка 'target'.")
    return df

def split_xy(df: pd.DataFrame, target: str = "target"):
    return df.drop(columns=[target]).copy(), df[target].astype(int).copy()

def train_test_split_stratified(x, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

def infer_feature_types(x: pd.DataFrame):
    numeric = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [col for col in x.columns if col not in numeric]
    return numeric, categorical

def build_preprocessor(x: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    numeric_features, categorical_features = infer_feature_types(x)
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    return ColumnTransformer([("num", numeric_transformer, numeric_features),
                              ("cat", categorical_transformer, categorical_features)])

def transform_with_names(preprocessor, x_train, x_test):
    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out().tolist()
    return x_train_t, x_test_t, feature_names

def get_feature_set_by_name(feature_set_name, feature_names, ranking_df, dataset_name, top_k=12):
    """Возвращает список признаков для заданного feature_set (full, shortlist, robust_D)."""
    if feature_set_name == "full":
        return feature_names
    elif feature_set_name == "shortlist":
        # средний ранг по фильтрам
        filter_methods = ['VarianceThreshold', 'Correlation', 'MutualInfo', 'ANOVA']
        sub = ranking_df[(ranking_df['dataset']==dataset_name) & (ranking_df['method'].isin(filter_methods))]
        agg = sub.groupby('feature')['rank'].mean().sort_values()
        return agg.head(top_k).index.tolist()
    elif feature_set_name == "robust_D":
        wrapper_methods = ['RFE', 'SFS_forward', 'L1_logreg', 'RF_importance', 'Permutation']
        sub = ranking_df[(ranking_df['dataset']==dataset_name) & (ranking_df['method'].isin(wrapper_methods))]
        method_sets = []
        for m in wrapper_methods:
            top = sub[sub['method']==m].nsmallest(10, 'rank')['feature'].tolist()
            if top:
                method_sets.append(set(top))
        if not method_sets:
            return feature_names[:top_k]
        intersection = set.intersection(*method_sets)
        if len(intersection) < 3:
            from collections import Counter
            all_feats = [f for s in method_sets for f in s]
            cnt = Counter(all_feats)
            intersection = set([f for f, c in cnt.most_common(10)])
        return list(intersection)
    else:
        raise ValueError(f"Неизвестный feature_set: {feature_set_name}")

def select_features(X, feature_names, selected_features):
    indices = [feature_names.index(f) for f in selected_features if f in feature_names]
    if not indices:
        return X
    if hasattr(X, 'tocsc'):
        return X[:, indices]
    return X[:, indices]

# ---------- Новые функции для интерпретации ----------
def compute_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    """Безопасное вычисление permutation importance."""
    return permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state, scoring='roc_auc')

def get_coef_importance(model, feature_names):
    """Абсолютные значения коэффициентов для линейных моделей."""
    coef = np.abs(model.coef_[0])
    return dict(zip(feature_names, coef))

def get_tree_importance(model, feature_names):
    """Важности из RandomForest."""
    return dict(zip(feature_names, model.feature_importances_))

def append_global_importance_rows(rows, dataset, model_name, feature_set, method, feature_scores):
    """Заполняет rows для global_importance_comparison.csv."""
    for feat, score in feature_scores.items():
        rows.append({
            'dataset': dataset,
            'model': model_name,
            'feature_set': feature_set,
            'method': method,
            'feature': feat,
            'score': float(score),
            'rank': 0  # позже пересчитаем
        })
    return rows

def rank_importance(df_group):
    """Добавляет ранги внутри dataset/model/feature_set/method."""
    df_group = df_group.sort_values('score', ascending=False)
    df_group['rank'] = range(1, len(df_group)+1)
    return df_group

def compute_partial_dependence(model, X, feature_idx, feature_name, grid_resolution=30):
    """Вычисляет PD для одного признака (усреднение по всем остальным)."""
    import numpy as np
    X_temp = X.copy()
    if hasattr(X_temp, 'toarray'):
        X_temp = X_temp.toarray()
    feature_values = X_temp[:, feature_idx]
    grid = np.linspace(np.percentile(feature_values, 1), np.percentile(feature_values, 99), grid_resolution)
    scores = []
    for val in grid:
        X_temp[:, feature_idx] = val
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_temp)[:, 1]
        else:
            proba = model.decision_function(X_temp)
            proba = 1/(1+np.exp(-np.clip(proba, -40, 40)))
        scores.append(np.mean(proba))
    return grid, np.array(scores)

def summarize_pd(grid, scores):
    """Извлекает тренд и score_delta из PD-кривой."""
    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    score_delta = score_max - score_min
    # простой тренд: возрастающий/убывающий/нелинейный
    first = scores[0]
    last = scores[-1]
    if last - first > 0.05:
        trend = "increasing"
    elif first - last > 0.05:
        trend = "decreasing"
    else:
        trend = "nonlinear"
    return {
        'grid_min': float(grid[0]),
        'grid_max': float(grid[-1]),
        'score_min': score_min,
        'score_max': score_max,
        'score_delta': score_delta,
        'trend': trend
    }

def perturbation_analysis(model, x_instance, feature_names, baseline_pred, n_perturbations=100, noise_scale=0.1):
    """Оценивает влияние каждого признака на предсказание для одного объекта.
       Возвращает словарь: feature -> среднее изменение score при возмущении.
    """
    import copy
    if hasattr(x_instance, 'toarray'):
        x_instance = x_instance.toarray().flatten()
    else:
        x_instance = x_instance.flatten()
    effects = {}
    for i, fname in enumerate(feature_names):
        orig_val = x_instance[i]
        changes = []
        for _ in range(n_perturbations):
            perturbed = copy.deepcopy(x_instance)
            # добавляем шум, пропорциональный std признака (если есть)
            std = max(0.1, abs(orig_val) * noise_scale)
            perturbed[i] = orig_val + np.random.normal(0, std)
            if hasattr(model, "predict_proba"):
                new_pred = model.predict_proba(perturbed.reshape(1, -1))[0, 1]
            else:
                new_pred = model.decision_function(perturbed.reshape(1, -1))[0]
                new_pred = 1/(1+np.exp(-new_pred))
            changes.append(abs(new_pred - baseline_pred))
        effects[fname] = np.mean(changes)
    return effects

def get_binary_score_vector(model, x_data):
    """Получает score-вектор для бинарной классификации в едином формате.
    Логика:
    - если есть `predict_proba` -> используем вероятность класса 1;
    - если есть `decision_function` -> применяем сигмоиду к margin;
    - иначе -> fallback на `predict` (0/1).
    """
    import numpy as np
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