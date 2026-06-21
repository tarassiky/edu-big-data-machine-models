#!/usr/bin/env python3
"""Smoke-check для ЛР 05.

Проверяет:
- структуру и идентичность workflow в `todo/solution`-ноутбуках;
- наличие обязательных теоретических разделов и шаблонных маркеров;
- контракты CSV и базовые инварианты итоговых артефактов.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "outputs"
MODULE_PATH = BASE_DIR / "lab_utils.py"

NOTEBOOKS_TO_EXECUTE = [
    BASE_DIR / "solutions/01_drift_detection_and_monitoring_solution.ipynb",
    BASE_DIR / "solutions/02_retraining_policy_solution.ipynb",
]

NOTEBOOK_PAIRS = [
    (
        BASE_DIR / "notebooks/01_drift_detection_and_monitoring_todo.ipynb",
        BASE_DIR / "solutions/01_drift_detection_and_monitoring_solution.ipynb",
    ),
    (
        BASE_DIR / "notebooks/02_retraining_policy_todo.ipynb",
        BASE_DIR / "solutions/02_retraining_policy_solution.ipynb",
    ),
]

THEORY_NOTEBOOK = BASE_DIR / "theory-notebooks/01_theory_drift_monitoring_retraining_policy.ipynb"

NOTEBOOK_STRUCTURE_RULES = {
    BASE_DIR / "notebooks/01_drift_detection_and_monitoring_todo.ipynb": {
        "required_markers": [
            "Перед началом",
            "Для кого эта ЛР",
            "Что получится в конце",
            "Что делать, если застрял",
            "Как работать с этим ноутбуком",
            "Мост из теории к практике",
            "Что уже знаем",
            "Что новое в этом шаге",
            "Зачем это в проекте",
            "Что делаем",
            "Зачем",
            "Вход",
            "Выход",
            "Проверь себя",
            "Ожидаемый формат ответа",
            "Мини-вывод",
            "Переход к следующему шагу",
            "Термины шага (на пальцах)",
            "Теоретическая карточка (перед выполнением)",
            "Как интерпретировать результат новичку",
            "Что вижу в таблице",
            "Что это значит",
            "Что делаю дальше",
            "Типичная ошибка новичка",
            "Короткий контрпример",
            "TODO(обязательно)",
            "drift_detection_audit.csv",
            "monitoring_quality_audit.csv",
        ],
        "forbidden_markers": [],
        "min_step_count": 4,
        "min_check_yourself_count": 4,
        "min_mini_summary_count": 4,
        "min_todo_count": 4,
        "min_theory_card_count": 2,
        "min_term_block_count": 4,
        "min_beginner_mistake_count": 4,
    },
    BASE_DIR / "solutions/01_drift_detection_and_monitoring_solution.ipynb": {
        "required_markers": [
            "Перед началом",
            "Для кого эта ЛР",
            "Что получится в конце",
            "Что делать, если застрял",
            "Как работать с этим ноутбуком",
            "Мост из теории к практике",
            "Что уже знаем",
            "Что новое в этом шаге",
            "Зачем это в проекте",
            "Что делаем",
            "Зачем",
            "Вход",
            "Выход",
            "Проверь себя",
            "Ожидаемый формат ответа",
            "Мини-вывод",
            "Переход к следующему шагу",
            "Термины шага (на пальцах)",
            "Теоретическая карточка (перед выполнением)",
            "Как интерпретировать результат новичку",
            "Что вижу в таблице",
            "Что это значит",
            "Что делаю дальше",
            "Типичная ошибка новичка",
            "Короткий контрпример",
            "drift_detection_audit.csv",
            "monitoring_quality_audit.csv",
        ],
        "forbidden_markers": ["TODO(обязательно)", "NotImplementedError"],
        "min_step_count": 4,
        "min_check_yourself_count": 4,
        "min_mini_summary_count": 4,
        "min_todo_count": 0,
        "min_theory_card_count": 2,
        "min_term_block_count": 4,
        "min_beginner_mistake_count": 4,
    },
    BASE_DIR / "notebooks/02_retraining_policy_todo.ipynb": {
        "required_markers": [
            "Перед началом",
            "Для кого эта ЛР",
            "Что получится в конце",
            "Что делать, если застрял",
            "Как работать с этим ноутбуком",
            "Мост из теории к практике",
            "Что уже знаем",
            "Что новое в этом шаге",
            "Зачем это в проекте",
            "Что делаем",
            "Зачем",
            "Вход",
            "Выход",
            "Проверь себя",
            "Ожидаемый формат ответа",
            "Мини-вывод",
            "Переход к следующему шагу",
            "Термины шага (на пальцах)",
            "Теоретическая карточка (перед выполнением)",
            "Как интерпретировать результат новичку",
            "Что вижу в таблице",
            "Что это значит",
            "Что делаю дальше",
            "Типичная ошибка новичка",
            "Короткий контрпример",
            "Что я теперь умею",
            "TODO(обязательно)",
            "retraining_policy_decisions.csv",
            "post_retrain_comparison.csv",
        ],
        "forbidden_markers": [],
        "min_step_count": 4,
        "min_check_yourself_count": 4,
        "min_mini_summary_count": 4,
        "min_todo_count": 4,
        "min_theory_card_count": 2,
        "min_term_block_count": 4,
        "min_beginner_mistake_count": 4,
    },
    BASE_DIR / "solutions/02_retraining_policy_solution.ipynb": {
        "required_markers": [
            "Перед началом",
            "Для кого эта ЛР",
            "Что получится в конце",
            "Что делать, если застрял",
            "Как работать с этим ноутбуком",
            "Мост из теории к практике",
            "Что уже знаем",
            "Что новое в этом шаге",
            "Зачем это в проекте",
            "Что делаем",
            "Зачем",
            "Вход",
            "Выход",
            "Проверь себя",
            "Ожидаемый формат ответа",
            "Мини-вывод",
            "Переход к следующему шагу",
            "Термины шага (на пальцах)",
            "Теоретическая карточка (перед выполнением)",
            "Как интерпретировать результат новичку",
            "Что вижу в таблице",
            "Что это значит",
            "Что делаю дальше",
            "Типичная ошибка новичка",
            "Короткий контрпример",
            "Что я теперь умею",
            "retraining_policy_decisions.csv",
            "post_retrain_comparison.csv",
            "before_retrain",
            "after_retrain",
        ],
        "forbidden_markers": ["TODO(обязательно)", "NotImplementedError"],
        "min_step_count": 4,
        "min_check_yourself_count": 4,
        "min_mini_summary_count": 4,
        "min_todo_count": 0,
        "min_theory_card_count": 2,
        "min_term_block_count": 4,
        "min_beginner_mistake_count": 4,
    },
}

THEORY_NOTEBOOK_RULES = {
    "required_markers": [
        "Перед началом",
        "Для кого эта ЛР",
        "Что получится в конце",
        "Что делать, если застрял",
        "Карманный словарь латиницы (первое знакомство)",
        "Что это на пальцах",
        "Официальный термин: `covariate`",
        "Официальный термин: `prior`",
        "Официальный термин: `combined`",
        "Официальный термин: `KS`",
        "Официальный термин: `chi2`",
        "Официальный термин: `H0`",
        "Официальный термин: `H1`",
        "Официальный термин: `p-value`",
        "Официальный термин: `alpha`",
        "Официальный термин: `power`",
        "Официальный термин: `PSI`",
        "Официальный термин: `confusion matrix`",
        "Официальный термин: `precision`",
        "Официальный термин: `recall`",
        "Официальный термин: `f1`",
        "Официальный термин: `threshold`",
        "Официальный термин: `Brier`",
        "Официальный термин: `ECE`",
        "Официальный термин: `expected_cost`",
        "Официальный термин: `random_state`",
        "Официальный термин: `stratify`",
        "Официальный термин: `data leakage`",
        "Зачем это в ЛР05",
        "Где видно в артефактах",
        "Типичная ошибка новичка",
        "Короткий контрпример",
        "Если объяснять человеку без техбэкграунда",
        "## Раздел 0. Почему эта тема важна именно новичку",
        "confusion matrix",
        "precision",
        "recall",
        "Brier",
        "ECE",
        "data leakage",
        "stratify",
        "## Раздел 1. Карта финальной темы",
        "## Раздел 2. Типы drift: covariate, prior, combined",
        "## Раздел 3. KS и chi2 как статистические детекторы",
        "Нулевой статистический модуль для новичка",
        "## Раздел 4. PSI как мера силы сдвига",
        "## Раздел 5. Связь drift с качеством и стоимостью",
        "## Раздел 6. Policy-правило решения observe/retrain",
        "## Раздел 7. Что меняет retrain и как это проверять",
        "## Раздел 8. Ограничения и риски",
        "## Раздел 9. Чеклист интерпретации мониторинга",
        "## Раздел 10. Проверь себя",
    ],
    "required_template_markers": [
        "### Идея",
        "### Формула",
        "### Мини-пример",
        "### Как читать результат/график",
        "### Где это в практическом ноутбуке",
        "Где это применится через 5 минут",
    ],
    "min_section_count": 11,
}

LATIN_TERMS_ORDER_CHECK = [
    "covariate",
    "prior",
    "combined",
    "KS",
    "chi2",
    "H0",
    "H1",
    "p-value",
    "alpha",
    "power",
    "PSI",
    "confusion matrix",
    "precision",
    "recall",
    "f1",
    "threshold",
    "Brier",
    "ECE",
    "expected_cost",
    "random_state",
    "stratify",
    "data leakage",
]

EXPECTED_COLUMNS = {
    "drift_detection_audit.csv": set(),
    "monitoring_quality_audit.csv": set(),
    "retraining_policy_decisions.csv": set(),
    "post_retrain_comparison.csv": set(),
}

SPEC = importlib.util.spec_from_file_location("lab05_utils", MODULE_PATH)
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)

EXPECTED_COLUMNS["drift_detection_audit.csv"] = set(lab.DRIFT_DETECTION_AUDIT_COLUMNS)
EXPECTED_COLUMNS["monitoring_quality_audit.csv"] = set(lab.MONITORING_QUALITY_AUDIT_COLUMNS)
EXPECTED_COLUMNS["retraining_policy_decisions.csv"] = set(lab.RETRAINING_POLICY_DECISIONS_COLUMNS)
EXPECTED_COLUMNS["post_retrain_comparison.csv"] = set(lab.POST_RETRAIN_COMPARISON_COLUMNS)


def run_solution_notebooks() -> None:
    """Исполняет solution-ноутбуки в отдельной временной директории."""
    with tempfile.TemporaryDirectory(prefix="lab05_verify_") as temp_dir:
        for notebook_path in NOTEBOOKS_TO_EXECUTE:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    "--output",
                    notebook_path.name,
                    "--output-dir",
                    temp_dir,
                    str(notebook_path.relative_to(BASE_DIR)),
                ],
                cwd=BASE_DIR,
                check=True,
            )


def load_output(name: str) -> pd.DataFrame:
    """Загружает обязательный CSV-артефакт и проверяет точные колонки."""
    path = OUTPUT_DIR / name
    if not path.exists():
        raise AssertionError(f"Не найден обязательный артефакт: {path}")

    frame = pd.read_csv(path)
    expected_columns = EXPECTED_COLUMNS[name]
    if set(frame.columns) != expected_columns:
        raise AssertionError(f"Неверные колонки в {name}: {list(frame.columns)}")
    return frame


def load_notebook_text(notebook_path: Path) -> tuple[str, str, str]:
    """Возвращает объединенный текст notebook по всем/markdown/code ячейкам."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    all_text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    markdown_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    code_text = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    return all_text, markdown_text, code_text


def load_notebook_cells(notebook_path: Path) -> list[dict]:
    """Возвращает список ячеек notebook для порядка-зависимых проверок."""
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    return notebook["cells"]


def extract_step_titles(markdown_text: str) -> list[str]:
    """Извлекает заголовки шагов вида `## Шаг ...` из markdown-текста."""
    return [line.strip() for line in markdown_text.splitlines() if line.strip().startswith("## Шаг")]


def assert_notebook_structure(notebook_path: Path, rules: dict) -> None:
    """Проверяет структурные правила для конкретного практического ноутбука."""
    all_text, markdown_text, _ = load_notebook_text(notebook_path)
    notebook_name = notebook_path.relative_to(BASE_DIR)

    for marker in rules["required_markers"]:
        if marker not in all_text:
            raise AssertionError(f"{notebook_name} должен содержать маркер `{marker}`.")

    for marker in rules["forbidden_markers"]:
        if marker in all_text:
            raise AssertionError(f"{notebook_name} не должен содержать маркер `{marker}`.")

    step_titles = extract_step_titles(markdown_text)
    if len(step_titles) < rules["min_step_count"]:
        raise AssertionError(
            f"{notebook_name}: ожидалось минимум {rules['min_step_count']} шага(ов), найдено {len(step_titles)}."
        )

    check_count = all_text.count("Проверь себя")
    if check_count < rules["min_check_yourself_count"]:
        raise AssertionError(
            f"{notebook_name}: ожидалось минимум {rules['min_check_yourself_count']} блока(ов) 'Проверь себя'."
        )

    mini_count = all_text.count("Мини-вывод")
    if mini_count < rules["min_mini_summary_count"]:
        raise AssertionError(
            f"{notebook_name}: ожидалось минимум {rules['min_mini_summary_count']} блока(ов) 'Мини-вывод'."
        )

    todo_count = all_text.count("TODO(обязательно)")
    if todo_count < rules["min_todo_count"]:
        raise AssertionError(
            f"{notebook_name}: ожидалось минимум {rules['min_todo_count']} TODO-блока(ов)."
        )

    theory_card_count = all_text.count("Теоретическая карточка (перед выполнением)")
    if theory_card_count < rules["min_theory_card_count"]:
        raise AssertionError(
            f"{notebook_name}: ожидалось минимум {rules['min_theory_card_count']} теоретических карточки."
        )

    term_block_count = all_text.count("Термины шага (на пальцах)")
    if term_block_count < rules["min_term_block_count"]:
        raise AssertionError(
            f"{notebook_name}: ожидалось минимум {rules['min_term_block_count']} блока(ов) 'Термины шага (на пальцах)'."
        )

    beginner_mistake_count = all_text.count("Типичная ошибка новичка")
    if beginner_mistake_count < rules["min_beginner_mistake_count"]:
        raise AssertionError(
            f"{notebook_name}: ожидалось минимум {rules['min_beginner_mistake_count']} блока(ов) 'Типичная ошибка новичка'."
        )


def assert_workflow_identity() -> None:
    """Проверяет совпадение последовательности шагов между todo и solution."""
    for todo_path, solution_path in NOTEBOOK_PAIRS:
        _, todo_md, _ = load_notebook_text(todo_path)
        _, solution_md, _ = load_notebook_text(solution_path)

        todo_steps = extract_step_titles(todo_md)
        solution_steps = extract_step_titles(solution_md)

        if todo_steps != solution_steps:
            raise AssertionError(
                f"Шаги в {todo_path.name} и {solution_path.name} должны совпадать по порядку и тексту."
            )


def assert_theory_notebook() -> None:
    """Проверяет минимальную структуру теоретического ноутбука."""
    all_text, markdown_text, _ = load_notebook_text(THEORY_NOTEBOOK)
    notebook_name = THEORY_NOTEBOOK.relative_to(BASE_DIR)

    for marker in THEORY_NOTEBOOK_RULES["required_markers"]:
        if marker not in all_text:
            raise AssertionError(f"{notebook_name} должен содержать раздел `{marker}`.")

    section_count = markdown_text.count("## Раздел")
    if section_count < THEORY_NOTEBOOK_RULES["min_section_count"]:
        raise AssertionError(
            f"{notebook_name} должен содержать минимум {THEORY_NOTEBOOK_RULES['min_section_count']} разделов."
        )

    for marker in THEORY_NOTEBOOK_RULES["required_template_markers"]:
        if marker not in all_text:
            raise AssertionError(
                f"{notebook_name} должен содержать шаблонный маркер `{marker}`."
            )

    if all_text.count("Короткий контрпример") < 4:
        raise AssertionError(
            f"{notebook_name} должен содержать минимум 4 блока с маркером 'Короткий контрпример'."
        )

    lower_markdown = markdown_text.lower()
    for term in LATIN_TERMS_ORDER_CHECK:
        marker = f"Официальный термин: `{term}`"
        marker_index = markdown_text.find(marker)
        if marker_index < 0:
            raise AssertionError(
                f"{notebook_name}: отсутствует обязательный маркер первичного объяснения `{marker}`."
            )

        plain_pattern = re.compile(rf"(?<![\\w`]){re.escape(term)}(?![\\w`])", flags=re.IGNORECASE)
        quoted_pattern = re.compile(rf"`{re.escape(term)}`", flags=re.IGNORECASE)

        plain_match = plain_pattern.search(lower_markdown)
        quoted_match = quoted_pattern.search(markdown_text)
        first_term_index_candidates = [
            m.start() for m in [plain_match, quoted_match] if m is not None
        ]
        if not first_term_index_candidates:
            continue
        first_term_index = min(first_term_index_candidates)

        if first_term_index < marker_index:
            raise AssertionError(
                f"{notebook_name}: термин `{term}` встречается до первичного объяснения "
                f"через маркер `Официальный термин: `{term}``."
            )


def assert_static_conditions() -> None:
    """Запускает статические проверки структуры до исполнения ноутбуков."""
    for notebook_path in NOTEBOOKS_TO_EXECUTE:
        if not notebook_path.exists():
            raise AssertionError(f"Не найден notebook для исполнения: {notebook_path}")

    for notebook_path, rules in NOTEBOOK_STRUCTURE_RULES.items():
        assert_notebook_structure(notebook_path, rules)

    assert_workflow_identity()
    assert_theory_notebook()


def assert_output_invariants() -> None:
    """Проверяет содержательные инварианты обязательных CSV-артефактов."""
    drift = load_output("drift_detection_audit.csv")
    quality = load_output("monitoring_quality_audit.csv")
    decisions = load_output("retraining_policy_decisions.csv")
    post = load_output("post_retrain_comparison.csv")

    for name, frame in [
        ("drift_detection_audit.csv", drift),
        ("monitoring_quality_audit.csv", quality),
        ("retraining_policy_decisions.csv", decisions),
        ("post_retrain_comparison.csv", post),
    ]:
        if frame.empty:
            raise AssertionError(f"{name} не должен быть пустым.")

    expected_datasets = sorted(lab.DATASET_PATHS)
    observed_datasets = sorted(set(drift["dataset"].unique()))
    if observed_datasets != expected_datasets:
        raise AssertionError(
            f"В drift_detection_audit.csv ожидались dataset={expected_datasets}, получено={observed_datasets}."
        )

    if not set(drift["scenario"].unique()).issubset(set(lab.DRIFT_SCENARIOS)):
        raise AssertionError("drift_detection_audit.csv содержит неожиданный scenario.")

    if not set(quality["model_variant"].unique()).issuperset({"LogisticRegression", "RandomForest"}):
        raise AssertionError("monitoring_quality_audit.csv должен содержать обе model_variant: LogisticRegression и RandomForest.")

    if not set(decisions["policy_action"].unique()).issubset({"observe", "retrain"}):
        raise AssertionError("retraining_policy_decisions.csv содержит неожиданный policy_action.")

    if (decisions["trigger_reason"].astype(str).str.len() == 0).any():
        raise AssertionError("retraining_policy_decisions.csv: trigger_reason не должен быть пустым.")

    phase_values = set(post["phase"].unique())
    if phase_values != {"before_retrain", "after_retrain"}:
        raise AssertionError(
            "post_retrain_comparison.csv должен содержать ровно две фазы: before_retrain и after_retrain."
        )


def main() -> None:
    assert_static_conditions()
    run_solution_notebooks()
    assert_output_invariants()
    print("Lab 05 smoke-check passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"[FAIL] Ошибка исполнения notebook: {exc}")
        sys.exit(1)
