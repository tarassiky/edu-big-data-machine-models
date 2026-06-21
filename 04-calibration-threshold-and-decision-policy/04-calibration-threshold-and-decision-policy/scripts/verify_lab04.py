#!/usr/bin/env python3
"""Smoke-check для ЛР 04.

Это внутренний QA-инструмент (не студенческий шаг).
Проверяет:
- структуру и идентичность workflow в todo/solution ноутбуках;
- структуру теоретического ноутбука;
- наличие обязательной визуализации;
- строгий контракт по данным (ноутбук 1 без test, ноутбук 2 с одной финальной проверкой);
- понятные переходы между шагами;
- отсутствие нежелательных англицизмов в markdown;
- контракты CSV и базовые инварианты артефактов.
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
    BASE_DIR / "solutions/01_calibration_basics_solution.ipynb",
    BASE_DIR / "solutions/02_threshold_policy_solution.ipynb",
]

NOTEBOOK_PAIRS = [
    (
        BASE_DIR / "notebooks/01_calibration_basics_todo.ipynb",
        BASE_DIR / "solutions/01_calibration_basics_solution.ipynb",
    ),
    (
        BASE_DIR / "notebooks/02_threshold_policy_todo.ipynb",
        BASE_DIR / "solutions/02_threshold_policy_solution.ipynb",
    ),
]

THEORY_NOTEBOOK = BASE_DIR / "theory-notebooks/01_theory_calibration_threshold_decision_policy.ipynb"

NOTEBOOK_STRUCTURE_RULES = {
    BASE_DIR / "notebooks/01_calibration_basics_todo.ipynb": {
        "required_markers": [
            "Как работать с этим ноутбуком",
            "Что делаем",
            "Зачем",
            "Вход",
            "Выход",
            "Проверь себя",
            "Мини-вывод",
            "TODO(обязательно)",
            "split='validation'",
        ],
        "forbidden_markers": [
            "split='test'",
            "test_scores",
        ],
        "min_step_count": 5,
        "min_check_yourself_count": 5,
        "min_mini_summary_count": 5,
        "min_todo_count": 5,
        "min_transition_count": 5,
    },
    BASE_DIR / "solutions/01_calibration_basics_solution.ipynb": {
        "required_markers": [
            "Как работать с этим ноутбуком",
            "Что делаем",
            "Зачем",
            "Вход",
            "Выход",
            "Проверь себя",
            "Мини-вывод",
            "split='validation'",
        ],
        "forbidden_markers": [
            "TODO(обязательно)",
            "NotImplementedError",
            "split='test'",
            "test_scores",
        ],
        "min_step_count": 5,
        "min_check_yourself_count": 5,
        "min_mini_summary_count": 5,
        "min_todo_count": 0,
        "min_transition_count": 5,
    },
    BASE_DIR / "notebooks/02_threshold_policy_todo.ipynb": {
        "required_markers": [
            "Как работать с этим ноутбуком",
            "Что делаем",
            "Зачем",
            "Вход",
            "Выход",
            "Проверь себя",
            "Мини-вывод",
            "TODO(обязательно)",
            "Одна финальная проверка на тестовой выборке `test` выбранного правила решения",
            "evaluate_policy_on_split",
        ],
        "forbidden_markers": [],
        "min_step_count": 5,
        "min_check_yourself_count": 5,
        "min_mini_summary_count": 5,
        "min_todo_count": 5,
        "min_transition_count": 5,
    },
    BASE_DIR / "solutions/02_threshold_policy_solution.ipynb": {
        "required_markers": [
            "Как работать с этим ноутбуком",
            "Что делаем",
            "Зачем",
            "Вход",
            "Выход",
            "Проверь себя",
            "Мини-вывод",
            "Одна финальная проверка на тестовой выборке `test` выбранного правила решения",
            "evaluate_policy_on_split",
            "test_scores",
        ],
        "forbidden_markers": [
            "TODO(обязательно)",
            "NotImplementedError",
        ],
        "min_step_count": 5,
        "min_check_yourself_count": 5,
        "min_mini_summary_count": 5,
        "min_todo_count": 0,
        "min_transition_count": 5,
    },
}

THEORY_NOTEBOOK_RULES = {
    "required_markers": [
        "## Раздел 1. Карта темы и связь с ЛР03/ЛР04",
        "## Раздел 2. Что такое вероятность модели и зачем нужна калибровка",
        "## Раздел 3. Калибровка и ранжирование: в чем разница",
        "## Раздел 4. Метрики вероятностей: Brier, LogLoss, ECE",
        "## Раздел 5. Методы калибровки: Platt scaling и isotonic regression",
        "## Раздел 6. Диаграмма надежности и разрыв калибровки",
        "## Раздел 7. Выбор порога как задача решения",
        "## Раздел 8. Сегментный аудит правила решения",
        "## Раздел 9. Типичные ошибки и как их избежать",
        "## Раздел 10. Проверь себя",
    ],
    "required_template_markers": [
        "### Идея",
        "### Формула",
        "### Мини-пример",
        "### Как читать результат/график",
        "### Где это в практическом ноутбуке",
    ],
    "min_section_count": 10,
    "min_plot_marker_count": 3,
}

BANNED_MARKDOWN_WORDS = [
    "strictly",
    "trade-off",
    "walkthrough",
    "skeleton",
    "test-check",
    "downstream",
]

EXPECTED_PLOT_MARKERS = [
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "plt.",
    "sns.",
]

EXPECTED_COLUMNS = {
    "calibration_audit.csv": set(),
    "threshold_policy_grid.csv": set(),
    "policy_test_report.csv": set(),
    "segment_policy_audit.csv": set(),
}

SPEC = importlib.util.spec_from_file_location("lab04_utils", MODULE_PATH)
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)

EXPECTED_COLUMNS["calibration_audit.csv"] = set(lab.CALIBRATION_AUDIT_COLUMNS)
EXPECTED_COLUMNS["threshold_policy_grid.csv"] = set(lab.THRESHOLD_POLICY_GRID_COLUMNS)
EXPECTED_COLUMNS["policy_test_report.csv"] = set(lab.POLICY_TEST_REPORT_COLUMNS)
EXPECTED_COLUMNS["segment_policy_audit.csv"] = set(lab.SEGMENT_POLICY_AUDIT_COLUMNS)


def run_solution_notebooks() -> None:
    with tempfile.TemporaryDirectory(prefix="lab04_verify_") as temp_dir:
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
    path = OUTPUT_DIR / name
    if not path.exists():
        raise AssertionError(f"Не найден обязательный артефакт: {path}")

    frame = pd.read_csv(path)
    expected_columns = EXPECTED_COLUMNS[name]
    if set(frame.columns) != expected_columns:
        raise AssertionError(f"Неверные колонки в {name}: {list(frame.columns)}")
    return frame


def load_notebook_text(notebook_path: Path) -> tuple[str, str, str]:
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


def extract_step_titles(markdown_text: str) -> list[str]:
    return [line.strip() for line in markdown_text.splitlines() if line.strip().startswith("## Шаг")]


def assert_notebook_structure(notebook_path: Path, rules: dict) -> None:
    all_text, markdown_text, code_text = load_notebook_text(notebook_path)
    notebook_name = notebook_path.relative_to(BASE_DIR)

    for marker in rules["required_markers"]:
        if marker not in all_text:
            raise AssertionError(f"{notebook_name} должен содержать маркер `{marker}`.")

    for marker in rules["forbidden_markers"]:
        if marker in all_text:
            raise AssertionError(f"{notebook_name} не должен содержать маркер `{marker}`.")

    step_count = markdown_text.count("## Шаг")
    if step_count < rules["min_step_count"]:
        raise AssertionError(
            f"{notebook_name} должен содержать минимум {rules['min_step_count']} шагов, найдено {step_count}."
        )

    check_yourself_count = all_text.count("Проверь себя")
    if check_yourself_count < rules["min_check_yourself_count"]:
        raise AssertionError(
            f"{notebook_name} должен содержать минимум {rules['min_check_yourself_count']} блоков `Проверь себя`."
        )

    mini_summary_count = all_text.count("Мини-вывод")
    if mini_summary_count < rules["min_mini_summary_count"]:
        raise AssertionError(
            f"{notebook_name} должен содержать минимум {rules['min_mini_summary_count']} блоков `Мини-вывод`."
        )

    todo_count = all_text.count("TODO(обязательно)")
    if todo_count < rules["min_todo_count"]:
        raise AssertionError(
            f"{notebook_name} должен содержать минимум {rules['min_todo_count']} блоков `TODO(обязательно)`."
        )

    transition_count = markdown_text.count("Переход к следующему шагу:")
    if transition_count < rules["min_transition_count"]:
        raise AssertionError(
            f"{notebook_name} должен содержать минимум {rules['min_transition_count']} явных переходов между шагами."
        )

    for marker in EXPECTED_PLOT_MARKERS:
        if marker not in code_text:
            raise AssertionError(f"{notebook_name} должен содержать визуализацию: маркер `{marker}`.")

    lower_markdown = markdown_text.lower()
    for word in BANNED_MARKDOWN_WORDS:
        if word in lower_markdown:
            raise AssertionError(
                f"{notebook_name} содержит нежелательное слово `{word}` в markdown."
            )

    step_blocks = re.split(r"(?=^## Шаг\s+\d+\.)", markdown_text, flags=re.M)
    step_blocks = [block for block in step_blocks if block.strip().startswith("## Шаг")]
    for index, block in enumerate(step_blocks, start=1):
        if "Переход к следующему шагу:" not in block:
            raise AssertionError(
                f"{notebook_name}: у шага {index} нет явной переходной связки к следующему шагу."
            )


def assert_workflow_identity() -> None:
    for todo_path, solution_path in NOTEBOOK_PAIRS:
        _, todo_markdown, _ = load_notebook_text(todo_path)
        _, solution_markdown, _ = load_notebook_text(solution_path)

        todo_steps = extract_step_titles(todo_markdown)
        solution_steps = extract_step_titles(solution_markdown)

        if todo_steps != solution_steps:
            raise AssertionError(
                "Workflow-шаги в todo/solution не совпадают: "
                f"{todo_path.name} vs {solution_path.name}."
            )


def assert_theory_notebook() -> None:
    if not THEORY_NOTEBOOK.exists():
        raise AssertionError(f"Не найден теоретический ноутбук: {THEORY_NOTEBOOK}")

    all_text, markdown_text, code_text = load_notebook_text(THEORY_NOTEBOOK)
    notebook_name = THEORY_NOTEBOOK.relative_to(BASE_DIR)

    for marker in THEORY_NOTEBOOK_RULES["required_markers"]:
        if marker not in markdown_text:
            raise AssertionError(f"{notebook_name} должен содержать раздел `{marker}`.")

    section_count = markdown_text.count("## Раздел")
    if section_count < THEORY_NOTEBOOK_RULES["min_section_count"]:
        raise AssertionError(
            f"{notebook_name} должен содержать минимум {THEORY_NOTEBOOK_RULES['min_section_count']} разделов."
        )

    for marker in THEORY_NOTEBOOK_RULES["required_template_markers"]:
        if marker not in markdown_text:
            raise AssertionError(f"{notebook_name} должен содержать шаблонный блок `{marker}`.")

    plot_hits = sum(1 for marker in EXPECTED_PLOT_MARKERS if marker in code_text)
    if plot_hits < THEORY_NOTEBOOK_RULES["min_plot_marker_count"]:
        raise AssertionError(
            f"{notebook_name} должен содержать обязательные графики и маркеры визуализации."
        )

    lower_markdown = markdown_text.lower()
    for word in BANNED_MARKDOWN_WORDS:
        if word in lower_markdown:
            raise AssertionError(
                f"{notebook_name} содержит нежелательное слово `{word}` в markdown."
            )

    if "TODO(обязательно)" in all_text:
        raise AssertionError(f"{notebook_name} не должен содержать TODO-блоки.")


def assert_static_conditions() -> None:
    required_notes = [
        BASE_DIR / "study-notes/calibration-vs-discrimination.md",
        BASE_DIR / "study-notes/brier-score-and-log-loss.md",
        BASE_DIR / "study-notes/threshold-and-cost-tradeoff.md",
        BASE_DIR / "study-notes/decision-policy-guardrails.md",
    ]
    for path in required_notes:
        if not path.exists():
            raise AssertionError(f"Отсутствует обязательная заметка: {path.name}")

    for notebook_path, rules in NOTEBOOK_STRUCTURE_RULES.items():
        assert_notebook_structure(notebook_path, rules)

    assert_workflow_identity()
    assert_theory_notebook()


def assert_output_invariants() -> None:
    calibration = load_output("calibration_audit.csv")
    threshold_grid = load_output("threshold_policy_grid.csv")
    policy_test = load_output("policy_test_report.csv")
    segment_audit = load_output("segment_policy_audit.csv")

    expected_datasets = sorted(lab.DATASET_PATHS)

    observed_calibration_datasets = sorted(calibration["dataset"].unique().tolist())
    if observed_calibration_datasets != expected_datasets:
        raise AssertionError("calibration_audit.csv должен содержать оба dataset.")

    if set(calibration["split"].unique()) != {"validation"}:
        raise AssertionError("calibration_audit.csv должен содержать только split=validation.")

    calibration_variant_counts = calibration.groupby("dataset")["variant"].nunique().to_dict()
    if any(count != 3 for count in calibration_variant_counts.values()):
        raise AssertionError("calibration_audit.csv должен содержать 3 variants на dataset.")

    if (calibration["ece"] < 0).any() or (calibration["ece"] > 1).any():
        raise AssertionError("ECE должен быть в диапазоне [0, 1].")

    if not threshold_grid["threshold"].between(0, 1).all():
        raise AssertionError("threshold_policy_grid.csv содержит threshold вне [0, 1].")

    variant_counts = threshold_grid.groupby("dataset")["variant"].nunique().to_dict()
    if any(count != 2 for count in variant_counts.values()):
        raise AssertionError("threshold_policy_grid.csv должен содержать ровно 2 variants на dataset.")

    policy_sizes = policy_test.groupby("dataset").size().to_dict()
    if any(size != 1 for size in policy_sizes.values()):
        raise AssertionError("policy_test_report.csv должен содержать ровно одну финальную policy на dataset.")

    for row in policy_test.itertuples(index=False):
        subset = threshold_grid[
            (threshold_grid["dataset"] == row.dataset)
            & (threshold_grid["variant"] == row.variant)
            & (threshold_grid["threshold"].round(6) == round(float(row.threshold), 6))
        ]
        if subset.empty:
            raise AssertionError(
                "policy_test_report.csv должен ссылаться на threshold из threshold_policy_grid.csv"
            )

    if (policy_test["cost_per_100"] < 0).any():
        raise AssertionError("cost_per_100 должен быть неотрицательным.")

    segment_datasets = sorted(segment_audit["dataset"].unique().tolist())
    if segment_datasets != expected_datasets:
        raise AssertionError("segment_policy_audit.csv должен содержать оба dataset.")

    if (segment_audit["expected_cost_per_100"] < 0).any():
        raise AssertionError("segment_policy_audit.csv содержит отрицательную стоимость.")


def main() -> None:
    assert_static_conditions()
    run_solution_notebooks()
    assert_output_invariants()
    print("Lab 04 smoke-check passed.")


if __name__ == "__main__":
    main()
