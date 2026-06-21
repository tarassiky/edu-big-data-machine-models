# Outputs

Сюда ноутбуки ЛР 04 сохраняют промежуточные и итоговые артефакты:
- `calibration_audit.csv` — сравнение `uncalibrated` vs `calibrated` по quality + calibration-метрикам.
- `threshold_policy_grid.csv` — перебор порогов на `validation` с расчетом метрик и expected cost.
- `policy_test_report.csv` — финальная проверка на `test` только для выбранной policy.
- `segment_policy_audit.csv` — срезы ошибок и стоимости по сегментам.

Generated CSV не должны коммититься в git (см. корневой `.gitignore`).
