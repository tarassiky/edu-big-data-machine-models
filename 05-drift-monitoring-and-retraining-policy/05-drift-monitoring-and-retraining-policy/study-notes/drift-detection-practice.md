# Drift Detection: практический конспект для новичка

## Зачем это нужно в ЛР05
В этой работе цель не в том, чтобы "поймать красивый тест", а в том, чтобы принять безопасное и объяснимое решение по модели:
- оставить `observe`, если риск контролируемый;
- выбрать `retrain`, если риск стал практическим.

Главная цепочка:
`термин -> статистический сигнал -> метрика качества/стоимости -> policy-действие`.

## Как читать четыре CSV как единую историю
1. `drift_detection_audit.csv`
- Что смотрим: `detector`, `p_value`, `effect_size`, `drift_flag`.
- Вопрос: есть ли статистический сигнал и насколько он сильный.

2. `monitoring_quality_audit.csv`
- Что смотрим: `f1`, `expected_cost`, `delta_f1_vs_reference`, `delta_cost_vs_reference`.
- Вопрос: ухудшение заметно только в статистике или уже в прикладной цене ошибки.

3. `retraining_policy_decisions.csv`
- Что смотрим: `policy_action`, `trigger_reason`.
- Вопрос: какое правило реально сработало и почему.

4. `post_retrain_comparison.csv`
- Что смотрим: `phase` (`before_retrain`, `after_retrain`), `f1`, `expected_cost`.
- Вопрос: принес ли retrain измеримую пользу.

## Мини-словарь прямо в контексте практики
- `covariate`: сдвиг профиля признаков.
- `prior`: сдвиг доли целевого класса.
- `combined`: одновременный сдвиг признаков и доли класса.
- `p-value`: статистическая заметность отличия.
- `PSI`: сила сдвига в практическом смысле.
- `expected_cost`: средняя цена ошибок модели.

## Сквозной мини-кейс 1: один признак -> тест -> policy
1. Что произошло:
- Признак `income` в новом окне сместился к более высоким значениям.

2. Что показывают тесты:
- `detector = ks`;
- `p_value = 0.008` (отличие статистически заметно);
- `effect_size (PSI) = 0.29` (эффект сильный).

3. Что с качеством:
- `delta_f1_vs_reference = -0.06`;
- `delta_cost_vs_reference = +0.18`.

4. Какое действие:
- В `retraining_policy_decisions.csv` получаем `policy_action = retrain`;
- `trigger_reason` обычно содержит `f1_drop` и/или `cost_increase`.

## Сквозной мини-кейс 2: метрика не пугает, а стоимость уже растет
1. Что произошло:
- `f1` ухудшился слабо, визуально кажется "терпимо".

2. Что критично:
- `expected_cost` вырос существенно из-за дорогих `FN`.

3. Какое действие:
- Даже при умеренной деградации качества policy может выбрать `retrain` по триггеру `cost_increase`.

4. Что важно для отчета:
- Обязательно показывайте оба поля: `delta_f1_vs_reference` и `delta_cost_vs_reference`.

## Сквозной мини-кейс 3: ложный оптимизм из-за утечки
1. Что произошло:
- Метрики неожиданно "слишком хорошие".

2. Подозрение:
- Риск `data leakage`: информация из теста попала в обучение.

3. Действие:
- Такой запуск нельзя использовать для policy.
- Сначала устраняем утечку, потом пересчитываем все CSV.

## Правило решения в ЛР05 (фиксированное)
- `retrain`, если:
  - `drift_feature_share >= 0.30`, или
  - `delta_f1_vs_reference <= -0.05`, или
  - `delta_cost_vs_reference >= +0.15`.
- Иначе: `observe`.

## Источники
- SciPy Developers. `scipy.stats.ks_2samp`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html (дата обращения: 2026-04-09).
- SciPy Developers. `scipy.stats.chi2_contingency`: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html (дата обращения: 2026-04-09).
- scikit-learn Developers. Metrics and model selection: https://scikit-learn.org/stable/ (дата обращения: 2026-04-09).
- Siddiqi N. *Intelligent Credit Scoring*. Wiley, 2012.
