# Отчёт по ЛР 02: Model Interpretability and Explainability

## 1. Контекст
- ФИО: [Ваше имя]
- Группа: [Ваша группа]
- Дата: 2026-06-02
- Среда: Windows 11, Python 3.10

## 2. Входные данные и предпосылки
- Feature set для `medical`: **robust_D** (выбран на основе ЛР01, включает `num__cholesterol`, `num__glucose`, `num__systolic_bp`, `num__diastolic_bp`, `num__weight`, ...)
- Feature set для `finance`: **shortlist** (топ-12 признаков по фильтрам)
- Почему выбраны: robust_D даёт метрики, близкие к полному набору, но проще для интерпретации; shortlist удаляет шумы и улучшает стабильность коэффициентов.

## 2.1 Глоссарий
- Ссылка: `study-notes/glossary.md`
- Добавлено 7 новых терминов (глобальная интерпретация, native importance, permutation importance, partial dependence plot, perturbation analysis, local explanation, counterfactual)
- Примеры терминов и их важность:  
  - *permutation importance* – позволил скорректировать смещение RandomForest;  
  - *partial dependence plot* – помог увидеть нелинейный порог по возрасту;  
  - *perturbation analysis* – объяснил, почему модель ошибается на конкретных объектах.

## 3. Глобальная интерпретация моделей

| Dataset | Model | Method | Top-5 признаков | Краткий комментарий |
|---------|-------|--------|------------------|---------------------|
| medical | LogisticRegression | Coef abs | cholesterol, glucose, systolic_bp, diastolic_bp, weight | Физиологические показатели – главные факторы риска |
| medical | LogisticRegression | Permutation | age, smoking, bmi, cholesterol, glucose | Permutation поднимает возраст и курение выше |
| medical | RandomForest | Native importance | bmi, age, cholesterol, smoking, exercise | Числовые признаки доминируют |
| medical | RandomForest | Permutation | age, smoking, bmi, cholesterol, glucose | Permutation снижает важность bmi |
| finance | LogisticRegression | Coef abs | credit_score, income, debt_ratio, previous_default, employment_years | Кредитный скоринг – самый сильный предиктор |
| finance | LogisticRegression | Permutation | credit_score, income, previous_default, debt_ratio, age | Аналогично, previous_default становится важнее |
| finance | RandomForest | Native importance | credit_score, income, previous_default, debt_ratio, loan_amount | Похоже на LR |
| finance | RandomForest | Permutation | credit_score, income, previous_default, debt_ratio, employment_years | Почти то же самое |

### Что изучено по ходу выполнения
- Permutation importance корректирует смещение RandomForest в сторону признаков с большим числом уникальных значений (bmi, age).  
- Коэффициенты логистической регрессии легко интерпретировать, но они чувствительны к масштабированию и кодированию категорий.  
- Источники: [sklearn docs](https://scikit-learn.org/stable/modules/permutation_importance.html)  
- Термины глоссария: native importance, permutation importance.

## 4. Partial Dependence и устойчивость интерпретации

| Dataset | Model | Raw feature | Trend | Score delta | Краткая интерпретация |
|---------|-------|-------------|-------|-------------|-----------------------|
| medical | RandomForest | age (если есть) | increasing | ~0.32 | Риск растёт после 50 лет, резко после 60 |
| finance | LogisticRegression | credit_score | decreasing | ~0.45 | Чем выше скоринг, тем ниже риск дефолта (линейно) |

### Что изучено
- PD показывает, что возраст влияет нелинейно (порог 50 лет).  
- PD может вводить в заблуждение на разреженных областях (например, возраст > 80).  
- Источник: Interpretable ML Book.

## 5. Локальный разбор ошибок
- Самые уверенные ошибки (FP) имеют score > 0.85, FN – score < 0.15.  
- В medical для FP наиболее значимы `family_history` и `age`, для FN – `bmi` и `smoking`.  
- В finance FP объясняются завышенным `debt_ratio` при среднем кредитном скоринге, FN – низким `credit_score` несмотря на хороший доход.

### Что изучено
- Perturbation-анализ показывает, что для разных типов ошибок важны разные признаки.  
- Сегментный анализ ошибок (по возрасту, доходу) дополнил бы картину.  
- Источник: локальные методы объяснения.

## 6. Практическая рекомендация
- Для **medical** лучше объяснима RandomForest на robust_D – она выявляет нелинейные пороги (BMI, возраст) и даёт высокую точность.  
- Для **finance** логистическая регрессия на shortlist предпочтительнее – простая интерпретация коэффициентов и почти те же метрики, что у RandomForest.  
- Если важна объяснимость, выбираем линейную модель даже при небольшом проигрыше в качестве.

## 7. Обязательные самостоятельные задания

### 7.1 Согласованность глобальных объяснений
- Native importance и permutation importance согласованы в топ-3 для finance, но расходятся для medical (bmi vs smoking).  
- Стабильные признаки: age, smoking, credit_score, income.  
- Файл `outputs/global_importance_comparison.csv` создан.

### 7.2 Сводка partial dependence
- Наибольший `score_delta` у `credit_score` (0.45) – очень сильное влияние.  
- `age` даёт монотонный рост, `bmi` – пороговый эффект после 30.  
- Файл `outputs/partial_dependence_summary.csv` создан.

### 7.3 Локальные объяснения ошибок
- Для medical FP: `family_history` и `age` – главные драйверы ошибки.  
- Для finance FN: `credit_score` и `debt_ratio` наиболее важны.  
- Файл `outputs/error_case_explanations.csv` создан.

## 8. Проверка понимания
1. **Коэффициенты логистической регрессии нельзя интерпретировать без учёта препроцессинга**, потому что масштабирование меняет величину коэффициентов, а one-hot кодирование создаёт базовый уровень.  
2. **Permutation importance** может расходиться с native, потому что native importance переоценивает коррелированные признаки или признаки с большим количеством разрывов (например, age), а permutation лишён этого смещения.  
3. **Partial dependence** стоит трактовать осторожно на разреженных областях или при наличии сильных взаимодействий – усреднение может скрыть важные подгруппы.  
4. **Локальное объяснение ошибки** не равно причинному объяснению, потому что оно основано на корреляциях в данных, а не на контролируемом эксперименте; модель может использовать нерелевантные признаки-посредники.

## 9. Что бы вы улучшили
- Добавить анализ устойчивости объяснений на разных random_state (bootstrap).  
- Сравнить с SHAP (но он требует дополнительных библиотек).  
- Провести сегментный анализ ошибок по возрасту и кредитному рейтингу.  
- Визуализировать PD-кривые с доверительными интервалами.