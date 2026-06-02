# Отчёт по ЛР 03: Overfitting, Validation and Honest Hyperparameter Tuning

## 1. Контекст
- Выполнила: Тарасова Даша
- Группа: ИВТ 1.2
- Дата: 2026-06-02


## 2. Candidate feature set из ЛР 01
- **medical**: full, shortlist, robust_D
- **finance**: full, shortlist, robust_D
- В ЛР03 эти наборы считаются гипотезами, потому что мы заново разделяем данные (train/validation/test) и переоцениваем все кандидаты – старый победитель мог быть случайным.

### 2.1 Какой feature set выбрала каждая модель
| Dataset | Model | Selected feature set | Train F1 | Validation F1 | F1 gap | Abs F1 gap | Причина выбора |
|---------|-------|---------------------|----------|---------------|--------|------------|----------------|
| medical | LogisticRegression | shortlist | 0.81 | 0.80 | 0.01 | 0.01 | best validation F1 |
| medical | RandomForest | robust_D | 0.92 | 0.85 | 0.07 | 0.07 | best validation F1 |
| finance | LogisticRegression | shortlist | 0.79 | 0.78 | 0.01 | 0.01 | best validation F1 |
| finance | RandomForest | robust_D | 0.88 | 0.80 | 0.08 | 0.08 | best validation F1 |

### 2.2 Глоссарий
- Ссылка: `study-notes/glossary.md`
- Добавлено 6 терминов
- Примеры: generalization gap, validation curve, GridSearchCV.

## 3. Где модель переобучается
| Dataset | Feature set | Model | Train F1 | Validation F1 | F1 gap | Train ROC-AUC | Validation ROC-AUC | ROC-AUC gap | Краткий вывод |
|---------|-------------|-------|----------|---------------|--------|---------------|--------------------|-------------|----------------|
| medical | full | RandomForest | 0.98 | 0.83 | 0.15 | 0.99 | 0.91 | 0.08 | Сильное переобучение |
| medical | robust_D | RandomForest | 0.92 | 0.85 | 0.07 | 0.96 | 0.93 | 0.03 | Умеренное переобучение |
| finance | full | RandomForest | 0.95 | 0.79 | 0.16 | 0.98 | 0.86 | 0.12 | Переобучение |
| finance | shortlist | LogisticRegression | 0.79 | 0.78 | 0.01 | 0.85 | 0.84 | 0.01 | Почти нет переобучения |

**Вывод**: RandomForest сильно переобучается на полном наборе, отбор признаков снижает gap. Линейная модель стабильна.

## 4. Что показали validation curves
| Dataset | Model | Feature set | Hyperparameter | Лучшее значение | При слишком слабой | При слишком сильной |
|---------|-------|-------------|----------------|-----------------|-------------------|---------------------|
| medical | RandomForest | robust_D | max_depth | 8 | Validation F1 низкий | Переобучение (gap растёт) |
| finance | LogisticRegression | shortlist | C | 1.0 | Недобучение | Стабильность |

## 5. Что выбрал GridSearchCV
| Dataset | Model | Feature set | Лучшая конфигурация | Mean CV F1 | Mean CV ROC-AUC | Комментарий |
|---------|-------|-------------|---------------------|------------|-----------------|-------------|
| medical | RandomForest | robust_D | {'max_depth':8, 'min_samples_leaf':5, 'class_weight':'balanced_subsample'} | 0.84 | 0.92 | Умеренная сложность |
| finance | LogisticRegression | shortlist | {'C':1.0, 'class_weight':'balanced'} | 0.78 | 0.85 | Простая модель |

## 6. Baseline vs Tuned на test
| Dataset | Model | Feature set | Variant | Accuracy | F1 | ROC-AUC | Fit time (sec) | Краткий вывод |
|---------|-------|-------------|---------|----------|-----|---------|----------------|----------------|
| medical | RandomForest | robust_D | baseline_default | 0.84 | 0.83 | 0.90 | 0.45 | Хороший базовый |
| medical | RandomForest | robust_D | tuned_best | 0.86 | 0.85 | 0.92 | 0.52 | Небольшое улучшение |
| finance | LogisticRegression | shortlist | baseline_default | 0.78 | 0.77 | 0.84 | 0.12 | Просто и быстро |
| finance | LogisticRegression | shortlist | tuned_best | 0.78 | 0.77 | 0.84 | 0.12 | Тюнинг не помог |

## 7. Практическая рекомендация
- **Medical**: использовать RandomForest на robust set D с tuned параметрами (max_depth=8). Это даёт F1~0.85 и ROC-AUC~0.92, переобучение умеренное.
- **Finance**: использовать LogisticRegression на shortlist – простая интерпретируемая модель с F1=0.77, не переобучается.
- Лучший feature set – не полный, а отобранный (robust_D для деревьев, shortlist для линейной модели).
- Человеку, смотрящему только на train-метрику, объясним: модель может идеально выучить шумы, но на новых данных ошибаться. Мы выбрали модель, которая показывает стабильный результат на валидации.

## 8. Проверка понимания
1. Нельзя выбирать гиперпараметры по test, потому что test должен имитировать новые, невидимые данные. Если мы подстроимся под test – получим завышенную оценку обобщения.
2. Train-метрика часто выше, потому что модель видела эти примеры и может выучить шумы. Validation – новые данные, поэтому качество обычно ниже.
3. Pipeline внутри CV гарантирует, что преобразования (масштабирование, one-hot) обучаются только на fold'ах train, а не на всей выборке. Но он не защищает от утечки, если мы до CV делали отбор признаков с участием validation.
4. Более сложная модель может переобучиться, особенно если данных мало. Тюнинг может не дать выигрыша, если базовая модель уже оптимальна (например, логистическая регрессия на линейно разделимых данных).
5. **Didactic shortcut**: мы использовали один и тот же validation для выбора feature set, validation curves и финального сравнения. В продакшне нужен отдельный selection split или nested CV.

## 9. Что бы вы улучшили
- Добавить третью модель (LinearSVC) и сравнить.
- Использовать RandomizedSearchCV для экономии времени.
- Провести nested CV для более честной оценки.
- Изучить влияние разных random_state.
