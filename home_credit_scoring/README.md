# Home Credit Default Risk Scoring

Пет-проект по классическому ML для кредитного скоринга на датасете **Home Credit Default Risk**.

Вся реализация находится в одном ноутбуке:

```text
home_credit_scoring.ipynb
```

## Задача

Предсказать вероятность дефолта клиента (`TARGET = 1`) по данным кредитной заявки и сравнить три модели:

- `LogisticRegression` из `scikit-learn`;
- `RandomForestClassifier` из `scikit-learn`;
- `XGBClassifier` из `XGBoost`.

## Что делает ноутбук

- загружает `application_train.csv`;
- делает EDA;
- добавляет несколько скоринговых признаков;
- обрабатывает пропуски и категориальные признаки через `sklearn Pipeline`;
- делит данные на train / validation / test;
- подбирает гиперпараметры через `Optuna`;
- оптимизация проходит по `ROC-AUC`;
- использует `timeout = 300` секунд для каждой модели;
- сравнивает модели по `ROC-AUC`, `PR-AUC` и `Gini`;
- сохраняет обученные модели и отчеты.

## Данные

датасет взят с Kaggle : Home Credit Default Risk 

```text
https://www.kaggle.com/competitions/home-credit-default-risk/data
```




## Метрики

- `ROC-AUC` — качество ранжирования клиентов по риску дефолта.
- `PR-AUC` — полезная метрика при дисбалансе классов.
- `Gini = 2 * ROC-AUC - 1` — популярная скоринговая метрика.
