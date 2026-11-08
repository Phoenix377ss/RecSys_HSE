# Рекомендательная система для e-commerce с объяснением рекомендаций

**Групповой проект · RecSys HSE**

---

## Тема проекта

**Рекомендательная система для e-commerce с объяснением рекомендаций**

### Описание

Пользователи в современном мире хотят видеть не только рекомендации товаров, но и их объяснение. Понимание, почему система предложила именно этот товар, повышает доверие к рекомендациям и ускоряет принятие решения о покупке. В исследовательском поле существует множество статей о внедрении LLM в сферу рекомендательных систем для решения подобных задач.

### Цели проекта

- улучшение качества модели рекомендаций
- улучшение обоснования рекомендаций
- улучшение пользовательского интерфейса

---

## Датасет

Используется [T-ECD](https://huggingface.co/datasets/t-tech/T-ECD) (T-Bank, KDD'26) — открытый датасет e-commerce взаимодействий. Берём домен **Marketplace** (small-версия) + справочник пользователей:

| Файл | Что содержит |
| ---- | ------------ |
| `dataset/small/marketplace/events/*.pq` | ~130 млн событий за 227 дней: `user_id, item_id, action_type, subdomain, os, timestamp` |
| `dataset/small/marketplace/items.pq` | каталог ~2.3 млн товаров: `brand_id, category, subcategory, price, embedding` |
| `dataset/full/users.pq` | ~44 млн пользователей: `socdem_cluster, region` |

Типы действий: `view` (показ, **seen**) → `click` (клик по карточке, **tap**) → `clickout` (уход на внешний сайт); отдельно `like`.
Ключевая величина для рекомендаций — **конверсия seen→tap** = `click / view`.

Скачивание (~3.6 ГБ, кэш HF хранится в `data/.hf_cache`):

```bash
make download          # или: python res_sys/download.py
```

---

## Быстрый старт

```bash
uv venv && uv pip install -r requirements.txt   # или python -m venv .venv && pip install -r requirements.txt
make download
jupyter lab notebooks/
```

Ноутбуки:
- `1_EDA.ipynb` — разведочный анализ (события, пользователи, соцдем-кластеры, товары).
- `2_baseline_top_pop.ipynb` — бейзлайн: top-pop товары по сглаженной конверсии seen→tap в разрезе соцдем-кластеров (`CTR = (clicks + α·CTR_global) / (views + α)`) против ALS (implicit, factors=64, 20 итераций, обучается на кликах train); оценка Recall/Precision/HitRate@20 на последних 7 днях.

---

## Задачи и план работ

### Ресерч

#### Сбор и очистка данных

перебор публичных датасетов

#### Подготовка данных

предобработка, построение портрета пользователя из истории взаимодействий: категории, бренды, ценовой диапазон

#### Бейзлайновые модели

например, TopPop, BayesMean

#### Модели машинного обучения

например, ALS

#### Модели глубинного обучения

например, графовые модели, SASRec

### Сервис

- Сервис с возможностью получить рекомендации с объяснением
- Персональная лента/подборка товаров на главной странице
- Личный кабинет с историей покупок/просмотров и возможностью ведения списка избранного
- Возможность общаться с LLM, чтобы уточнить критерии подбора и получить сравнение товаров

---

## Команда


| ФИО                             | Контакт          |
| ------------------------------- | ---------------- |
| *Зеленин Василий Игоревич*      | *@H47DS0ig*      |
| *Пискунов Сергей Александрович* | *@phoenix377*    |
| *Клоконос Дарья Вячеславовна*   | *@KlokonosDarya* |


---

## Куратор


|             |                  |
| ----------- | ---------------- |
| **ФИО**     | *Хамрин Роман*   |
| **Контакт** | *@roman_khamrin* |


---

## Структура репозитория

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         res_sys and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── res_sys   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes res_sys a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```
