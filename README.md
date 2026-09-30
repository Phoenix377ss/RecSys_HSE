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
