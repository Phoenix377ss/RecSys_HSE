from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from replay.data.dataset import (
    Dataset,
    FeatureHint,
    FeatureInfo,
    FeatureSchema,
    FeatureType,
)
from replay.data.dataset_utils import DatasetLabelEncoder
from replay.preprocessing.filters import (
    MinCountFilter,
    NumInteractionsFilter,
)
from replay.splitters.last_n_splitter import LastNSplitter
from replay.splitters.random_splitter import RandomSplitter
from replay.splitters.time_splitter import TimeSplitter
from tqdm import tqdm
import typer

from res_sys.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

KEEP_COLS = ["user_id", "item_id", "timestamp"]

# Правило T-ECD (карточка датасета t-tech/T-ECD): в данных есть временной шум,
# поэтому между последним событием пользователя в train и событиями в test нужен
# зазор не менее 12 часов, а последние 12 часов датасета использовать нельзя
# ("applies to both training and testing scenarios").
TIME_DELTA = pd.Timedelta(hours=12)
# ---------------------------------------------------------------------------
# 1. Загрузка
# ---------------------------------------------------------------------------
def load_raw(input_dir: Path) -> pd.DataFrame:
    events_dir = input_dir / "dataset/small/marketplace/events"
    event_files = sorted(events_dir.glob("*.pq"))
    if not event_files:
        raise FileNotFoundError(f"Нет parquet-файлов в {events_dir}")

    logger.info(f"Загрузка {len(event_files)} файлов событий (полный датасет)...")
    df = pd.concat(
        [pd.read_parquet(f) for f in tqdm(event_files, desc="events")],
        ignore_index=True,
    )

    for col in ["action_type", "subdomain", "os"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if "timestamp" in df.columns:
        df["day"] = df["timestamp"].dt.days

    # RePlay ожидает datetime в колонке timestamp для сплиттеров
    if np.issubdtype(df["timestamp"].dtype, np.timedelta64):
        df["timestamp"] = pd.Timestamp("2000-01-01") + df["timestamp"]

    logger.info(f"events shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. Статистика
# ---------------------------------------------------------------------------
def dataset_stats(df: pd.DataFrame, stage: str) -> dict:
    s = {
        "stage": stage,
        "events": len(df),
        "users": df["user_id"].nunique(),
        "items": df["item_id"].nunique(),
    }
    logger.info(
        f"[{stage}] events={s['events']:,} | users={s['users']:,} | items={s['items']:,}"
    )
    return s


# ---------------------------------------------------------------------------
# 3. Фильтрация через replay
# ---------------------------------------------------------------------------
def replay_filter(
    df: pd.DataFrame,
    max_user_interactions: int,
    min_user_count: int,
    min_item_count: int,
) -> pd.DataFrame:
    df = NumInteractionsFilter(num_interactions=max_user_interactions).transform(df)
    logger.info(f"NumInteractionsFilter -> {len(df):,}")

    df = MinCountFilter(num_entries=min_user_count, groupby_column="user_id").transform(df)
    logger.info(f"MinCountFilter(user) -> {len(df):,}")

    df = MinCountFilter(num_entries=min_item_count, groupby_column="item_id").transform(df)
    logger.info(f"MinCountFilter(item) -> {len(df):,}")

    return df


# ---------------------------------------------------------------------------
# 4. Сплиттеры
# ---------------------------------------------------------------------------
def drop_noisy_tail(df: pd.DataFrame, delta: pd.Timedelta = TIME_DELTA) -> pd.DataFrame:
    """Отбрасывает последние `delta` датасета.

    Правило T-ECD: финальные 12 часов содержат временной шум (логирование
    обрывается несинхронно, событиям хвоста нельзя доверять) — их не используем
    ни в train, ни в test.
    """
    cutoff = df["timestamp"].max() - delta
    kept = df[df["timestamp"] <= cutoff].reset_index(drop=True)
    logger.info(f"Шумный хвост: отброшено {len(df) - len(kept):,} событий (после {cutoff})")
    return kept


def enforce_time_gap(
    train: pd.DataFrame,
    test: pd.DataFrame,
    delta: pd.Timedelta = TIME_DELTA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Гарантирует per-user зазор >= `delta` между train и test (правило T-ECD).

    Из-за временного шума T-ECD порядок событий ближе 12 часов ненадёжен: test,
    начинающийся сразу за train-хвостом пользователя, означает утечку будущего.
    События test, нарушающие зазор, переносятся в train (данные не теряются:
    они старше оставшихся test-событий и утечкой не являются). Цикл нужен,
    потому что перенос может поднять train-хвост пользователя.
    """
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    moved_total = 0

    while True:
        last_train = train.groupby("user_id", sort=False)["timestamp"].max()
        cutoff = test["user_id"].map(last_train) + delta  # NaT у юзеров без train -> False
        violating = test["timestamp"] < cutoff
        if not violating.any():
            break
        train = pd.concat([train, test[violating]], ignore_index=True)
        test = test[~violating].reset_index(drop=True)
        moved_total += int(violating.sum())

    if moved_total:
        logger.info(f"12h-зазор: {moved_total:,} test-событий перенесено в train")
    return train, test


def split_data(df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    if mode == "random":
        train, test = RandomSplitter(
            test_size=0.3,
            drop_cold_items=True,
            drop_cold_users=True,
            seed=32,
            query_column="user_id",
        ).split(df)

    elif mode == "time":
        ts_median = df["timestamp"].median()
        train, test = TimeSplitter(
            time_threshold=ts_median,
            drop_cold_items=True,
            drop_cold_users=True,
            query_column="user_id",
        ).split(df)

    elif mode == "last_n":
        train, test = LastNSplitter(
            N=2,
            drop_cold_items=True,
            drop_cold_users=True,
            divide_column="user_id",
            query_column="user_id",
        ).split(df)
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    train, test = enforce_time_gap(train, test)

    logger.info(f"[{mode}] train={len(train):,} | test={len(test):,}")
    return train, test


# ---------------------------------------------------------------------------
# 5. Dataset + LabelEncoder
# ---------------------------------------------------------------------------
def encode_datasets(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schema = FeatureSchema(
        [
            FeatureInfo(
                column="user_id",
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.QUERY_ID,
            ),
            FeatureInfo(
                column="item_id",
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.ITEM_ID,
            ),
            FeatureInfo(
            column="timestamp",
            feature_type=FeatureType.NUMERICAL,
            feature_hint=FeatureHint.TIMESTAMP,
        )
        ]
    )

    train = train[[c for c in KEEP_COLS if c in train.columns]].copy()
    test  = test[[c for c in KEEP_COLS if c in test.columns]].copy()

    train_ds = Dataset(feature_schema=schema, interactions=train)
    test_ds = Dataset(feature_schema=schema, interactions=test)

    encoder = DatasetLabelEncoder()
    train_ds = encoder.fit_transform(train_ds)
    test_ds = encoder.transform(test_ds)

    return train_ds.interactions, test_ds.interactions


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
@app.command()
def main(
    input_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    max_user_interactions: int = 128,
    min_user_count: int = 5,
    min_item_count: int = 5,
    split_mode: str = "last_n",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(input_dir)
    df = drop_noisy_tail(df)
    before = dataset_stats(df, "before_filter")

    df = replay_filter(
        df,
        max_user_interactions=max_user_interactions,
        min_user_count=min_user_count,
        min_item_count=min_item_count,
    )
    after = dataset_stats(df, "after_filter")

    train, test = split_data(df, split_mode)
    train_stats = dataset_stats(train, "train")
    test_stats = dataset_stats(test, "test")

    train_enc, test_enc = encode_datasets(train, test)
    logger.info(f"Encoded: train={len(train_enc):,}, test={len(test_enc):,}")

    train_enc.to_parquet(output_dir / f"train_{split_mode}.parquet", index=False)
    test_enc.to_parquet(output_dir / f"test_{split_mode}.parquet", index=False)
    df.to_parquet(output_dir / "events_filtered.parquet", index=False)

    stats_df = pd.DataFrame([before, after, train_stats, test_stats])
    stats_df.to_csv(output_dir / "stats.csv", index=False)
    logger.info(f"Stats saved to {output_dir / 'stats.csv'}")

    logger.success(f"Готово. Всё сохранено в {output_dir}")


if __name__ == "__main__":
    app()
