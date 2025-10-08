#!/usr/bin/env python3
"""Generate enriched parquet files without loading the entire dataset in memory."""

import os
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm


CFG = {
    "SEED": 42,
    "MAX_SEQ_LEN": 128,
    "TARGET_ENC_PRIOR": 50.0,
    "TARGET_ENC_SPLITS": 5,
    "CHUNK_SIZE": 200_000,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


seed_everything(CFG["SEED"])


def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    alt = os.path.join("data", os.path.basename(path))
    if os.path.exists(alt):
        return alt
    raise FileNotFoundError(f"Cannot locate file: {path}")


def _normalize_categorical(series: pd.Series) -> pd.Series:
    return series.fillna("UNK").astype(str)


def build_label_maps(paths: Iterable[str], columns: List[str], chunk_size: int) -> Dict[str, Dict[str, int]]:
    unique_values: Dict[str, set] = {col: set() for col in columns}
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=columns, batch_size=chunk_size):
            pdf = batch.to_pandas()
            for col in columns:
                unique_values[col].update(_normalize_categorical(pdf[col]))

    label_maps: Dict[str, Dict[str, int]] = {}
    for col, values in unique_values.items():
        values.add("UNK")
        sorted_vals = sorted(values)
        label_maps[col] = {val: idx for idx, val in enumerate(sorted_vals)}
    return label_maps


def encode_series(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
    encoded = _normalize_categorical(series).map(mapping)
    unk_idx = mapping.get("UNK", 0)
    return encoded.fillna(unk_idx).astype(np.int32)


def compute_sequence_features(series: pd.Series, seq_col: str) -> Dict[str, np.ndarray]:
    max_len = CFG["MAX_SEQ_LEN"]
    window = min(10, max_len)
    length = np.zeros(len(series), dtype=np.float32)
    mean = np.zeros(len(series), dtype=np.float32)
    std = np.zeros(len(series), dtype=np.float32)
    last = np.zeros(len(series), dtype=np.float32)
    recent = np.zeros(len(series), dtype=np.float32)

    seq_iter = series.fillna("").astype(str).values
    for idx, seq_str in enumerate(seq_iter):
        if seq_str:
            arr = np.fromstring(seq_str, sep=",", dtype=np.float32)
            if arr.size == 0:
                arr = np.zeros(1, dtype=np.float32)
        else:
            arr = np.zeros(1, dtype=np.float32)

        if arr.size > max_len:
            arr = arr[-max_len:]

        length[idx] = float(arr.size)
        mean[idx] = float(arr.mean()) if arr.size else 0.0
        std[idx] = float(arr.std()) if arr.size > 1 else 0.0
        last[idx] = float(arr[-1]) if arr.size else 0.0
        recent[idx] = float(arr[-window:].mean()) if arr.size else 0.0

    prefix = f"{seq_col}"
    return {
        f"{prefix}_length": length,
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_last": last,
        f"{prefix}_recent_mean": recent,
    }


def accumulate_target_stats(
    train_path: str,
    cat_cols: List[str],
    target_col: str,
    encoders: Dict[str, Dict[str, int]],
    chunk_size: int,
) -> Tuple[
    Dict[str, Dict[int, Tuple[float, int]]],
    Dict[str, List[Dict[int, Tuple[float, int]]]],
    float,
]:
    total_stats: Dict[str, Dict[int, List[float]]] = {
        col: defaultdict(lambda: [0.0, 0]) for col in cat_cols
    }
    fold_stats: Dict[str, List[Dict[int, List[float]]]] = {
        col: [defaultdict(lambda: [0.0, 0]) for _ in range(CFG["TARGET_ENC_SPLITS"])]
        for col in cat_cols
    }

    total_sum = 0.0
    total_count = 0
    pf = pq.ParquetFile(train_path)
    rng = np.random.default_rng(CFG["SEED"])

    for batch in tqdm(
        pf.iter_batches(batch_size=chunk_size),
        desc="Accumulating target stats",
    ):
        df = batch.to_pandas()
        targets = df[target_col].astype(np.float32)
        fold_ids = rng.integers(0, CFG["TARGET_ENC_SPLITS"], size=len(df))
        df[target_col] = targets
        df["fold"] = fold_ids

        total_sum += float(targets.sum())
        total_count += int(len(df))

        for col in cat_cols:
            df[col] = encode_series(df[col], encoders[col])

            grouped = df.groupby(col)[target_col].agg(["sum", "count"])
            for val, row in grouped.iterrows():
                stats = total_stats[col][int(val)]
                stats[0] += float(row["sum"])
                stats[1] += int(row["count"])

            fold_grouped = df.groupby(["fold", col])[target_col].agg(["sum", "count"]).reset_index()
            for _, row in fold_grouped.iterrows():
                val = int(row[col])
                fold = int(row["fold"])
                stats = fold_stats[col][fold][val]
                stats[0] += float(row["sum"])
                stats[1] += int(row["count"])

            unk_idx = encoders[col]["UNK"]
            _ = total_stats[col][unk_idx]
            for fold_dict in fold_stats[col]:
                _ = fold_dict[unk_idx]

    global_mean = total_sum / max(total_count, 1)

    # Convert accumulated lists to tuples to prevent accidental mutation downstream.
    total_stats_typed: Dict[str, Dict[int, Tuple[float, int]]] = {
        col: {val: (float(stats[0]), int(stats[1])) for val, stats in col_dict.items()}
        for col, col_dict in total_stats.items()
    }
    fold_stats_typed: Dict[str, List[Dict[int, Tuple[float, int]]]] = {}
    for col, folds in fold_stats.items():
        fold_list: List[Dict[int, Tuple[float, int]]] = []
        for fold_dict in folds:
            fold_list.append({val: (float(stats[0]), int(stats[1])) for val, stats in fold_dict.items()})
        fold_stats_typed[col] = fold_list

    return total_stats_typed, fold_stats_typed, global_mean


def build_te_tables(
    total_stats: Dict[str, Dict[int, Tuple[float, int]]],
    fold_stats: Dict[str, List[Dict[int, Tuple[float, int]]]],
    global_mean: float,
    prior: float,
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, Dict[int, float]],
]:
    fold_tables: Dict[str, pd.DataFrame] = {}
    count_tables: Dict[str, pd.DataFrame] = {}
    global_te_lookup: Dict[str, Dict[int, float]] = {}

    for col, stats_dict in total_stats.items():
        data_rows: List[Tuple[int, int, float]] = []
        cnt_rows: List[Tuple[int, float]] = []
        global_te_lookup[col] = {}

        for val, (tot_sum, tot_cnt) in stats_dict.items():
            base_te = (tot_sum + global_mean * prior) / (tot_cnt + prior)
            global_te_lookup[col][val] = base_te
            cnt_rows.append((val, float(tot_cnt)))

            for fold_idx, fold_dict in enumerate(fold_stats[col]):
                fold_sum, fold_cnt = fold_dict.get(val, (0.0, 0))
                leave_sum = tot_sum - fold_sum
                leave_cnt = tot_cnt - fold_cnt
                if leave_cnt <= 0:
                    te_val = base_te
                else:
                    te_val = (leave_sum + global_mean * prior) / (leave_cnt + prior)
                data_rows.append((val, fold_idx, float(te_val)))

        fold_df = pd.DataFrame(data_rows, columns=[col, "fold", f"{col}_te"])
        fold_df[col] = fold_df[col].astype(np.int32)
        fold_df["fold"] = fold_df["fold"].astype(np.int16)
        fold_df[f"{col}_te"] = fold_df[f"{col}_te"].astype(np.float32)
        fold_tables[col] = fold_df

        cnt_df = pd.DataFrame(cnt_rows, columns=[col, f"{col}_cnt"])
        cnt_df[col] = cnt_df[col].astype(np.int32)
        cnt_df[f"{col}_cnt"] = cnt_df[f"{col}_cnt"].astype(np.float32)
        count_tables[col] = cnt_df

    return fold_tables, count_tables, global_te_lookup


def write_enriched_train(
    train_path: str,
    output_path: str,
    cat_cols: List[str],
    te_cols: List[str],
    seq_col: str,
    target_col: str,
    encoders: Dict[str, Dict[str, int]],
    fold_tables: Dict[str, pd.DataFrame],
    count_tables: Dict[str, pd.DataFrame],
    global_te_lookup: Dict[str, Dict[int, float]],
    chunk_size: int,
) -> None:
    pf = pq.ParquetFile(train_path)
    rng = np.random.default_rng(CFG["SEED"])

    writer = None
    summary_cols = []

    for batch in tqdm(pf.iter_batches(batch_size=chunk_size), desc="Writing train parquet"):
        df = batch.to_pandas()
        fold_ids = rng.integers(0, CFG["TARGET_ENC_SPLITS"], size=len(df))
        df["fold"] = fold_ids.astype(np.int16)

        for col in cat_cols:
            df[col] = encode_series(df[col], encoders[col])

        seq_features = compute_sequence_features(df[seq_col], seq_col)
        for name, arr in seq_features.items():
            df[name] = arr
            if name not in summary_cols:
                summary_cols.append(name)

        for col in te_cols:
            default_te = global_te_lookup[col].get(
                encoders[col].get("UNK", 0),
                float(np.mean(list(global_te_lookup[col].values()))),
            )
            df = df.merge(fold_tables[col], on=[col, "fold"], how="left")
            df[f"{col}_te"] = df[f"{col}_te"].fillna(default_te)
            df = df.merge(count_tables[col], on=col, how="left")
            df[f"{col}_cnt"] = df[f"{col}_cnt"].fillna(0.0)
            df[f"{col}_te"] = df[f"{col}_te"].astype(np.float32)
            df[f"{col}_cnt"] = df[f"{col}_cnt"].astype(np.float32)
            if f"{col}_te" not in summary_cols:
                summary_cols.append(f"{col}_te")
            if f"{col}_cnt" not in summary_cols:
                summary_cols.append(f"{col}_cnt")

        df.drop(columns=["fold"], inplace=True)

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)

    if writer is not None:
        writer.close()

    print(f"추가된 학습 피처: {summary_cols}")


def write_enriched_test(
    test_path: str,
    output_path: str,
    cat_cols: List[str],
    te_cols: List[str],
    seq_col: str,
    encoders: Dict[str, Dict[str, int]],
    global_te_lookup: Dict[str, Dict[int, float]],
    count_tables: Dict[str, pd.DataFrame],
    chunk_size: int,
) -> None:
    pf = pq.ParquetFile(test_path)
    writer = None

    cnt_lookup = {
        col: dict(zip(table[col], table[f"{col}_cnt"])) for col, table in count_tables.items()
    }

    for batch in tqdm(pf.iter_batches(batch_size=chunk_size), desc="Writing test parquet"):
        df = batch.to_pandas()

        for col in cat_cols:
            df[col] = encode_series(df[col], encoders[col])

        seq_features = compute_sequence_features(df[seq_col], seq_col)
        for name, arr in seq_features.items():
            df[name] = arr

        for col in te_cols:
            te_map = global_te_lookup[col]
            default_te = te_map.get(encoders[col].get("UNK", 0), float(np.mean(list(te_map.values()))))
            df[f"{col}_te"] = df[col].map(te_map).fillna(default_te).astype(np.float32)
            df[f"{col}_cnt"] = df[col].map(cnt_lookup[col]).fillna(0.0).astype(np.float32)

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
        writer.write_table(table)

    if writer is not None:
        writer.close()


def main() -> None:
    train_path = resolve_path("train.parquet")
    test_path = resolve_path("test.parquet")

    print("데이터 경로 확인 완료")
    target_col = "clicked"
    seq_col = "seq"
    cat_cols = ["gender", "age_group", "inventory_id", "l_feat_14"]
    te_cols = ["inventory_id", "age_group", "gender"]
    chunk_size = CFG["CHUNK_SIZE"]

    print("1) 범주형 라벨 인코딩 매핑 생성 중...")
    encoders = build_label_maps([train_path, test_path], cat_cols, chunk_size)

    print("2) 타깃 인코딩 통계 누적 중...")
    total_stats, fold_stats, global_mean = accumulate_target_stats(
        train_path,
        te_cols,
        target_col,
        encoders,
        chunk_size,
    )

    print("3) 타깃 인코딩 테이블 구성 중...")
    fold_tables, count_tables, global_te_lookup = build_te_tables(
        total_stats,
        fold_stats,
        global_mean,
        CFG["TARGET_ENC_PRIOR"],
    )

    output_train = os.path.join(os.path.dirname(train_path), "train_enriched.parquet")
    output_test = os.path.join(os.path.dirname(test_path), "test_enriched.parquet")

    print("4) 학습 데이터 저장 중...")
    write_enriched_train(
        train_path,
        output_train,
        cat_cols,
        te_cols,
        seq_col,
        target_col,
        encoders,
        fold_tables,
        count_tables,
        global_te_lookup,
        chunk_size,
    )

    print("5) 테스트 데이터 저장 중...")
    write_enriched_test(
        test_path,
        output_test,
        cat_cols,
        te_cols,
        seq_col,
        encoders,
        global_te_lookup,
        count_tables,
        chunk_size,
    )

    print(f"저장 완료: {output_train}, {output_test}")


if __name__ == "__main__":
    main()
