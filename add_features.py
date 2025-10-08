#!/usr/bin/env python3
"""Generate enriched train/test parquet files with sequence and target-encoding features."""

import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


CFG = {
    "SEED": 42,
    "MAX_SEQ_LEN": 128,
    "TARGET_ENC_PRIOR": 50.0,
    "TARGET_ENC_SPLITS": 5,
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


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    for col in cols:
        le = LabelEncoder()
        all_vals = pd.concat([train_df[col], test_df[col]], axis=0).astype(str).fillna("UNK")
        le.fit(all_vals)
        train_df[col] = le.transform(train_df[col].astype(str).fillna("UNK"))
        test_df[col] = le.transform(test_df[col].astype(str).fillna("UNK"))
    return train_df, test_df


def add_sequence_features(df: pd.DataFrame, seq_col: str) -> List[str]:
    max_len = CFG["MAX_SEQ_LEN"]
    window = min(10, max_len)
    length = np.zeros(len(df), dtype=np.float32)
    mean = np.zeros(len(df), dtype=np.float32)
    std = np.zeros(len(df), dtype=np.float32)
    last = np.zeros(len(df), dtype=np.float32)
    recent = np.zeros(len(df), dtype=np.float32)

    seq_iter = df[seq_col].fillna("").astype(str).values
    for idx, seq_str in enumerate(tqdm(seq_iter, desc=f"Seq features ({seq_col})")):
        if seq_str:
            arr = np.fromstring(seq_str, sep=",", dtype=np.float32)
            if arr.size == 0:
                arr = np.zeros(1, dtype=np.float32)
        else:
            arr = np.zeros(1, dtype=np.float32)

        if arr.size > max_len:
            arr = arr[-max_len:]

        length[idx] = arr.size
        mean[idx] = arr.mean() if arr.size else 0.0
        std[idx] = arr.std() if arr.size > 1 else 0.0
        last[idx] = arr[-1] if arr.size else 0.0
        recent[idx] = arr[-window:].mean() if arr.size else 0.0

    cols = [
        f"{seq_col}_length",
        f"{seq_col}_mean",
        f"{seq_col}_std",
        f"{seq_col}_last",
        f"{seq_col}_recent_mean",
    ]
    df[cols[0]] = length
    df[cols[1]] = mean
    df[cols[2]] = std
    df[cols[3]] = last
    df[cols[4]] = recent
    return cols


def add_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: List[str],
    target_col: str,
) -> List[str]:
    global_mean = train_df[target_col].mean()
    prior = CFG["TARGET_ENC_PRIOR"]
    folds = StratifiedKFold(n_splits=CFG["TARGET_ENC_SPLITS"], shuffle=True, random_state=CFG["SEED"])
    added_cols = []

    for col in columns:
        te_col = f"{col}_te"
        cnt_col = f"{col}_cnt"
        added_cols.extend([te_col, cnt_col])
        train_df[te_col] = 0.0
        train_df[cnt_col] = 0.0

        for tr_idx, val_idx in folds.split(train_df[col], train_df[target_col]):
            fold = train_df.iloc[tr_idx]
            stats = fold.groupby(col)[target_col].agg(["sum", "count"])
            stats["te"] = (stats["sum"] + global_mean * prior) / (stats["count"] + prior)

            val = train_df.iloc[val_idx]
            train_df.loc[val.index, te_col] = val[col].map(stats["te"]).fillna(global_mean)
            train_df.loc[val.index, cnt_col] = val[col].map(stats["count"]).fillna(0)

        full_stats = train_df.groupby(col)[target_col].agg(["sum", "count"])
        full_stats["te"] = (full_stats["sum"] + global_mean * prior) / (full_stats["count"] + prior)
        test_df[te_col] = test_df[col].map(full_stats["te"]).fillna(global_mean)
        test_df[cnt_col] = test_df[col].map(full_stats["count"]).fillna(0)

    return added_cols


def main():
    train_path = resolve_path("train.parquet")
    test_path = resolve_path("test.parquet")

    print("데이터 로드 시작")
    train = pd.read_parquet(train_path, engine="pyarrow")
    test = pd.read_parquet(test_path, engine="pyarrow")
    print(f"Train: {train.shape}, Test: {test.shape}")

    target_col = "clicked"
    seq_col = "seq"
    cat_cols = ["gender", "age_group", "inventory_id", "l_feat_14"]

    train, test = encode_categoricals(train, test, cat_cols)
    seq_features = add_sequence_features(train, seq_col)
    _ = add_sequence_features(test, seq_col)

    te_cols = add_target_encoding(
        train,
        test,
        columns=["inventory_id", "age_group", "gender"],
        target_col=target_col,
    )

    output_train = os.path.join(os.path.dirname(train_path), "train_enriched.parquet")
    output_test = os.path.join(os.path.dirname(test_path), "test_enriched.parquet")

    cols_summary = seq_features + te_cols
    print(f"추가된 피처: {cols_summary}")

    train.to_parquet(output_train, index=False)
    test.to_parquet(output_test, index=False)
    print(f"저장 완료: {output_train}, {output_test}")


if __name__ == "__main__":
    main()
