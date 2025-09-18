"""
EDA for CTR dataset (Polars streaming; memory-safe, no full load)
- Overall CTR
- CTR by key categorical features (gender, age_group, hour, day_of_week, l_feat_14, inventory_id Top-K)
- Missing rate for selected columns
- Sequence length distribution (percentiles)  <-- 빈/NULL 시퀀스는 길이 0으로 처리
- Numeric feature summaries for selected prefixes (overall mean/std and by clicked=1/0)
Outputs CSVs to out_dir.

python eda_ctr_polars.py
"""

import os
import polars as pl

CFG = {
    # ---- paths ----
    "train_path": "data/train.parquet",

    "out_dir":    "out_eda",

    # ---- schema hints ----
    "target_col": "clicked",
    "id_col":     "ID",
    "seq_col":    "seq",

    # CTR by category (top-K cut to avoid huge tables)
    "topk_inventory": 20,
    "topk_ads_set":   20,   # for l_feat_14

    # numeric prefixes to summarize (limit N columns per prefix to keep it light)
    "numeric_prefixes": ["history_a_", "feat_a_", "feat_b_", "feat_c_", "feat_d_", "feat_e_"],
    "numeric_max_cols_per_prefix": 10,   # limit per prefix (adjust as you like)

    # how many rows to sample to infer column lists (header-only is enough, but we use it anyway)
    "inspect_rows": 1,
}

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_csv(df: pl.DataFrame, path: str):
    df.write_csv(path)
    print(f"[saved] {path} rows={df.height}")

def main():
    ensure_dir(CFG["out_dir"])
    train = pl.scan_parquet(CFG["train_path"])

    # ---------------------------
    # 0) Column inventory
    # ---------------------------
    train_cols = train.head(CFG["inspect_rows"]).collect().columns
    print(f"[info] train columns: {len(train_cols)}")

    def has_col(name: str) -> bool:
        return name in train_cols

    # ---------------------------
    # 1) Overall CTR
    # ---------------------------
    overall = (
        train
        .select([
            pl.len().alias("rows"),
            pl.col(CFG["target_col"]).mean().alias("ctr"),
        ])
        .collect(streaming=True)
    )
    print("\n[Overall]")
    print(overall)
    save_csv(overall, os.path.join(CFG["out_dir"], "overall.csv"))

    # ---------------------------
    # 2) CTR by category features
    # ---------------------------
    def ctr_by(col: str, topk: int | None = None, sort_count_desc: bool = True):
        if not has_col(col):
            return None
        df = (
            train
            .group_by(col)
            .agg([
                pl.count().alias("count"),
                pl.col(CFG["target_col"]).mean().alias("ctr"),
            ])
        )
        if sort_count_desc:
            df = df.sort("count", descending=True)
        if topk is not None:
            df = df.limit(topk)
        out = df.collect(streaming=True)
        save_csv(out, os.path.join(CFG["out_dir"], f"ctr_by_{col}.csv"))
        print(f"\n[CTR by {col}]")
        print(out.head(10))
        return out

    ctr_by("gender")
    ctr_by("age_group")
    ctr_by("hour")
    ctr_by("day_of_week")
    ctr_by("l_feat_14", topk=CFG["topk_ads_set"])
    ctr_by("inventory_id", topk=CFG["topk_inventory"])

    # ---------------------------
    # 3) Missing rate for selected useful columns
    # ---------------------------
    candidate_cols = [
        "gender", "age_group", "inventory_id",
        "day_of_week", "hour",
        "l_feat_14",
        CFG["seq_col"],
    ]
    cols_exist = [c for c in candidate_cols if has_col(c)]
    miss = (
        train
        .select([
            pl.len().alias("__n__"),
            *[pl.col(c).null_count().alias(f"{c}__nulls") for c in cols_exist]
        ])
        .collect(streaming=True)
    )
    n_total = int(miss["__n__"][0])
    rows = []
    for c in cols_exist:
        nulls = int(miss[f"{c}__nulls"][0])
        rows.append({"column": c, "nulls": nulls, "null_rate": nulls / max(1, n_total)})
    miss_df = pl.DataFrame(rows).sort("null_rate", descending=True)
    save_csv(miss_df, os.path.join(CFG["out_dir"], "missing_rates_keycols.csv"))
    print("\n[Missing rates (key cols)]")
    print(miss_df)

    # ---------------------------
    # 4) Sequence length distribution  (빈/NULL → 0)
    # ---------------------------
    if has_col(CFG["seq_col"]):
        # 문자열 정규화 후, 내용이 비어 있으면 0, 아니면 (공백 수 + 1)
        norm = (
            pl.col(CFG["seq_col"])
            .cast(pl.Utf8)
            .str.replace_all(",", " ")
            .str.strip_chars()
            .str.replace_all(r"\s+", " ")
        )

        seq_len_expr = pl.when(norm.is_null() | (norm == ""))
        seq_len_expr = seq_len_expr.then(pl.lit(0)).otherwise(
            norm.str.count_matches(" ") + 1
        ).alias("seq_len")

        seq_stats = (
            train
            .select([seq_len_expr])
            .select([
                pl.len().alias("rows"),
                pl.col("seq_len").median().alias("p50"),
                pl.col("seq_len").quantile(0.75, "nearest").alias("p75"),
                pl.col("seq_len").quantile(0.90, "nearest").alias("p90"),
                pl.col("seq_len").quantile(0.95, "nearest").alias("p95"),
                pl.col("seq_len").max().alias("max"),
            ])
            .collect(streaming=True)
        )
        save_csv(seq_stats, os.path.join(CFG["out_dir"], "seq_length_stats.csv"))
        print("\n[Sequence length stats]")
        print(seq_stats)

    # ---------------------------
    # 5) Numeric feature summaries (selected prefixes; light mode)
    # ---------------------------
    to_summarize = []
    for pref in CFG["numeric_prefixes"]:
        pref_cols = [c for c in train_cols if c.startswith(pref)]
        to_summarize.extend(pref_cols[: CFG["numeric_max_cols_per_prefix"]])

    to_summarize = [c for c in to_summarize if has_col(c)]
    if to_summarize:
        overall_aggs = []
        for c in to_summarize:
            overall_aggs.append(pl.col(c).mean().alias(f"{c}__mean"))
            overall_aggs.append(pl.col(c).std().alias(f"{c}__std"))
        overall_stats = train.select(overall_aggs).collect(streaming=True)

        by_mean = (
            train
            .group_by(CFG["target_col"])
            .agg([pl.col(c).mean().alias(f"{c}__mean") for c in to_summarize])
            .sort(CFG["target_col"])
            .collect(streaming=True)
        )

        rows = []
        for c in to_summarize:
            rows.append({
                "feature": c,
                "overall_mean": float(overall_stats[f"{c}__mean"][0]) if overall_stats[f"{c}__mean"][0] is not None else None,
                "overall_std":  float(overall_stats[f"{c}__std"][0])  if overall_stats[f"{c}__std"][0]  is not None else None,
                "mean_clicked0": None,
                "mean_clicked1": None,
            })

        gvals = set(by_mean[CFG["target_col"]].to_list())
        row0 = by_mean.filter(pl.col(CFG["target_col"]) == 0) if 0 in gvals else None
        row1 = by_mean.filter(pl.col(CFG["target_col"]) == 1) if 1 in gvals else None

        def get_cell(df: pl.DataFrame, name: str):
            return None if df is None or df.is_empty() else (float(df[name][0]) if df[name][0] is not None else None)

        for r in rows:
            c = r["feature"]
            r["mean_clicked0"] = get_cell(row0, f"{c}__mean")
            r["mean_clicked1"] = get_cell(row1, f"{c}__mean")

        num_summary = pl.DataFrame(rows).sort("feature")
        save_csv(num_summary, os.path.join(CFG["out_dir"], "numeric_feature_summary.csv"))
        print("\n[Numeric feature summary] (first 10)")
        print(num_summary.head(10))

    # ---------------------------
    # 6) (Optional) Simple train drift checks (counts only)
    # ---------------------------
    if has_col("hour"):
        tr_hour = train.group_by("hour").count().collect(streaming=True).rename({"count": "train_count"})
        save_csv(tr_hour.sort("hour"), os.path.join(CFG["out_dir"], "dist_hour_train.csv"))

    print("\n[Done] EDA artifacts saved to:", CFG["out_dir"])

if __name__ == "__main__":
    main()
