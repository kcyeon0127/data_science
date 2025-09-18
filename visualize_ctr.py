# -*- coding: utf-8 -*-
"""
Visualize CTR EDA results saved in out_eda/*.csv
- 바/라인 차트 PNG로 저장
- matplotlib만 사용 (노트북/스크립트 모두 호환)

python visualize_ctr.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CFG = {
    "eda_dir": "out_eda",              # EDA 결과 CSV 폴더
    "plot_dir": "out_eda/plots",       # 출력 이미지 폴더
    "topk_bar": 20,                    # 막대그래프에서 최대 바 개수
    "figsize": (12, 6),                # 기본 그림 크기
    "dpi": 150,                        # 저장 DPI
}

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=CFG["dpi"])
    plt.close()
    print(f"[saved] {path}")

def load_csv(name):
    path = os.path.join(CFG["eda_dir"], name)
    if not os.path.exists(path):
        print(f"[warn] not found: {path}")
        return None
    return pd.read_csv(path)

def plot_overall():
    df = load_csv("overall.csv")
    if df is None: return
    rows = int(df["rows"].iloc[0])
    ctr  = float(df["ctr"].iloc[0])
    plt.figure(figsize=CFG["figsize"])
    plt.bar(["CTR"], [ctr])
    plt.title(f"Overall CTR (rows={rows:,})")
    plt.ylabel("CTR")
    savefig(os.path.join(CFG["plot_dir"], "overall_ctr.png"))

def bar_ctr(name, label_col):
    df = load_csv(name)
    if df is None: return
    # 정렬: count 내림차순 -> CTR도 같이 보이게
    if "count" in df.columns:
        df = df.sort_values("count", ascending=False)
    # 상위 K만
    df = df.head(CFG["topk_bar"])

    x = df[label_col].astype(str).values
    ctr = df["ctr"].values
    cnt = df["count"].values if "count" in df.columns else None

    fig, ax1 = plt.subplots(figsize=CFG["figsize"])
    ax1.bar(x, ctr)
    ax1.set_ylabel("CTR")
    ax1.set_title(f"CTR by {label_col}")

    # count가 있으면 2축으로 함께 표시
    if cnt is not None:
        ax2 = ax1.twinx()
        ax2.plot(x, cnt, marker="o")
        ax2.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    savefig(os.path.join(CFG["plot_dir"], f"ctr_by_{label_col}.png"))

def plot_seq_len_stats():
    df = load_csv("seq_length_stats.csv")
    if df is None: return
    stats = ["p50","p75","p90","p95","max"]
    vals = [df[s].iloc[0] for s in stats if s in df.columns]
    plt.figure(figsize=CFG["figsize"])
    plt.plot(stats[:len(vals)], vals, marker="o")
    plt.title("Sequence Length Percentiles")
    plt.ylabel("Sequence length")
    savefig(os.path.join(CFG["plot_dir"], "seq_length_percentiles.png"))

def plot_missing_rates():
    df = load_csv("missing_rates_keycols.csv")
    if df is None or df.empty: return
    df = df.sort_values("null_rate", ascending=True)  # 낮은→높은
    plt.figure(figsize=CFG["figsize"])
    plt.barh(df["column"], df["null_rate"])
    plt.xlabel("Missing rate")
    plt.title("Missing rate by key columns")
    savefig(os.path.join(CFG["plot_dir"], "missing_rates_keycols.png"))

def plot_inventory_top():
    # inventory_id Top-K
    df = load_csv("ctr_by_inventory_id.csv")
    if df is None: return
    df = df.sort_values("count", ascending=False).head(CFG["topk_bar"])
    # x축이 너무 길면 문자열 슬라이스
    df["inventory_id"] = df["inventory_id"].astype(str).str.slice(0, 20)
    plt.figure(figsize=CFG["figsize"])
    plt.bar(df["inventory_id"], df["ctr"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("CTR")
    plt.title("CTR by inventory_id (Top)")
    savefig(os.path.join(CFG["plot_dir"], "ctr_by_inventory_id_top.png"))

def plot_ads_set_top():
    # l_feat_14 (Ads set) Top-K
    df = load_csv("ctr_by_l_feat_14.csv")
    if df is None: return
    df = df.sort_values("count", ascending=False).head(CFG["topk_bar"])
    df["l_feat_14"] = df["l_feat_14"].astype(str).str.slice(0, 20)
    plt.figure(figsize=CFG["figsize"])
    plt.bar(df["l_feat_14"], df["ctr"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("CTR")
    plt.title("CTR by l_feat_14 (Top)")
    savefig(os.path.join(CFG["plot_dir"], "ctr_by_l_feat_14_top.png"))

def plot_numeric_feature_summary():
    df = load_csv("numeric_feature_summary.csv")
    if df is None or df.empty: return
    # 예시: mean_clicked1 - mean_clicked0 차이를 절대값으로 보고 Top-K
    df = df.dropna(subset=["mean_clicked0","mean_clicked1"])
    if df.empty: return
    df["lift_abs"] = (df["mean_clicked1"] - df["mean_clicked0"]).abs()
    df = df.sort_values("lift_abs", ascending=False).head(CFG["topk_bar"])

    plt.figure(figsize=CFG["figsize"])
    plt.barh(df["feature"], df["lift_abs"])
    plt.xlabel("|mean(1) - mean(0)|")
    plt.title("Numeric features: absolute mean lift (Top)")
    plt.gca().invert_yaxis()
    savefig(os.path.join(CFG["plot_dir"], "numeric_feature_lift_top.png"))

def plot_train_test_hour():
    df = load_csv("dist_hour_train_test.csv")
    if df is None: return
    # 결측 채우기
    for c in ["train_count", "test_count"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)
    df = df.sort_values("hour")

    plt.figure(figsize=CFG["figsize"])
    plt.plot(df["hour"], df.get("train_count", 0), marker="o", label="train")
    plt.plot(df["hour"], df.get("test_count", 0), marker="o", label="test")
    plt.legend()
    plt.xlabel("hour")
    plt.ylabel("count")
    plt.title("Train vs Test: hour distribution")
    savefig(os.path.join(CFG["plot_dir"], "dist_hour_train_test.png"))

def main():
    ensure_dir(CFG["plot_dir"])

    plot_overall()
    # 기본 카테고리들
    bar_ctr("ctr_by_gender.csv", "gender")
    bar_ctr("ctr_by_age_group.csv", "age_group")
    bar_ctr("ctr_by_hour.csv", "hour")
    bar_ctr("ctr_by_day_of_week.csv", "day_of_week")
    # Top-K heavy
    plot_inventory_top()
    plot_ads_set_top()

    plot_missing_rates()
    plot_seq_len_stats()
    plot_numeric_feature_summary()
    plot_train_test_hour()

    print("\n[Done] Plots saved to", CFG["plot_dir"])

if __name__ == "__main__":
    main()

