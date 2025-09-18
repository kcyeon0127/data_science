"""
train.parquet → 여러 shard로 분할 저장 (Polars 스트리밍)
- clicked==1: 전부 보존
- clicked==0: 확률 샘플링(neg = pos * neg_pos_ratio)
- feature_cols.txt 저장 (Colab 학습 단계에서 재사용)
- test는 건드리지 않음 (1GB는 Colab에서 한 번에 읽어도 안전)
train.parquet [파일] :
총 10,704,179개 샘플
총 119개 ('clicked' Target 컬럼 포함) 컬럼 존재
gender : 성별
age_group : 연령 그룹
inventory_id : 지면 ID
day_of_week : 주번호
hour : 시간
seq : 유저 서버 로그 시퀀스
l_feat_* : 속성 정보 피처 (l_feat_14는 Ads set)
feat_e_* : 정보영역 e 피처
feat_d_* : 정보영역 d 피처
feat_c_* : 정보영역 c 피처
feat_b_* : 정보영역 b 피처
feat_a_* : 정보영역 a 피처
history_a_* : 과거 인기도 피처
clicked : 클릭 여부 (Label)


test.parquet [파일] :
총 1,527,298개 샘플
총 119개 ('ID' 식별자 컬럼 포함) 컬럼 존재
ID : 샘플 식별자
gender : 성별
age_group : 연령 그룹
inventory_id : 지면 ID
day_of_week : 주번호
hour : 시간
seq : 유저 서버 로그 시퀀스
l_feat_* : 속성 정보 피처 (l_feat_14는 Ads set)
feat_e_* : 정보영역 e 피처
feat_d_* : 정보영역 d 피처
feat_c_* : 정보영역 c 피처
feat_b_* : 정보영역 b 피처
feat_a_* : 정보영역 a 피처
history_a_* : 과거 인기도 피처


sample_submission.csv [파일] - 제출 양식
ID : 샘플 식별자
clicked : 광고를 클릭할 확률 (0 ~ 1)

"""

import os
import polars as pl

# ==============================
# Hyperparameters (수정만 하면 됨)
# ==============================
CFG = {
    "train_path": "data/train.parquet",
    "out_dir":   "data/out_shards",  # out_dir/train_shards/ 생성됨

    "target_col": "clicked",
    "seq_col":    "seq",

    "neg_pos_ratio": 2,        # neg = pos * ratio
    "shard_rows":   1_000_000, # 샤드당 행 수 (Colab RAM에 맞게 5e5~1e6 권장)
    "compression": "snappy",   # snappy | zstd | lz4
}

def main():
    TRAIN_SHARD_DIR = os.path.join(CFG["out_dir"], "train_shards")
    os.makedirs(TRAIN_SHARD_DIR, exist_ok=True)

    # 1) 전체 컬럼 파악 → feature_cols 결정
    cols = pl.scan_parquet(CFG["train_path"]).select(pl.all().first()).collect().columns
    feature_exclude = {CFG["target_col"], CFG["seq_col"], "ID"}
    feature_cols = [c for c in cols if c not in feature_exclude]

    use_cols_train = feature_cols + [CFG["seq_col"], CFG["target_col"]]

    # 2) pos/neg 개수 (스트리밍 집계)
    lazy = pl.scan_parquet(CFG["train_path"])
    pos_cnt = lazy.filter(pl.col(CFG["target_col"]) == 1).select(pl.len()).collect(streaming=True).item()
    neg_cnt = lazy.filter(pl.col(CFG["target_col"]) == 0).select(pl.len()).collect(streaming=True).item()
    print(f"[train] pos={pos_cnt:,}, neg={neg_cnt:,}")

    # 3) neg 확률샘플링 비율
    target_neg = min(neg_cnt, pos_cnt * CFG["neg_pos_ratio"])
    # neg_keep_ratio = (target_neg / neg_cnt) if neg_cnt > 0 else 1.0
    neg_keep_ratio = 1.0 
    print(f"neg_keep_ratio={neg_keep_ratio:.6f}")

    # 4) 긍정 전체 + 부정 확률샘플링 → 결합 후 셔플(스트리밍)
    pos_lazy = (
        pl.scan_parquet(CFG["train_path"])
        .select(use_cols_train)
        .filter(pl.col(CFG["target_col"]) == 1)
    )
    neg_lazy = (
        pl.scan_parquet(CFG["train_path"])
        .select(use_cols_train)
        .filter(pl.col(CFG["target_col"]) == 0)
        .with_columns(keep = (pl.random() < neg_keep_ratio))
        .filter(pl.col("keep") == True)
        .drop("keep")
    )
    train_small = pl.concat([pos_lazy, neg_lazy]).shuffle()

    # 5) 샤드로 저장
    rows_written = 0
    shard_idx = 0
    for df in train_small.collect(streaming=True).iter_slices(n_rows=CFG["shard_rows"]):
        out = os.path.join(TRAIN_SHARD_DIR, f"train_shard_{shard_idx:03d}.parquet")
        df.write_parquet(out, compression=CFG["compression"])
        rows_written += df.height
        shard_idx += 1
        print(f"saved {out} rows={df.height}")
    print(f"Total train rows written: {rows_written:,}")

    # 6) feature_cols 저장 (Colab에서 그대로 사용)
    with open(os.path.join(CFG["out_dir"], "feature_cols.txt"), "w") as f:
        for c in feature_cols:
            f.write(c + "\n")
    print("Saved feature_cols.txt")

if __name__ == "__main__":
    main()
