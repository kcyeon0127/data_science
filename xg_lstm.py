#!/usr/bin/env python3
"""CTR 파이프라인: Wide&Deep + LSTM 임베딩 -> XGBoost 앙상블"""

import os
import random
import gc
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import xgboost as xgb


# ------------------------- 설정 -------------------------
CFG = {
    "BATCH_SIZE": 1024,
    "EPOCHS": 5,
    "LEARNING_RATE": 1e-3,
    "SEED": 42,
    "MAX_SEQ_LEN": 128,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG["SEED"])

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")


# ------------------------- 데이터 로딩 -------------------------
print("데이터 로드 시작")
train_path = "./train.parquet"
test_path = "./test.parquet"
if not os.path.exists(train_path):
    train_path = "./data/train.parquet"
if not os.path.exists(test_path):
    test_path = "./data/test.parquet"

train = pd.read_parquet(train_path, engine="pyarrow")
test = pd.read_parquet(test_path, engine="pyarrow")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("데이터 로드 완료")


TARGET_COL = "clicked"
SEQ_COL = "seq"
FEATURE_EXCLUDE = {TARGET_COL, SEQ_COL, "ID"}
feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]

cat_cols = ["gender", "age_group", "inventory_id", "l_feat_14"]
num_cols = [c for c in feature_cols if c not in cat_cols]
print(f"Num features: {len(num_cols)} | Cat features: {len(cat_cols)}")


def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: List[str]):
    encoders = {}
    for col in cols:
        le = LabelEncoder()
        all_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str).fillna("UNK")
        le.fit(all_values)
        train_df[col] = le.transform(train_df[col].astype(str).fillna("UNK"))
        test_df[col] = le.transform(test_df[col].astype(str).fillna("UNK"))
        encoders[col] = le
        print(f"{col} unique categories: {len(le.classes_)}")
    return train_df, test_df, encoders


train, test, cat_encoders = encode_categoricals(train, test, cat_cols)


# ------------------------- Dataset -------------------------
class ClickDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        num_cols: List[str],
        cat_cols: List[str],
        seq_col: Optional[str],
        target_col: Optional[str] = None,
    ) -> None:
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.has_target = target_col is not None

        self.num_X = df[num_cols].astype(np.float32).fillna(0).values
        self.cat_X = df[cat_cols].astype(np.int64).values

        if seq_col and seq_col in df.columns:
            seq_strings = df[seq_col].fillna("").astype(str).values
            self.seq_arrays = []
            for s in seq_strings:
                if s:
                    arr = np.fromstring(s, sep=",", dtype=np.float32)
                    if arr.size == 0:
                        arr = np.zeros(1, dtype=np.float32)
                else:
                    arr = np.zeros(1, dtype=np.float32)
                if arr.size > CFG["MAX_SEQ_LEN"]:
                    arr = arr[-CFG["MAX_SEQ_LEN"]:]
                self.seq_arrays.append(arr)
        else:
            self.seq_arrays = [np.zeros(1, dtype=np.float32)] * len(df)

        if self.has_target:
            self.y = df[target_col].astype(np.float32).values

    def __len__(self) -> int:
        return len(self.num_X)

    def __getitem__(self, idx: int):
        num_x = torch.tensor(self.num_X[idx], dtype=torch.float32)
        cat_x = torch.tensor(self.cat_X[idx], dtype=torch.long)
        seq = torch.from_numpy(self.seq_arrays[idx])
        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return num_x, cat_x, seq, y
        return num_x, cat_x, seq


def collate_fn_train(batch):
    num_x, cat_x, seqs, ys = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    ys = torch.stack(ys)
    return num_x, cat_x, seqs_padded, seq_lengths, ys


def collate_fn_infer(batch):
    num_x, cat_x, seqs = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths


# ------------------------- 모델 정의 -------------------------
class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, 1, bias=True) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for layer in self.layers:
            x = x0 * layer(x) + x
        return x


class WideDeepCTR(nn.Module):
    def __init__(
        self,
        num_features: int,
        cat_cardinalities: List[int],
        emb_dim: int = 16,
        lstm_hidden: int = 64,
        hidden_units: Optional[List[int]] = None,
        dropout: Optional[List[float]] = None,
    ) -> None:
        super().__init__()

        if hidden_units is None:
            hidden_units = [512, 256, 128]
        if dropout is None:
            dropout = [0.1, 0.2, 0.3]

        self.emb_layers = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim) for cardinality in cat_cardinalities
        ])
        cat_input_dim = emb_dim * len(cat_cardinalities)

        self.bn_num = nn.BatchNorm1d(num_features)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        seq_out_dim = lstm_hidden * 2

        self.fused_dim = num_features + cat_input_dim + seq_out_dim
        self.cross = CrossNetwork(self.fused_dim, num_layers=2)

        layers = []
        input_dim = self.fused_dim
        for i, h in enumerate(hidden_units):
            layers.extend([
                nn.Linear(input_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout[i % len(dropout)]),
            ])
            input_dim = h
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, num_x, cat_x, seqs, seq_lengths, return_features: bool = False):
        num_x = self.bn_num(num_x)
        cat_embs = [emb(cat_x[:, i]) for i, emb in enumerate(self.emb_layers)]
        cat_feat = torch.cat(cat_embs, dim=1)

        seqs = seqs.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            seqs, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        seq_feat = torch.cat([h_n[-2], h_n[-1]], dim=1)

        fused = torch.cat([num_x, cat_feat, seq_feat], dim=1)
        fused_cross = self.cross(fused)

        if return_features:
            return fused_cross

        out = self.mlp(fused_cross)
        return out.squeeze(1)


# ------------------------- 학습 루프 -------------------------
def train_model(
    train_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    seq_col: str,
    target_col: str,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> WideDeepCTR:
    dataset = ClickDataset(train_df, num_cols, cat_cols, seq_col, target_col)
    pin = torch.cuda.is_available()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pin = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_train,
        pin_memory=pin,
    )

    cat_cardinalities = [len(cat_encoders[c].classes_) for c in cat_cols]
    model = WideDeepCTR(
        num_features=len(num_cols),
        cat_cardinalities=cat_cardinalities,
    ).to(device)

    pos_weight_value = (len(train_df) - train_df[target_col].sum()) / train_df[target_col].sum()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)

    print("학습 시작")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for num_x, cat_x, seqs, lens, ys in tqdm(loader, desc=f"[Train Epoch {epoch}]"):
            num_x, cat_x, seqs, lens, ys = (
                num_x.to(device),
                cat_x.to(device),
                seqs.to(device),
                lens.to(device),
                ys.to(device),
            )
            optimizer.zero_grad()
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * ys.size(0)

        total_loss /= len(dataset)
        print(f"[Epoch {epoch}] Train Loss: {total_loss:.4f}")
        if torch.cuda.is_available():
            print(f"   GPU Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

    print("학습 완료")
    return model


# ------------------------- 임베딩 추출 -------------------------
@torch.no_grad()
def extract_embeddings(
    model: WideDeepCTR,
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    seq_col: str,
    batch_size: int,
) -> np.ndarray:
    dataset = ClickDataset(df, num_cols, cat_cols, seq_col, None)
    pin = torch.cuda.is_available()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pin = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_infer,
        pin_memory=pin,
    )

    model = model.to(DEVICE)
    model.eval()
    embeddings = np.zeros((len(dataset), model.fused_dim), dtype=np.float32)
    offset = 0

    for num_x, cat_x, seqs, lens in tqdm(loader, desc="[Embedding]"):
        num_x, cat_x, seqs, lens = (
            num_x.to(DEVICE),
            cat_x.to(DEVICE),
            seqs.to(DEVICE),
            lens.to(DEVICE),
        )
        feats = model(num_x, cat_x, seqs, lens, return_features=True)
        feats = feats.cpu().numpy()
        embeddings[offset : offset + feats.shape[0]] = feats
        offset += feats.shape[0]

    return embeddings


# ------------------------- 파이프라인 실행 -------------------------
model = train_model(
    train_df=train,
    num_cols=num_cols,
    cat_cols=cat_cols,
    seq_col=SEQ_COL,
    target_col=TARGET_COL,
    batch_size=CFG["BATCH_SIZE"],
    epochs=CFG["EPOCHS"],
    lr=CFG["LEARNING_RATE"],
    device=DEVICE,
)

print("임베딩 추출 시작")
train_emb = extract_embeddings(model, train, num_cols, cat_cols, SEQ_COL, CFG["BATCH_SIZE"])
test_emb = extract_embeddings(model, test, num_cols, cat_cols, SEQ_COL, CFG["BATCH_SIZE"])

embed_cols = [f"lstm_emb_{i}" for i in range(train_emb.shape[1])]
train_emb_df = pd.DataFrame(train_emb, columns=embed_cols, index=train.index)
test_emb_df = pd.DataFrame(test_emb, columns=embed_cols, index=test.index)

train = pd.concat([train, train_emb_df], axis=1)
test = pd.concat([test, test_emb_df], axis=1)

aug_num_cols = num_cols + embed_cols

del train_emb, test_emb
model.cpu()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ------------------------- XGBoost 학습 -------------------------
print("XGBoost 학습 시작")

feature_cols_final = aug_num_cols + cat_cols
X = train[feature_cols_final]
y = train[TARGET_COL]

pos_ratio = y.mean()
scale_pos_weight = (1 - pos_ratio) / pos_ratio

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

tree_method = "gpu_hist" if torch.cuda.is_available() else "hist"

xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    tree_method=tree_method,
    eval_metric="aucpr",
    max_depth=6,
    learning_rate=0.05,
    n_estimators=800,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

best_iter = getattr(xgb_model, "best_iteration", None)
if best_iter is not None and best_iter >= 0:
    val_pred = xgb_model.predict_proba(X_val, iteration_range=(0, best_iter + 1))[:, 1]
    test_pred = xgb_model.predict_proba(test[feature_cols_final], iteration_range=(0, best_iter + 1))[:, 1]
else:
    val_pred = xgb_model.predict_proba(X_val)[:, 1]
    test_pred = xgb_model.predict_proba(test[feature_cols_final])[:, 1]

val_ap = average_precision_score(y_val, val_pred)
val_wll = -(0.5 * np.log(val_pred + 1e-12) * y_val + 0.5 * np.log(1 - val_pred + 1e-12) * (1 - y_val)).mean()
print(f"Validation AP: {val_ap:.4f}")


# ------------------------- 제출 -------------------------
print("추론 완료, 제출 파일 생성")
submit = pd.read_csv("./sample_submission.csv")
submit["clicked"] = test_pred
submit.to_csv("./submission_xg_lstm.csv", index=False)
print("저장 완료 -> submission_xg_lstm.csv")
xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
best_iter = getattr(xgb_model, "best_iteration", None)
