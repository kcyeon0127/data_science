#!/usr/bin/env python3
"""CTR 파이프라인: Transformer 기반 시퀀스 임베딩 + XGBoost"""

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


CFG = {
    "BATCH_SIZE": 1024,
    "EPOCHS": 10,
    "LEARNING_RATE": 5e-4,
    "SEED": 42,
    "MAX_SEQ_LEN": 128,
    "D_MODEL": 64,
    "N_HEAD": 4,
    "N_LAYERS": 2,
    "PATIENCE": 2,
    "MAX_GRAD_NORM": 1.0,
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


def compute_weighted_logloss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1 - eps, neginf=eps)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    pos_mask = y_true == 1
    neg_mask = ~pos_mask
    pos_count = pos_mask.sum()
    neg_count = neg_mask.sum()
    weights = np.zeros_like(y_pred, dtype=float)
    if pos_count:
        weights[pos_mask] = 0.5 / pos_count
    if neg_count:
        weights[neg_mask] = 0.5 / neg_count
    loss = -weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    denom = weights.sum()
    if denom == 0:
        return float(np.mean(loss))
    value = loss.sum() / denom
    if not np.isfinite(value):
        return float(np.nanmean(np.nan_to_num(loss, nan=0.0)))
    return float(value)


# ------------------------- 데이터 로딩 -------------------------
def resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    alt = os.path.join("data", os.path.basename(path))
    return alt if os.path.exists(alt) else path


train_path = resolve_path("train.parquet")
test_path = resolve_path("test.parquet")

print("데이터 로드 시작")
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


def add_sequence_features(train_df: pd.DataFrame, test_df: pd.DataFrame, seq_col: str) -> List[str]:
    def build_features(df: pd.DataFrame) -> np.ndarray:
        seq_values = df[seq_col].fillna("").astype(str).values
        length = np.zeros(len(df), dtype=np.float32)
        mean = np.zeros(len(df), dtype=np.float32)
        std = np.zeros(len(df), dtype=np.float32)
        last = np.zeros(len(df), dtype=np.float32)
        recent_mean = np.zeros(len(df), dtype=np.float32)

        window = min(10, CFG["MAX_SEQ_LEN"])

        for idx, seq_str in enumerate(seq_values):
            if seq_str:
                arr = np.fromstring(seq_str, sep=",", dtype=np.float32)
                if arr.size == 0:
                    arr = np.zeros(1, dtype=np.float32)
            else:
                arr = np.zeros(1, dtype=np.float32)

            if arr.size > CFG["MAX_SEQ_LEN"]:
                arr = arr[-CFG["MAX_SEQ_LEN"]:]

            length[idx] = arr.size
            mean[idx] = arr.mean() if arr.size else 0.0
            std[idx] = arr.std() if arr.size > 1 else 0.0
            last[idx] = arr[-1] if arr.size else 0.0
            recent_mean[idx] = arr[-window:].mean() if arr.size else 0.0

        return np.vstack([length, mean, std, last, recent_mean]).T

    cols = [
        f"{seq_col}_length",
        f"{seq_col}_mean",
        f"{seq_col}_std",
        f"{seq_col}_last",
        f"{seq_col}_recent_mean",
    ]

    train_feats = build_features(train_df)
    test_feats = build_features(test_df)

    for i, col in enumerate(cols):
        train_df[col] = train_feats[:, i]
        test_df[col] = test_feats[:, i]

    return cols


def add_target_encoding_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: List[str],
    target_col: str,
    prior: float = 50.0,
    n_splits: int = 5,
) -> List[str]:
    from sklearn.model_selection import StratifiedKFold

    global_mean = train_df[target_col].mean()
    new_cols = []

    for col in columns:
        te_col = f"{col}_target_enc"
        cnt_col = f"{col}_count"
        new_cols.extend([te_col, cnt_col])

        train_df[te_col] = 0.0
        train_df[cnt_col] = 0.0

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG["SEED"])
        for train_idx, val_idx in skf.split(train_df[col], train_df[target_col]):
            fold = train_df.iloc[train_idx]
            stats = fold.groupby(col)[target_col].agg(["sum", "count"])
            stats["te"] = (stats["sum"] + global_mean * prior) / (stats["count"] + prior)

            val = train_df.iloc[val_idx]
            train_df.loc[val.index, te_col] = val[col].map(stats["te"]).fillna(global_mean)
            train_df.loc[val.index, cnt_col] = val[col].map(stats["count"]).fillna(0)

        full_stats = train_df.groupby(col)[target_col].agg(["sum", "count"])
        full_stats["te"] = (full_stats["sum"] + global_mean * prior) / (full_stats["count"] + prior)

        test_df[te_col] = test_df[col].map(full_stats["te"]).fillna(global_mean)
        test_df[cnt_col] = test_df[col].map(full_stats["count"]).fillna(0)

    return new_cols


sequence_feature_cols = add_sequence_features(train, test, SEQ_COL)
target_encoding_cols = add_target_encoding_features(
    train,
    test,
    columns=["inventory_id", "age_group", "gender"],
    target_col=TARGET_COL,
)

num_cols.extend(sequence_feature_cols + target_encoding_cols)
num_cols = list(dict.fromkeys(num_cols))


# ------------------------- Dataset & Collate -------------------------
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

        max_len = CFG["MAX_SEQ_LEN"]
        if seq_col and seq_col in df.columns:
            seq_strings = df[seq_col].fillna("").astype(str).values
            seq_arrays = []
            for s in seq_strings:
                if s:
                    arr = np.fromstring(s, sep=",", dtype=np.float32)
                    if arr.size == 0:
                        arr = np.zeros(1, dtype=np.float32)
                else:
                    arr = np.zeros(1, dtype=np.float32)
                if arr.size > max_len:
                    arr = arr[-max_len:]
                seq_arrays.append(arr)
            self.seq_arrays = seq_arrays
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


# ------------------------- Transformer 모델 -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return x + self.pe[:, :length, :]


class TransformerCTR(nn.Module):
    def __init__(
        self,
        num_features: int,
        cat_cardinalities: List[int],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.d_model = d_model

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, d_model) for card in cat_cardinalities
        ])
        self.numeric_proj = nn.Linear(num_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=CFG["MAX_SEQ_LEN"] + 1)
        self.seq_proj = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        fusion_dim = d_model * (len(cat_cardinalities) + 2)  # CLS + numeric + cat tokens
        self.cross = CrossNetwork(fusion_dim, num_layers=2)

        layers = [
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, num_x, cat_x, seqs, seq_lengths, return_features: bool = False):
        batch_size = num_x.size(0)

        # Numeric + categorical tokens
        num_token = self.numeric_proj(num_x).unsqueeze(1)
        cat_tokens = [emb(cat_x[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_embeddings)]

        # Sequence tokens with CLS
        seqs = seqs.unsqueeze(-1)
        seq_tokens = self.seq_proj(seqs)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        seq_tokens = torch.cat([cls_token, seq_tokens], dim=1)
        seq_tokens = self.pos_encoding(seq_tokens)

        # Mask (True for padding positions)
        length = seq_tokens.size(1) - 1
        mask = torch.arange(length, device=seqs.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
        mask = torch.cat([torch.zeros((batch_size, 1), dtype=torch.bool, device=seqs.device), mask], dim=1)

        trans_out = self.transformer(seq_tokens, src_key_padding_mask=mask)
        seq_repr = trans_out[:, 0, :]

        tokens = torch.cat([num_token, *cat_tokens, seq_repr.unsqueeze(1)], dim=1)
        fused = tokens.reshape(batch_size, -1)
        fused_cross = self.cross(fused)

        if return_features:
            return fused_cross

        logits = self.mlp(fused_cross).squeeze(-1)
        return logits


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(num_layers)])

    def forward(self, x0):
        x = x0
        for layer in self.layers:
            x = x0 * layer(x) + x
        return x


# ------------------------- 학습 함수 -------------------------
def create_dataloaders(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    seq_col: str,
    target_col: str,
    batch_size: int,
):
    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=CFG["SEED"],
        stratify=df[target_col],
    )

    train_dataset = ClickDataset(train_df.reset_index(drop=True), num_cols, cat_cols, seq_col, target_col)
    val_dataset = ClickDataset(val_df.reset_index(drop=True), num_cols, cat_cols, seq_col, target_col)

    pin = torch.cuda.is_available()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pin = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_train,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_train,
        pin_memory=pin,
    )

    return train_loader, val_loader


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    preds, targets = [], []
    for num_x, cat_x, seqs, lens, ys in loader:
        num_x = num_x.to(device)
        cat_x = cat_x.to(device)
        seqs = seqs.to(device)
        lens = lens.to(device)
        ys = ys.to(device)
        logits = model(num_x, cat_x, seqs, lens)
        prob = torch.sigmoid(logits).cpu().numpy()
        preds.append(prob)
        targets.append(ys.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    ap = average_precision_score(targets, preds)
    wll = compute_weighted_logloss(targets, preds)
    return ap, wll


def train_model(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    seq_col: str,
    target_col: str,
    batch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> TransformerCTR:
    train_loader, val_loader = create_dataloaders(df, num_cols, cat_cols, seq_col, target_col, batch_size)

    cat_cardinalities = [len(cat_encoders[c].classes_) for c in cat_cols]
    model = TransformerCTR(
        num_features=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        d_model=CFG["D_MODEL"],
        n_heads=CFG["N_HEAD"],
        n_layers=CFG["N_LAYERS"],
    ).to(device)

    pos_weight_value = (len(df) - df[target_col].sum()) / df[target_col].sum()
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_score = -np.inf
    best_state = None
    patience_counter = 0

    print("학습 시작")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for num_x, cat_x, seqs, lens, ys in tqdm(train_loader, desc=f"[Train Epoch {epoch}]"):
            num_x = num_x.to(device)
            cat_x = cat_x.to(device)
            seqs = seqs.to(device)
            lens = lens.to(device)
            ys = ys.to(device)

            optimizer.zero_grad()
            logits = model(num_x, cat_x, seqs, lens)
            loss = criterion(logits, ys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["MAX_GRAD_NORM"])
            optimizer.step()

            total_loss += loss.item() * ys.size(0)

        total_loss /= len(train_loader.dataset)
        val_ap, val_wll = evaluate_model(model, val_loader, device)
        blended = 0.5 * val_ap + 0.5 * (1 - val_wll)
        print(f"[Epoch {epoch}] Train Loss: {total_loss:.4f} | Val AP: {val_ap:.4f} | Val WLL: {val_wll:.4f} | Blended: {blended:.4f}")
        if torch.cuda.is_available():
            print(f"   GPU Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")

        if blended > best_score:
            best_score = blended
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CFG["PATIENCE"]:
                print("⚠️ Early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("학습 완료")
    return model


@torch.no_grad()
def extract_embeddings(
    model: TransformerCTR,
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
    embeddings = np.zeros((len(dataset), model.cross.layers[0].in_features), dtype=np.float32)
    offset = 0

    for num_x, cat_x, seqs, lens in tqdm(loader, desc="[Embedding]"):
        num_x = num_x.to(DEVICE)
        cat_x = cat_x.to(DEVICE)
        seqs = seqs.to(DEVICE)
        lens = lens.to(DEVICE)
        feats = model(num_x, cat_x, seqs, lens, return_features=True)
        feats = feats.cpu().numpy()
        embeddings[offset : offset + feats.shape[0]] = feats
        offset += feats.shape[0]

    return embeddings


# ------------------------- 파이프라인 -------------------------
model = train_model(
    df=train,
    num_cols=num_cols,
    cat_cols=cat_cols,
    seq_col=SEQ_COL,
    target_col=TARGET_COL,
    batch_size=CFG["BATCH_SIZE"],
    epochs=CFG["EPOCHS"],
    lr=CFG["LEARNING_RATE"],
    device=DEVICE,
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "cfg": CFG,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    },
    "wd_transformer_checkpoint.pt",
)
print("체크포인트 저장: wd_transformer_checkpoint.pt")

print("임베딩 추출 시작")
train_emb = extract_embeddings(model, train, num_cols, cat_cols, SEQ_COL, CFG["BATCH_SIZE"])
test_emb = extract_embeddings(model, test, num_cols, cat_cols, SEQ_COL, CFG["BATCH_SIZE"])

embed_cols = [f"trans_emb_{i}" for i in range(train_emb.shape[1])]
train = pd.concat([train, pd.DataFrame(train_emb, columns=embed_cols, index=train.index)], axis=1)
test = pd.concat([test, pd.DataFrame(test_emb, columns=embed_cols, index=test.index)], axis=1)

aug_num_cols = num_cols + embed_cols

del train_emb, test_emb
model.cpu()
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()


print("XGBoost 학습 시작")
feature_cols_final = aug_num_cols + cat_cols
X = train[feature_cols_final].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
y = train[TARGET_COL]
test_features = test[feature_cols_final].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)

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
    test_pred = xgb_model.predict_proba(test_features, iteration_range=(0, best_iter + 1))[:, 1]
else:
    val_pred = xgb_model.predict_proba(X_val)[:, 1]
    test_pred = xgb_model.predict_proba(test_features)[:, 1]

val_ap = average_precision_score(y_val, val_pred)
print(f"Validation AP: {val_ap:.4f}")


print("추론 완료, 제출 파일 생성")
submit_path = resolve_path("sample_submission.csv")
submit = pd.read_csv(submit_path)
submit["clicked"] = test_pred
submit.to_csv("./submission_xg_trans.csv", index=False)
print("저장 완료 -> submission_xg_trans.csv")
