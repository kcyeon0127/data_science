#!/usr/bin/env python3
"""
Mac용 CTR 예측 파이프라인 (Attention 기반 신경망)

대용량 데이터 로딩 → 간단한 전처리 → Self-Attention 네트워크 학습

필수 패키지
    pip install torch pandas numpy scikit-learn tqdm
"""

import os
import gc
import time
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from xgboost import callback as xgb_callback
from tqdm.auto import tqdm

MAX_SEQ_LEN = 128

warnings.filterwarnings("ignore")

try:
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise ImportError(
        "PyTorch가 설치되어 있지 않습니다. `pip install torch` 후 다시 실행하세요."
    ) from exc


def compute_weighted_logloss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """50:50 class-balanced logloss."""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    y_true = np.asarray(y_true)

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
    return float(loss.sum() / denom)


def pad_sequences(sequences: List[np.ndarray], max_len: int = MAX_SEQ_LEN):
    trimmed = []
    for seq in sequences:
        if len(seq) == 0:
            trimmed.append(np.zeros(1, dtype=np.float32))
            continue
        if len(seq) > max_len:
            seq = seq[-max_len:]
        trimmed.append(seq)

    if not trimmed:
        return np.zeros((0, max_len), dtype=np.float32), np.zeros(0, dtype=np.int64)

    cur_max = max(len(seq) for seq in trimmed)
    padded = np.zeros((len(trimmed), cur_max), dtype=np.float32)
    lengths = np.zeros(len(trimmed), dtype=np.int64)
    for i, seq in enumerate(trimmed):
        length = len(seq)
        padded[i, :length] = seq
        lengths[i] = length
    return padded, lengths


def collate_train(batch):
    numeric, cats, seqs, targets = zip(*batch)

    if numeric[0].size == 0:
        numeric_tensor = torch.empty(len(numeric), 0, dtype=torch.float32)
    else:
        numeric_tensor = torch.tensor(np.stack(numeric), dtype=torch.float32)

    cat_tensor = torch.tensor(np.stack(cats), dtype=torch.long)
    seq_padded, lengths = pad_sequences(seqs)
    seq_tensor = torch.from_numpy(seq_padded)
    length_tensor = torch.from_numpy(lengths)
    target_tensor = torch.tensor(np.stack(targets), dtype=torch.float32)

    return numeric_tensor, cat_tensor, seq_tensor, length_tensor, target_tensor


def collate_eval(batch):
    numeric, cats, seqs, targets = zip(*batch)

    if numeric[0].size == 0:
        numeric_tensor = torch.empty(len(numeric), 0, dtype=torch.float32)
    else:
        numeric_tensor = torch.tensor(np.stack(numeric), dtype=torch.float32)

    cat_tensor = torch.tensor(np.stack(cats), dtype=torch.long)
    seq_padded, lengths = pad_sequences(seqs)
    seq_tensor = torch.from_numpy(seq_padded)
    length_tensor = torch.from_numpy(lengths)

    if targets[0] == -1:
        target_tensor = None
    else:
        target_tensor = torch.tensor(np.stack(targets), dtype=torch.float32)

    return numeric_tensor, cat_tensor, seq_tensor, length_tensor, target_tensor


class CTRDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
        seq_col: Optional[str],
    ) -> None:
        self.numeric = df[numeric_cols].values.astype(np.float32) if numeric_cols else None
        self.categoricals = [df[col].values.astype(np.int64) for col in categorical_cols]
        self.targets = df["clicked"].values.astype(np.float32) if "clicked" in df else None
        self.seq_col = seq_col
        if seq_col and seq_col in df.columns:
            self.seqs = df[seq_col].fillna("").astype(str).values
        else:
            self.seqs = None

    def __len__(self) -> int:
        if self.categoricals:
            return len(self.categoricals[0])
        if self.numeric is not None:
            return len(self.numeric)
        return len(self.seqs) if self.seqs is not None else 0

    def __getitem__(self, idx: int):
        numeric_feat = self.numeric[idx] if self.numeric is not None else np.empty(0, dtype=np.float32)
        cat_feat = [cat[idx] for cat in self.categoricals]
        if self.seqs is not None:
            seq_str = self.seqs[idx]
            if seq_str:
                seq_array = np.fromstring(seq_str, sep=",", dtype=np.float32)
                if seq_array.size == 0:
                    seq_array = np.zeros(1, dtype=np.float32)
            else:
                seq_array = np.zeros(1, dtype=np.float32)
        else:
            seq_array = np.zeros(1, dtype=np.float32)

        target = self.targets[idx] if self.targets is not None else -1
        return numeric_feat, cat_feat, seq_array, target


class AttentionCTRNet(nn.Module):
    def __init__(
        self,
        num_numeric: int,
        cat_cardinalities: List[int],
        use_sequence: bool,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_numeric = num_numeric
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card, embed_dim) for card in cat_cardinalities])
        self.numeric_proj = nn.Linear(num_numeric, embed_dim) if num_numeric else None
        self.use_sequence = use_sequence
        self.seq_proj = nn.Linear(1, embed_dim) if use_sequence else None

        token_count = len(cat_cardinalities) + (1 if num_numeric else 0) + 1  # +CLS
        # seq tokens will extend position embedding dynamically
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, token_count, embed_dim))

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, numeric, cats, seqs, seq_lengths, return_repr: bool = False):
        batch_size = cats[0].size(0) if cats else numeric.size(0)

        tokens = []
        if self.numeric_proj is not None:
            numeric_token = self.numeric_proj(numeric).unsqueeze(1)
            tokens.append(numeric_token)

        for embedding, cat in zip(self.cat_embeddings, cats):
            tokens.append(embedding(cat).unsqueeze(1))

        key_padding_masks = []
        device = numeric.device if numeric is not None else cats[0].device

        tokens = torch.cat(tokens, dim=1) if tokens else numeric.unsqueeze(1)
        if tokens.dim() == 2:  # fallback if no tokens, should not happen
            tokens = tokens.unsqueeze(1)
        key_padding_masks.append(torch.zeros(batch_size, tokens.size(1), dtype=torch.bool, device=device))

        if self.use_sequence and self.seq_proj is not None and seqs is not None:
            seq_emb = self.seq_proj(seqs.unsqueeze(-1))  # (B, L, D)
            tokens = torch.cat([tokens, seq_emb], dim=1)
            seq_len = seqs.size(1)
            seq_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            padding_mask = seq_positions >= seq_lengths.unsqueeze(1)
            key_padding_masks.append(padding_mask)

        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, tokens], dim=1)

        # Extend positional embeddings if sequence makes tokens longer
        if tokens.size(1) > self.position.size(1):
            pos_expand = self.position
            extra = tokens.size(1) - self.position.size(1)
            extra_pos = torch.zeros(1, extra, self.embed_dim, device=self.position.device)
            pos_expand = torch.cat([self.position, extra_pos], dim=1)
        else:
            pos_expand = self.position

        tokens = tokens + pos_expand[:, : tokens.size(1)]

        mask = torch.cat([torch.zeros(batch_size, 1, dtype=torch.bool, device=device)] + key_padding_masks, dim=1)

        attn_out, _ = self.attn(tokens, tokens, tokens, key_padding_mask=mask)
        tokens = self.dropout(attn_out) + tokens
        tokens = self.ffn(tokens) + tokens
        cls_repr = tokens[:, 0, :]
        logits = self.head(cls_repr).squeeze(-1)
        if return_repr:
            return logits, cls_repr
        return logits


class MacXGBoostAttention:
    def __init__(self) -> None:
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.cardinalities: List[int] = []
        self.seq_col: Optional[str] = "seq"
        self.max_seq_len: int = MAX_SEQ_LEN
        print("🍎 Attention CTR 파이프라인 초기화 완료")

    # ------------------------------------------------------------------
    # 데이터 로딩 & 전처리
    # ------------------------------------------------------------------
    def load_data(self, sample_ratio: float = 0.3, use_batch: bool = False) -> bool:
        print("\n📂 1) 데이터 로딩 시작...")

        train_path = "data/train.parquet"
        test_path = "data/test.parquet"

        if sample_ratio < 1.0:
            print(f"🔄 샘플링 모드 - {sample_ratio*100:.0f}% 데이터 사용")
            full = pd.read_parquet(train_path)
            clicked = full[full["clicked"] == 1]
            not_clicked = full[full["clicked"] == 0]
            n_clicked = int(len(clicked) * sample_ratio)
            n_not_clicked = int(len(not_clicked) * sample_ratio)
            sample_clicked = clicked.sample(min(len(clicked), n_clicked), random_state=42)
            sample_not_clicked = not_clicked.sample(min(len(not_clicked), n_not_clicked), random_state=42)
            self.train_df = pd.concat([sample_clicked, sample_not_clicked], ignore_index=True)
            del full, clicked, not_clicked, sample_clicked, sample_not_clicked
            gc.collect()
        elif use_batch:
            print("🛡️ 배치 모드 - PyArrow 필요")
            try:
                import pyarrow.parquet as pq

                parquet_file = pq.ParquetFile(train_path)
                batches = []
                for batch in tqdm(parquet_file.iter_batches(batch_size=400_000), desc="파케 배치"):
                    batches.append(batch.to_pandas())
                self.train_df = pd.concat(batches, ignore_index=True)
            except Exception as err:  # pragma: no cover - runtime safety
                print(f"❌ 배치 로딩 실패: {err}. 전체 로딩으로 전환합니다.")
                self.train_df = pd.read_parquet(train_path)
        else:
            print("📦 전체 데이터 로딩")
            self.train_df = pd.read_parquet(train_path)

        self.test_df = pd.read_parquet(test_path)

        if self.seq_col not in self.train_df.columns:
            self.seq_col = None
        else:
            self.train_df[self.seq_col] = self.train_df[self.seq_col].fillna("")
            if self.seq_col in self.test_df.columns:
                self.test_df[self.seq_col] = self.test_df[self.seq_col].fillna("")
            print(f"   ↪ 시퀀스 길이 최대 {self.max_seq_len} 토큰으로 제한")

        print("✅ 데이터 로딩 완료")
        print(f"   Train: {self.train_df.shape}")
        print(f"   Test : {self.test_df.shape}")
        print(f"   클릭률: {self.train_df['clicked'].mean():.4f}")
        return True

    def add_target_encoding(self, columns: Optional[List[str]] = None, prior: int = 50) -> None:
        if columns is None:
            columns = ["inventory_id", "gender", "age_group"]

        available = [col for col in columns if col in self.train_df.columns]
        if not available:
            return

        print("\n🎯 2) 타깃 인코딩 기반 파생 특성 생성...")
        global_mean = self.train_df["clicked"].mean()

        for col in tqdm(available, desc="타깃 인코딩"):
            tqdm.write(f"   ➕ {col} 처리 중...")
            te_col = f"{col}_ctr_te"
            count_col = f"{col}_count"

            stats = self.train_df.groupby(col)["clicked"].agg(["sum", "count"])
            ctr_map = (stats["sum"] + global_mean * prior) / (stats["count"] + prior)

            sum_map = self.train_df[col].map(stats["sum"])
            count_map = self.train_df[col].map(stats["count"])
            numerator = sum_map - self.train_df["clicked"] + global_mean * prior
            denominator = count_map - 1 + prior
            self.train_df[te_col] = (numerator / denominator).fillna(global_mean)
            self.train_df[count_col] = count_map.fillna(0)

            if col in self.test_df.columns:
                self.test_df[te_col] = self.test_df[col].map(ctr_map).fillna(global_mean)
                self.test_df[count_col] = self.test_df[col].map(stats["count"]).fillna(0)
            else:
                self.test_df[te_col] = global_mean
                self.test_df[count_col] = 0

            self.train_df[te_col] = self.train_df[te_col].astype(np.float32)
            self.test_df[te_col] = self.test_df[te_col].astype(np.float32)
            self.train_df[count_col] = self.train_df[count_col].astype(np.float32)
            self.test_df[count_col] = self.test_df[count_col].astype(np.float32)

        gc.collect()

    def preprocess(self) -> bool:
        print("\n🔧 3) 특성 전처리 진행...")

        numeric_cols = [col for col in self.train_df.columns if col.startswith(("feat_", "history_", "l_feat_"))]
        extra_numeric = [col for col in self.train_df.columns if col.endswith(('_ctr_te', '_count'))]
        numeric_cols = list(dict.fromkeys(numeric_cols + extra_numeric))

        categorical_cols = [col for col in ["gender", "age_group", "inventory_id"] if col in self.train_df.columns]

        for col in tqdm(numeric_cols, desc="수치형"):
            mean_val = self.train_df[col].fillna(0).mean()
            self.train_df[col] = self.train_df[col].fillna(mean_val).astype(np.float32)
            self.test_df[col] = self.test_df[col].fillna(mean_val).astype(np.float32)

        self.label_encoders = {}
        for col in tqdm(categorical_cols, desc="카테고리"):
            le = LabelEncoder()
            combined = pd.concat([self.train_df[col], self.test_df[col]], ignore_index=True).fillna("unknown").astype(str)
            le.fit(combined)
            self.train_df[col] = le.transform(self.train_df[col].fillna("unknown").astype(str))
            self.test_df[col] = le.transform(self.test_df[col].fillna("unknown").astype(str))
            self.label_encoders[col] = le

        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.cardinalities = [int(self.train_df[col].max()) + 2 for col in categorical_cols]

        print(f"✅ 전처리 완료: numeric {len(self.numeric_cols)}, categorical {len(self.categorical_cols)}")
        return True

    # ------------------------------------------------------------------
    # 학습 루프
    # ------------------------------------------------------------------
    def train_attention_model(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
        batch_size: int = 1024,
        max_epochs: int = 15,
        patience: int = 3,
        lr: float = 3e-4,
    ) -> Optional[nn.Module]:
        print("\n🚀 4) Attention CTR 모델 학습 시작")

        X_train, X_val = train_test_split(
            self.train_df,
            test_size=0.2,
            random_state=42,
            stratify=self.train_df["clicked"],
        )

        train_ds = CTRDataset(X_train, self.numeric_cols, self.categorical_cols, self.seq_col)
        val_ds = CTRDataset(X_val, self.numeric_cols, self.categorical_cols, self.seq_col)

        pin = torch.cuda.is_available()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pin = False

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_train,
            pin_memory=pin,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_eval,
            pin_memory=pin,
        )

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"🖥️ 사용 디바이스: {device}")
        if device.type == "mps":
            print("   ➕ MPS(Apple Metal) 가속 사용 중")
        elif device.type == "cuda":
            print(f"   ➕ CUDA GPU: {torch.cuda.get_device_name(0)}")

        use_seq = self.seq_col is not None
        model = AttentionCTRNet(
            len(self.numeric_cols),
            self.cardinalities,
            use_sequence=use_seq,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        best_score = -np.inf
        best_state = None
        wait = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
            for step, (numeric, cats, seqs, seq_len, target) in enumerate(epoch_bar, start=1):
                optimizer.zero_grad()

                numeric = numeric.to(device)
                cats = [cats[:, i].to(device) for i in range(cats.size(1))] if cats.size(1) > 0 else []
                seqs = seqs.to(device)
                seq_len = seq_len.to(device)
                target = target.to(device)

                logits = model(numeric, cats, seqs, seq_len)
                loss = criterion(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item() * len(target)
                if step % 20 == 0:
                    epoch_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_loss /= len(train_ds)
            epoch_bar.close()

            ap, wll = self.evaluate(model, val_loader, device)
            blended = 0.5 * ap + 0.5 * (1 - wll)

            print(
                f"   Epoch {epoch:02d} | TrainLoss {epoch_loss:.4f} | Val AP {ap:.4f} | Val WLL {wll:.4f} | Blended {blended:.4f}"
            )

            if blended > best_score:
                best_score = blended
                best_state = model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("⚠️ Early stopping 발동")
                    break

        if best_state is None:
            print("❌ 학습 실패")
            return None

        model.load_state_dict(best_state)
        return model

    @torch.no_grad()
    def evaluate(self, model: nn.Module, loader: DataLoader, device: torch.device):
        model.eval()
        preds = []
        targets = []

        for numeric, cats, seqs, seq_len, target in loader:
            numeric = numeric.to(device)
            cats = [cats[:, i].to(device) for i in range(cats.size(1))] if cats.size(1) > 0 else []
            seqs = seqs.to(device)
            seq_len = seq_len.to(device)
            logits = model(numeric, cats, seqs, seq_len)
            prob = torch.sigmoid(logits).cpu().numpy()
            preds.append(prob)
            targets.append(target.numpy())

        preds = np.concatenate(preds)
        targets = np.concatenate(targets)
        ap = average_precision_score(targets, preds)
        wll = compute_weighted_logloss(targets, preds)
        return ap, wll

    @torch.no_grad()
    def generate_embeddings(
        self,
        model: nn.Module,
        df: pd.DataFrame,
        numeric_cols: List[str],
        batch_size: int = 2048,
    ) -> np.ndarray:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        dataset = CTRDataset(df, numeric_cols, self.categorical_cols, self.seq_col)
        pin = torch.cuda.is_available()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pin = False

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_eval,
            pin_memory=pin,
        )

        model = model.to(device)
        model.eval()
        embeddings = np.zeros((len(dataset), model.embed_dim), dtype=np.float32)
        offset = 0

        for numeric, cats, seqs, seq_len, _ in tqdm(loader, desc="임베딩 추출", leave=False):
            numeric = numeric.to(device)
            cats = [cats[:, i].to(device) for i in range(cats.size(1))] if cats.size(1) > 0 else []
            seqs = seqs.to(device)
            seq_len = seq_len.to(device)
            _, cls_repr = model(numeric, cats, seqs, seq_len, return_repr=True)
            batch_emb = cls_repr.cpu().numpy()
            embeddings[offset : offset + batch_emb.shape[0]] = batch_emb
            offset += batch_emb.shape[0]

        return embeddings

    def append_attention_embeddings(self, model: nn.Module) -> None:
        print("\n🔗 5) Attention 임베딩 생성...")
        base_numeric_cols = list(self.numeric_cols)
        train_emb = self.generate_embeddings(model, self.train_df, base_numeric_cols)
        test_emb = self.generate_embeddings(model, self.test_df, base_numeric_cols)

        embed_cols = [f"attn_emb_{i}" for i in range(train_emb.shape[1])]
        for idx, col in enumerate(embed_cols):
            self.train_df[col] = train_emb[:, idx].astype(np.float32)
            self.test_df[col] = test_emb[:, idx].astype(np.float32)

        self.numeric_cols.extend(embed_cols)
        self.numeric_cols = list(dict.fromkeys(self.numeric_cols))

        model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def predict(self, model: nn.Module, batch_size: int = 4096) -> np.ndarray:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model = model.to(device)
        print(f"🖥️ 예측 디바이스: {device}")
        if device.type == "mps":
            print("   ➕ MPS(Apple Metal) 가속 사용 중")
        elif device.type == "cuda":
            print(f"   ➕ CUDA GPU: {torch.cuda.get_device_name(0)}")
        test_ds = CTRDataset(self.test_df, self.numeric_cols, self.categorical_cols, self.seq_col)

        pin = torch.cuda.is_available()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pin = False

        loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_eval,
            pin_memory=pin,
        )

        all_preds = []
        model.eval()
        for numeric, cats, seqs, seq_len, _ in tqdm(loader, desc="테스트 예측"):
            numeric = numeric.to(device)
            cats = [cats[:, i].to(device) for i in range(cats.size(1))] if cats.size(1) > 0 else []
            seqs = seqs.to(device)
            seq_len = seq_len.to(device)
            logits = model(numeric, cats, seqs, seq_len)
            prob = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(prob)

        return np.concatenate(all_preds)

    def train_xgboost_with_embeddings(self) -> np.ndarray:
        print("\n🌲 6) XGBoost 학습 및 예측 시작")

        feature_cols = self.numeric_cols + self.categorical_cols
        X = self.train_df[feature_cols]
        y = self.train_df["clicked"]

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

        model_configs = {
            "xgb_fast": {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 500,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "scale_pos_weight": scale_pos_weight,
                "random_state": 42,
                "n_jobs": -1,
            },
            "xgb_deep": {
                "max_depth": 8,
                "learning_rate": 0.05,
                "n_estimators": 800,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "scale_pos_weight": scale_pos_weight,
                "random_state": 42,
                "n_jobs": -1,
            },
        }

        test_preds = []
        for name, params in model_configs.items():
            print(f"\n🌳 {name} 학습 중...")
            cfg = params.copy()
            n_estimators = cfg.pop("n_estimators")
            booster = xgb.XGBClassifier(
                objective="binary:logistic",
                tree_method=tree_method,
                eval_metric="aucpr",
                n_estimators=n_estimators,
                enable_categorical=False,
                **cfg,
            )
            booster.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
                callbacks=[
                    xgb_callback.EarlyStopping(
                        rounds=20,
                        maximize=True,
                        metric_name="aucpr",
                        save_best=True,
                    )
                ],
            )
            best_iter = getattr(booster, "best_iteration", None)
            if best_iter is not None and best_iter >= 0:
                val_pred = booster.predict_proba(X_val, iteration_range=(0, best_iter + 1))[:, 1]
                test_pred = booster.predict_proba(self.test_df[feature_cols], iteration_range=(0, best_iter + 1))[:, 1]
            else:
                val_pred = booster.predict_proba(X_val)[:, 1]
                test_pred = booster.predict_proba(self.test_df[feature_cols])[:, 1]
            ap = average_precision_score(y_val, val_pred)
            wll = compute_weighted_logloss(y_val.values, val_pred)
            blended = 0.5 * ap + 0.5 * (1 - wll)
            print(
                f"   📊 Val AP: {ap:.4f} | Val WLL: {wll:.4f} | Blended: {blended:.4f}"
            )

            test_preds.append(test_pred)

        final_pred = np.mean(test_preds, axis=0)
        return final_pred

    # ------------------------------------------------------------------
    # 전체 파이프라인
    # ------------------------------------------------------------------
    def run(
        self,
        sample_ratio: float = 0.3,
        use_batch: bool = False,
        add_te: bool = True,
    ) -> None:
        start = time.time()

        if not self.load_data(sample_ratio=sample_ratio, use_batch=use_batch):
            return

        if add_te:
            self.add_target_encoding()

        if not self.preprocess():
            return

        print("\n🔁 전체 파이프라인 시작")

        model = self.train_attention_model()
        if model is None:
            return

        self.append_attention_embeddings(model)

        preds = self.train_xgboost_with_embeddings()
        submission = self._build_submission(preds)
        path = "submission_mac_xgboost_atten.csv"
        submission.to_csv(path, index=False)
        elapsed = time.time() - start

        print("\n🎉 7) 파이프라인 완료")
        print(f"⏱️ 총 소요 시간: {elapsed/60:.1f}분")
        print(f"📁 제출 파일: {path}")

    def _build_submission(self, preds: np.ndarray) -> pd.DataFrame:
        try:
            submission = pd.read_csv("data/sample_submission.csv")
            submission["clicked"] = preds
        except Exception:
            submission = pd.DataFrame({
                "ID": [f"TEST_{i:07d}" for i in range(len(preds))],
                "clicked": preds,
            })
        return submission


def choose_setting(choice: str):
    mapping = {
        "1": (0.3, False),
        "2": (0.5, False),
        "3": (0.7, False),
        "4": (1.0, False),
        "5": (1.0, True),
    }
    return mapping.get(choice, (0.3, False))


def main():
    print("🚀 Attention 기반 CTR 파이프라인 시작")
    print("=" * 50)
    print("1. 🚀 초고속 (30% 샘플)")
    print("2. ⚡ 빠른  (50% 샘플)")
    print("3. 🎯 정확 (70% 샘플)")
    print("4. 🏆 전체 데이터 (직접 로딩)")
    print("5. 🛡️ 전체 데이터 (배치 로딩)")

    choice = input("선택 (1-5, 기본 1): ").strip() or "1"
    sample_ratio, use_batch = choose_setting(choice)

    pipeline = MacXGBoostAttention()
    pipeline.run(sample_ratio=sample_ratio, use_batch=use_batch, add_te=True)


if __name__ == "__main__":
    main()
