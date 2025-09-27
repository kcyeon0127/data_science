#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mac용 XGBoost 전용 CTR 예측 (TensorFlow 없음)
- 세그폴트 방지: MPS 워터마크, 배치 단위 이동, 단방향 GRU, 샘플 학습→전량 추출
- 자동 배치 축소(Adaptive Batch): MPS OOM 발생 시 즉시 배치 축소 후 재시도
- 시퀀스: GRU + 간단 Attention 임베딩 추출 → XGBoost 결합
- XGBoost: 모든 피처를 float32/int32로 강제 → 실패 시 QuantileDMatrix 폴백
"""

import os
import sys
import platform

# ======== MPS OOM 시 세그폴트 대신 예외로 전환 (환경변수 우선) ========
DEFAULT_WATERMARK = "0.5"  # 디버깅 시 "0.0", 관대 모드 "0.8"
if "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = DEFAULT_WATERMARK

# (선택) 과도한 스레드 증가 방지
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
from tqdm.auto import tqdm
import gc
import time

# =========================
# 공용 유틸
# =========================
def _ensure_numeric32(df: pd.DataFrame) -> pd.DataFrame:
    """XGBoost 친화적으로 모든 컬럼을 float32/int32로 강제 변환."""
    out = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            out[c] = s.astype(np.uint8)
        elif pd.api.types.is_integer_dtype(s):
            # pandas nullable(Int64 등)도 int32로
            out[c] = pd.to_numeric(s, errors='coerce').fillna(0).astype(np.int32)
        elif pd.api.types.is_float_dtype(s):
            out[c] = s.astype(np.float32)
        elif pd.api.types.is_categorical_dtype(s):
            # category → 코드(int32)
            out[c] = s.cat.codes.replace(-1, 0).astype(np.int32)
        else:
            # object/ArrowDtype/기타 → 숫자 변환 실패는 0으로
            out[c] = pd.to_numeric(s, errors='coerce').fillna(0).astype(np.float32)
    return pd.DataFrame(out, index=df.index)

# =========================
# PyTorch (선택) - 시퀀스 임베딩 전용
# =========================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"🍎 Apple Silicon MPS 가속 활성화 (워터마크={os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']})")
    else:
        device = torch.device("cpu")
        print("💻 CPU 모드")

    TORCH_AVAILABLE = True
    print("🧠 PyTorch 로드됨 - GRU + Attention 사용 가능")
except ImportError:
    TORCH_AVAILABLE = False
    device = None
    print("⚠️ PyTorch 없음 - 시퀀스 통계 특성으로 대체")

# =========================
# 평가 지표
# =========================
def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    """가중 LogLoss (0/1 클래스 기여 50:50)"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)
    ll_0 = -np.mean(np.log(1 - y_pred[mask_0])) if mask_0.sum() > 0 else 0.0
    ll_1 = -np.mean(np.log(y_pred[mask_1])) if mask_1.sum() > 0 else 0.0
    return 0.5 * ll_0 + 0.5 * ll_1

def calculate_competition_score(y_true, y_pred):
    """대회 평가: 0.5*AP + 0.5*(1/(1+WLL))"""
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1.0 / (1.0 + wll))
    return score, ap, wll

# =========================
# GRU + Attention (단방향, 경량)
# =========================
if TORCH_AVAILABLE:
    class GRUAttentionModel(nn.Module):
        def __init__(self, vocab_size=50000, embedding_dim=64, hidden=96, output_dim=32):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.gru = nn.GRU(embedding_dim, hidden, batch_first=True)  # 단방향
            self.attn = nn.Linear(hidden, 1)
            self.fc1 = nn.Linear(hidden, 96)
            self.dropout = nn.Dropout(0.2)
            self.fc2 = nn.Linear(96, output_dim)

        def forward(self, x):  # x: (B, T)
            emb = self.embedding(x)                       # (B, T, E)
            out, _ = self.gru(emb)                        # (B, T, H)
            w = torch.softmax(self.attn(out), dim=1)      # (B, T, 1)
            pooled = torch.sum(out * w, dim=1)            # (B, H)
            x = torch.relu(self.fc1(pooled))
            x = self.dropout(x)
            x = torch.relu(self.fc2(x))                   # (B, 32)
            return x

def _is_mps_oom(err: Exception) -> bool:
    s = str(err).lower()
    return ("mps" in s and "memory" in s) or "out of memory" in s

class PyTorchSequenceProcessor:
    """배치 단위 디바이스 이동 + 샘플 학습 → 전량 추출 + 자동 배치 축소"""
    def __init__(self, vocab_size=50000, embedding_dim=64, hidden=96,
                 output_dim=32, train_cap=100_000, batch_size=256, max_len=50):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden = hidden
        self.output_dim = output_dim
        self.train_cap = train_cap
        self.batch_size = batch_size
        self.max_len = max_len
        self.model = None
        self.device = device if TORCH_AVAILABLE else None

    def create_model(self):
        if not TORCH_AVAILABLE:
            return None
        self.model = GRUAttentionModel(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            hidden=self.hidden,
            output_dim=self.output_dim
        )
        if self.device:
            self.model = self.model.to(self.device)
        return self.model

    def _make_targets(self, batch_np):
        """간단 자기지도 표적: 비영(0 제외) 평균값을 32차로 broadcast"""
        nz_mean = np.where(batch_np != 0, batch_np, np.nan).astype(np.float32)
        means = np.nanmean(nz_mean, axis=1)
        means = np.nan_to_num(means, nan=0.0).astype(np.float32)
        return torch.tensor(np.repeat(means[:, None], self.output_dim, axis=1),
                            dtype=torch.float32)

    def _safe_train_step(self, batch_np, optimizer, criterion):
        x = torch.tensor(batch_np, dtype=torch.long, device=self.device)
        y = self._make_targets(batch_np).to(self.device)
        optimizer.zero_grad(set_to_none=True)
        out = self.model(x)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    def _safe_infer_step(self, batch_np):
        with torch.no_grad():
            x = torch.tensor(batch_np, dtype=torch.long, device=self.device)
            out = self.model(x)
            return out.detach().cpu().numpy()

    def train_model(self, sequences, epochs=2):
        """대용량일 때 샘플만으로 표현 학습 + OOM 시 배치 자동 축소"""
        if (not TORCH_AVAILABLE) or (self.model is None):
            return

        n = len(sequences)
        use_n = min(n, self.train_cap)
        rng = np.random.RandomState(42)
        idx = rng.choice(n, use_n, replace=False)
        train_seqs = sequences[idx]

        print(f"🚀 PyTorch 시퀀스 모델 훈련: 사용 샘플 {use_n:,}/{n:,} (epochs={epochs}, max_len={self.max_len})")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for ep in range(epochs):
            total, steps = 0.0, 0
            i = 0
            cur_bs = self.batch_size
            while i < use_n:
                ok = False
                # 최대 3회까지 배치 축소 시도
                for _ in range(3):
                    batch_np = train_seqs[i:i+cur_bs]
                    try:
                        loss_val = self._safe_train_step(batch_np, optimizer, criterion)
                        total += loss_val; steps += 1
                        i += len(batch_np)
                        ok = True
                        break
                    except RuntimeError as e:
                        if _is_mps_oom(e):
                            if torch.backends.mps.is_available():
                                torch.mps.empty_cache()
                            new_bs = max(cur_bs // 2, 32)
                            if new_bs == cur_bs:
                                raise
                            print(f"⚠️ MPS OOM 감지 → 배치 {cur_bs} → {new_bs} 축소")
                            cur_bs = new_bs
                            continue
                        else:
                            raise
                if not ok:
                    print("❌ 배치 32에서도 OOM → hidden/max_len/워터마크 조정 필요")
                    break

                if (steps % 50 == 0) and TORCH_AVAILABLE and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            print(f"   Epoch {ep+1}: Loss={total/max(steps,1):.4f} (최종 배치={cur_bs})")

    def extract_features(self, sequences):
        """전량 배치 추론(배치만 디바이스 이동) 또는 통계 피처 폴백 + OOM 자동 축소"""
        if (not TORCH_AVAILABLE) or (self.model is None):
            return self.extract_statistical_features(sequences)

        self.model.eval()
        feats = []
        i = 0
        n = len(sequences)
        cur_bs = self.batch_size
        with torch.no_grad():
            while i < n:
                ok = False
                for _ in range(3):
                    batch_np = sequences[i:i+cur_bs]
                    try:
                        out = self._safe_infer_step(batch_np)
                        feats.append(out)
                        i += len(batch_np)
                        ok = True
                        break
                    except RuntimeError as e:
                        if _is_mps_oom(e):
                            if torch.backends.mps.is_available():
                                torch.mps.empty_cache()
                            new_bs = max(cur_bs // 2, 32)
                            if new_bs == cur_bs:
                                raise
                            print(f"⚠️ MPS OOM(추론) → 배치 {cur_bs} → {new_bs} 축소")
                            cur_bs = new_bs
                            continue
                        else:
                            raise
                if not ok:
                    print("❌ (추론) 배치 32에서도 OOM → hidden/max_len/워터마크 조정 필요")
                    break

                if TORCH_AVAILABLE and torch.backends.mps.is_available() and (i // cur_bs) % 50 == 0:
                    torch.mps.empty_cache()

        return np.concatenate(feats, axis=0).astype(np.float32)

    def preprocess_sequences(self, df, seq_col='seq', max_len=None):
        """문자열 '1,2,3' → 길이 max_len의 int32 시퀀스(패딩=0)"""
        if max_len is None:
            max_len = self.max_len
        sequences = []
        for seq_str in tqdm(df[seq_col], desc="시퀀스 전처리"):
            if pd.isna(seq_str) or seq_str == '':
                seq = [0]*max_len
            else:
                try:
                    seq = [int(x) for x in str(seq_str).split(',') if x != '']
                    seq = (seq[:max_len] if len(seq) >= max_len
                           else seq + [0]*(max_len-len(seq)))
                except Exception:
                    seq = [0]*max_len
            sequences.append(seq)
        return np.asarray(sequences, dtype=np.int32)

    def extract_statistical_features(self, sequences):
        """간단 통계 피처(길이, 평균, 표준편차, min/max, median, 유니크 수, 마지막, 최근5평균)"""
        features = []
        for seq in sequences:
            nz = seq[seq != 0]
            if len(nz) == 0:
                feat = [0]*32
            else:
                feat = [
                    int(len(nz)),
                    float(np.mean(nz)),
                    float(np.std(nz)),
                    int(np.min(nz)),
                    int(np.max(nz)),
                    float(np.median(nz)),
                    int(len(np.unique(nz))),
                    int(nz[-1]),
                    float(np.mean(nz[-5:])) if len(nz) >= 5 else float(np.mean(nz)),
                ]
                feat = (feat + [0]*(32-len(feat)))[:32]
            features.append(feat)
        return np.asarray(features, dtype=np.float32)

print("🚀 Mac용 XGBoost + (선택) GRU Attention 기반 시퀀스 임베딩 CTR 예측!")
print("📊 대회 평가지표: 0.5*AP + 0.5*(1/(1+WLL))")
print("=" * 68)

# =========================
# 파이프라인
# =========================
class MacXGBoostCTR:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.feature_cols = []
        self.sequence_model = None
        print("🍎 Mac용 XGBoost + GRU Attention CTR 초기화 완료")

    # ---------- 데이터 로딩 ----------
    def load_data_efficiently(self, sample_ratio=0.3, use_batch=False):
        print("\n📂 데이터 로딩 시작...")
        try:
            train_size_gb = os.path.getsize('data/train.parquet') / (1024**3)
            print(f"📊 훈련 데이터 크기: {train_size_gb:.1f} GB")
        except Exception:
            print("📊 훈련 데이터 크기 확인 실패")

        if sample_ratio < 1.0:
            print(f"🔄 샘플링 모드 - {sample_ratio*100:.0f}% 데이터 사용")
            full_df = pd.read_parquet('data/train.parquet')
            clicked = full_df[full_df['clicked'] == 1]
            not_clicked = full_df[full_df['clicked'] == 0]

            n_clicked = int(len(clicked) * sample_ratio)
            n_not_clicked = int(len(not_clicked) * sample_ratio)

            print(f"샘플링: 클릭 {len(clicked):,} → {n_clicked:,}, 비클릭 {len(not_clicked):,} → {n_not_clicked:,}")
            sample_clicked = clicked.sample(min(len(clicked), n_clicked), random_state=42) if n_clicked > 0 else pd.DataFrame()
            sample_not_clicked = not_clicked.sample(min(len(not_clicked), n_not_clicked), random_state=42) if n_not_clicked > 0 else pd.DataFrame()
            self.train_df = pd.concat([sample_clicked, sample_not_clicked], ignore_index=True)

            del full_df, clicked, not_clicked, sample_clicked, sample_not_clicked
            gc.collect()

        elif use_batch:
            print("🔄 배치 처리 모드 - 전체 데이터를 안전하게 로딩")
            self.train_df = self.load_data_in_batches('data/train.parquet')
        else:
            print("📂 전체 데이터 직접 로딩 중...")
            try:
                self.train_df = pd.read_parquet('data/train.parquet')
                print("✅ 직접 로딩 성공")
            except MemoryError:
                print("❌ 메모리 부족! 배치 모드로 전환...")
                self.train_df = self.load_data_in_batches('data/train.parquet')

        print("📂 테스트 데이터 로딩...")
        self.test_df = pd.read_parquet('data/test.parquet')

        print(f"\n✅ 로딩 완료!")
        print(f"   훈련: {self.train_df.shape}")
        print(f"   테스트: {self.test_df.shape}")
        if 'clicked' in self.train_df.columns:
            print(f"   클릭률: {self.train_df['clicked'].mean():.4f}")

        return True

    def load_data_in_batches(self, file_path, batch_size=500_000):
        print(f"📦 배치 크기 {batch_size:,}행으로 안전 로딩...")
        try:
            print("   전체 로드 시도 중...")
            full_df = pd.read_parquet(file_path)
            print(f"✅ 전체 로드 성공: {full_df.shape}")
            return full_df
        except MemoryError:
            print("   메모리 부족! 70% 샘플링으로 대체...")
            full_df = pd.read_parquet(file_path)
            clicked = full_df[full_df['clicked'] == 1]
            not_clicked = full_df[full_df['clicked'] == 0]

            sample_ratio = 0.7
            n_clicked = int(len(clicked) * sample_ratio)
            n_not_clicked = int(len(not_clicked) * sample_ratio)

            sample_clicked = clicked.sample(min(len(clicked), n_clicked), random_state=42)
            sample_not_clicked = not_clicked.sample(min(len(not_clicked), n_not_clicked), random_state=42)
            result_df = pd.concat([sample_clicked, sample_not_clicked], ignore_index=True)

            del full_df, clicked, not_clicked, sample_clicked, sample_not_clicked
            gc.collect()
            print(f"✅ 샘플링 로드 완료: {result_df.shape}")
            return result_df
        except Exception as e:
            print(f"❌ 로딩 실패: {e}")
            print("   30% 샘플링으로 재시도...")
            full_df = pd.read_parquet(file_path)
            return full_df.sample(frac=0.3, random_state=42)

    # ---------- 전처리 ----------
    def preprocess_features(self):
        print("\n🔧 특성 전처리 시작...")

        # 수치형
        numeric_cols = [c for c in self.train_df.columns if c.startswith(('feat_', 'history_', 'l_feat_'))]

        # 카테고리
        categorical_cols = ['gender', 'age_group']
        if 'inventory_id' in self.train_df.columns:
            categorical_cols.append('inventory_id')

        # 시퀀스
        has_sequence = 'seq' in self.train_df.columns

        print(f"📊 특성 정보: 수치형 {len(numeric_cols)}개 | 카테고리 {len(categorical_cols)}개 | 시퀀스 {'있음' if has_sequence else '없음'}")

        # 수치형 결측 대체(평균)
        print("🔧 수치형 특성 처리...")
        for col in tqdm(numeric_cols, desc="수치형"):
            if col in self.train_df.columns:
                mean_val = self.train_df[col].fillna(0).mean()
                self.train_df[col] = self.train_df[col].fillna(mean_val)
                self.test_df[col] = self.test_df[col].fillna(mean_val)

        # 카테고리 라벨인코딩(Train/Test 합쳐서 fit)
        print("🔧 카테고리 특성 처리...")
        for col in tqdm(categorical_cols, desc="카테고리"):
            if col in self.train_df.columns:
                le = LabelEncoder()
                combined = pd.concat([
                    self.train_df[col].fillna('unknown').astype(str),
                    self.test_df[col].fillna('unknown').astype(str)
                ])
                le.fit(combined)
                self.train_df[col] = le.transform(self.train_df[col].fillna('unknown').astype(str))
                self.test_df[col]  = le.transform(self.test_df[col].fillna('unknown').astype(str))
                self.encoders[col] = le

        # 시퀀스 임베딩 추출 → 수치 피처로 추가
        if has_sequence:
            print("🧠 시퀀스 특성 처리 (PyTorch GRU + Attention)...")
            self.sequence_model = PyTorchSequenceProcessor(
                vocab_size=50_000, embedding_dim=64, hidden=96, output_dim=32,
                train_cap=100_000, batch_size=256, max_len=50
            )

            train_sequences = self.sequence_model.preprocess_sequences(self.train_df, 'seq')
            test_sequences  = self.sequence_model.preprocess_sequences(self.test_df , 'seq')

            if TORCH_AVAILABLE:
                print("🔧 PyTorch GRU + Attention 모델 생성...")
                self.sequence_model.create_model()
                try:
                    self.sequence_model.train_model(train_sequences, epochs=2)  # 샘플 학습
                except Exception as e:
                    print(f"❌ 시퀀스 학습 실패 → 통계 특성으로 대체: {e}")
                    self.sequence_model.model = None
            else:
                print("⚠️ PyTorch 없음 - 통계 특성으로 대체")

            print("🔍 시퀀스 특성 추출...")
            train_seq_features = self.sequence_model.extract_features(train_sequences)
            test_seq_features  = self.sequence_model.extract_features(test_sequences)
            del train_sequences, test_sequences; gc.collect()

            # 컬럼 결합
            seq_feature_names = [f'seq_feat_{i}' for i in range(train_seq_features.shape[1])]
            for i, name in enumerate(seq_feature_names):
                self.train_df[name] = train_seq_features[:, i]
                self.test_df[name]  = test_seq_features[:, i]
            numeric_cols.extend(seq_feature_names)
            print(f"✅ 시퀀스 특성 {len(seq_feature_names)}개 추가")

        # 최종 사용 피처
        self.feature_cols = [c for c in (numeric_cols + categorical_cols) if c in self.train_df.columns]
        print(f"✅ 전처리 완료: 총 {len(self.feature_cols)}개 특성 사용")
        return True

    # ---------- XGBoost 학습 (수정판: dtype 강제 + QuantileDMatrix 폴백) ----------
    def train_xgboost_models(self):
        print("\n🚀 XGBoost 모델 훈련 시작...")

        # 1) 안전한 dtype으로 강제 변환(메모리 절약도 됨)
        X = _ensure_numeric32(self.train_df[self.feature_cols].copy())
        y = self.train_df['clicked'].astype(np.int32).values

        # 메모리 사용량 로그
        feat_mem_gb = X.memory_usage(deep=True).sum() / (1024**3)
        print(f"🧮 특성 테이블 메모리(훈련 전체): {feat_mem_gb:.2f} GB")

        pos_ratio = float(np.mean(y))
        scale_pos_weight = (1 - pos_ratio) / max(pos_ratio, 1e-8)
        print(f"📊 클릭률: {pos_ratio:.4f}")
        print(f"📊 Scale pos weight: {scale_pos_weight:.2f}")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"📊 데이터 분할: 훈련 {X_train.shape[0]:,}, 검증 {X_val.shape[0]:,}")
        print(f"📊 클릭률 분포 - 전체 {pos_ratio:.4f} | 훈련 {y_train.mean():.4f} | 검증 {y_val.mean():.4f}")

        # 메모리 절약을 위해 float32 보장
        X_train = X_train.astype(np.float32, copy=False)
        X_val   = X_val.astype(np.float32,   copy=False)

        base_cfgs = {
            'xgb_ap_focused': dict(
                objective='binary:logistic',
                eval_metric=['auc', 'aucpr'],
                tree_method='hist',
                max_depth=6,
                learning_rate=0.08,
                n_estimators=600,
                subsample=0.85,
                colsample_bytree=0.85,
                scale_pos_weight=float(scale_pos_weight),
                max_bin=256,                 # 메모리 감소
                n_jobs=-1,
                random_state=42,
                verbosity=0
            ),
            'xgb_balanced': dict(
                objective='binary:logistic',
                eval_metric=['logloss', 'aucpr'],
                tree_method='hist',
                max_depth=7,
                learning_rate=0.06,
                n_estimators=800,
                subsample=0.9,
                colsample_bytree=0.9,
                scale_pos_weight=float(scale_pos_weight),
                reg_alpha=0.1,
                reg_lambda=0.1,
                max_bin=256,                 # 메모리 감소
                n_jobs=-1,
                random_state=42,
                verbosity=0
            )
        }

        self.models = {}

        # ---- 2단계: XGBClassifier → 실패 시 QuantileDMatrix 폴백 ----
        for name, params in base_cfgs.items():
            print(f"\n🔄 {name} 훈련 중...")
            try:
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=20,
                    early_stopping_rounds=20
                )

                val_pred = model.predict_proba(X_val)[:, 1]
                comp_score, ap, wll = calculate_competition_score(y_val, val_pred)
                auc = roc_auc_score(y_val, val_pred)
                best_iter = getattr(model, 'best_iteration', None)
                if best_iter is None:
                    best_iter = getattr(model, 'best_ntree_limit', None)
                print(f"   ✅ Best iteration: {best_iter}")
                print(f"   📊 {name} 성능: 대회 {comp_score:.4f} | AP {ap:.4f} | WLL {wll:.4f} | AUC {auc:.4f}")

                self.models[name] = [model]  # 그대로 사용
                continue

            except Exception as e:
                print(f"⚠️ {name} XGBClassifier 학습 실패: {e}")
                print("   → 메모리 절약형 QuantileDMatrix로 폴백합니다.")

            # ===== 폴백 경로: QuantileDMatrix + xgb.train =====
            try:
                try:
                    QDM = xgb.QuantileDMatrix    # 1.6+ 에서 지원
                except AttributeError:
                    QDM = xgb.DMatrix            # 구버전: 일반 DMatrix로 대체

                dtrain = QDM(X_train, label=y_train)
                dval   = QDM(X_val,   label=y_val)

                num_boost_round = params.get('n_estimators', 600)
                train_params = params.copy()
                train_params.pop('n_estimators', None)

                booster = xgb.train(
                    train_params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=[(dval, 'val')],
                    early_stopping_rounds=20,
                    verbose_eval=20
                )

                # 예측용 래퍼( predict_proba 인터페이스 통일 )
                class _BoosterWrapper:
                    def __init__(self, booster, feature_cols):
                        self.booster = booster
                        self.feature_cols = feature_cols
                        # best_iteration 속성 폴백 처리
                        self.best_iteration = getattr(booster, 'best_iteration', None)
                        self.best_ntree_limit = getattr(booster, 'best_ntree_limit', None)

                    def predict_proba(self, Xdf):
                        Xdf = _ensure_numeric32(Xdf[self.feature_cols])
                        dm = xgb.DMatrix(Xdf.values)
                        if self.best_iteration is not None:
                            preds = self.booster.predict(dm, iteration_range=(0, self.best_iteration + 1))
                        elif self.best_ntree_limit is not None:
                            preds = self.booster.predict(dm, ntree_limit=self.best_ntree_limit)
                        else:
                            preds = self.booster.predict(dm)
                        return np.vstack([1 - preds, preds]).T

                wrapper = _BoosterWrapper(booster, self.feature_cols)

                val_pred = wrapper.predict_proba(X_val)[:, 1]
                comp_score, ap, wll = calculate_competition_score(y_val, val_pred)
                auc = roc_auc_score(y_val, val_pred)
                print(f"   ✅ (폴백) {name} 성능: 대회 {comp_score:.4f} | AP {ap:.4f} | WLL {wll:.4f} | AUC {auc:.4f}")

                self.models[name] = [wrapper]

            except Exception as e2:
                print(f"❌ {name} 폴백 학습도 실패: {e2}")
                print("   → 파라미터를 더 보수적으로 낮추거나( max_depth↓, max_bin↓, n_estimators↓ ) 샘플링 비율을 높여주세요.")

        print("✅ 모델 훈련 완료!")

    # ---------- 예측 & 제출 ----------
    def predict_and_submit(self):
        print("\n🎯 예측 시작...")
        X_test = self.test_df[self.feature_cols].copy()
        all_predictions = []

        for name, fold_models in self.models.items():
            print(f"🔄 {name} 예측 중...")
            fold_preds = []
            for i, model in enumerate(fold_models):
                pred = model.predict_proba(X_test)[:, 1]
                fold_preds.append(pred)
                print(f"   Fold {i+1} 완료")
            avg_pred = np.mean(fold_preds, axis=0)
            all_predictions.append(avg_pred)

        final_predictions = np.mean(all_predictions, axis=0)

        # 제출 파일 생성
        try:
            submission = pd.read_csv('data/sample_submission.csv')
            submission['clicked'] = final_predictions
            print(f"✅ 올바른 ID 형식 사용: {submission.columns.tolist()}")
        except Exception:
            submission = pd.DataFrame({
                'ID': [f'TEST_{i:07d}' for i in range(len(final_predictions))],
                'clicked': final_predictions
            })
            print("⚠️ sample_submission.csv 없음 → ID 직접 생성")

        submission_path = 'submission_mac_xgboost_gru.csv'
        submission.to_csv(submission_path, index=False, encoding='utf-8')

        print(f"\n✅ 제출 파일 생성: {submission_path}")
        print(f"📊 예측 통계: mean={final_predictions.mean():.4f} | min={final_predictions.min():.4f} | max={final_predictions.max():.4f}")
        return submission_path

    # ---------- 전체 파이프라인 ----------
    def run_pipeline(self, sample_ratio=0.3, use_batch=False):
        start_time = time.time()

        if not self.load_data_efficiently(sample_ratio, use_batch):
            return False
        if not self.preprocess_features():
            return False

        self.train_xgboost_models()
        submission_path = self.predict_and_submit()

        elapsed = time.time() - start_time
        print("\n" + "🎉"*60)
        print("🎉 파이프라인 완료! 🎉")
        print("🎉"*60)
        print(f"⏱️ 총 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
        print(f"📁 제출 파일: {submission_path}")
        return True

# =========================
# 실행부
# =========================
def main():
    pipeline = MacXGBoostCTR()

    print("\n📋 실행 옵션:")
    print("1. 🚀 초고속 모드 (30% 샘플링)")
    print("2. ⚡ 빠른 모드 (50% 샘플링)")
    print("3. 🎯 정확 모드 (70% 샘플링)")
    print("4. 🏆 최고 성능 모드 (전체 직접 로딩) ⚠️ 메모리 위험")
    print("5. 🛡️ 안전 최고 성능 (전체 배치 로딩) ✅ 메모리 안전")

    choice = input("선택 (1-5, 기본값 1): ").strip() or '1'

    if choice == '5':
        sample_ratio = 1.0
        use_batch = True
        print(f"\n🛡️ 배치 처리로 전체 데이터 안전 로딩!")
    elif choice == '4':
        sample_ratio = 1.0
        use_batch = False
        print(f"\n🏆 전체 데이터 직접 로딩 (메모리 위험 감수)!")
    else:
        sample_ratios = {'1': 0.3, '2': 0.5, '3': 0.7}
        sample_ratio = sample_ratios.get(choice, 0.3)
        use_batch = False
        print(f"\n🚀 {sample_ratio*100:.0f}% 데이터로 실행 시작!")

    print("=" * 68)
    success = pipeline.run_pipeline(sample_ratio, use_batch)
    print("\n🎊 성공! 제출 파일이 준비되었습니다!" if success else "\n❌ 실행 실패")

if __name__ == "__main__":
    main()
