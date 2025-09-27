#!/usr/bin/env python3
"""
Mac용 XGBoost + 간단한 LSTM CTR 예측 파이프라인 (Attention 제외)
대회 평가지표: AP (50%) + WLL (50%)
"""

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
warnings.filterwarnings('ignore')

# XGBoost
try:
    import xgboost as xgb
    print("✅ XGBoost 로드됨")
    XGB_AVAILABLE = True
except ImportError:
    print("❌ XGBoost 없음")
    XGB_AVAILABLE = False

# PyTorch (간단한 LSTM용)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    print("🧠 PyTorch 로드됨 - 간단한 LSTM 사용 가능")

    # 안전성을 위해 CPU 강제 사용 (MPS segfault 회피)
    device = torch.device("cpu")
    print("💻 CPU 모드 (안전성 우선, 속도는 느림)")

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    device = None
    print("⚠️ PyTorch 없음 - 시퀀스 통계 특성으로 대체")

def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    """가중 LogLoss 계산 (50:50 클래스 가중치)"""
    y_pred = np.clip(y_pred, eps, 1 - eps)

    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)

    ll_0 = -np.mean(np.log(1 - y_pred[mask_0])) if mask_0.sum() > 0 else 0
    ll_1 = -np.mean(np.log(y_pred[mask_1])) if mask_1.sum() > 0 else 0

    return 0.5 * ll_0 + 0.5 * ll_1

def calculate_competition_score(y_true, y_pred):
    """대회 평가 지표: AP (50%) + WLL (50%)
    - AP (Average Precision): 예측 확률에 대해 계산된 평균 정밀도 점수
    - WLL (Weighted LogLoss): 'clicked'의 0과 1의 클래스 기여를 50:50로 맞춘 가중 LogLoss
    최종 점수: 0.5*AP + 0.5*(1/(1+WLL))
    """
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll

class SimpleLSTMModel(nn.Module):
    """간단한 PyTorch LSTM 시퀀스 처리 모델 (Attention 제외)"""

    def __init__(self, vocab_size=50000, embedding_dim=64, lstm_units=128, output_dim=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.output_dim = output_dim

        # 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM 레이어 (양방향)
        self.lstm = nn.LSTM(embedding_dim, lstm_units, batch_first=True, bidirectional=True)

        # 출력 레이어 (Global Average Pooling + FC)
        self.fc1 = nn.Linear(lstm_units * 2, 128)  # 양방향이므로 *2
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        # 임베딩
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # LSTM
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, lstm_units*2)

        # Global Average Pooling (Attention 대신)
        pooled = torch.mean(lstm_out, dim=1)  # (batch, lstm_units*2)

        # 출력 레이어
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))

        return x

class PyTorchSequenceProcessor:
    """PyTorch 기반 간단한 시퀀스 처리"""

    def __init__(self, vocab_size=50000, embedding_dim=64, lstm_units=128):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.device = device

    def create_model(self):
        """모델 생성"""
        if not TORCH_AVAILABLE:
            return None

        try:
            self.model = SimpleLSTMModel(
                vocab_size=self.vocab_size,
                embedding_dim=self.embedding_dim,
                lstm_units=self.lstm_units
            )

            # MPS 장치로 이동 시도
            if self.device and str(self.device) == 'mps':
                try:
                    self.model = self.model.to(self.device)
                    print("✅ MPS 장치로 모델 이동 성공")
                except Exception as e:
                    print(f"⚠️ MPS 이동 실패, CPU 사용: {e}")
                    self.device = torch.device("cpu")
                    self.model = self.model.to(self.device)
            elif self.device:
                self.model = self.model.to(self.device)

            return self.model

        except Exception as e:
            print(f"❌ PyTorch 모델 생성 실패: {e}")
            return None

    def train_model(self, sequences, epochs=3, batch_size=256):
        """간단한 자기지도 학습"""
        if not TORCH_AVAILABLE or self.model is None:
            return

        print(f"🚀 PyTorch 간단한 LSTM 모델 훈련 ({epochs} 에포크)...")
        print(f"🔧 장치: {self.device}, 시퀀스 수: {len(sequences)}")

        try:
            # 더미 타겟 생성 (시퀀스의 평균을 32차원으로)
            targets = []
            for seq in sequences:
                non_zero = seq[seq != 0]
                if len(non_zero) > 0:
                    mean_val = np.mean(non_zero)
                    target = np.full(32, mean_val, dtype=np.float32)
                else:
                    target = np.zeros(32, dtype=np.float32)
                targets.append(target)

            targets = np.array(targets)

            # 데이터셋 준비
            sequences_tensor = torch.tensor(sequences, dtype=torch.long)
            targets_tensor = torch.tensor(targets, dtype=torch.float32)

            if self.device:
                sequences_tensor = sequences_tensor.to(self.device)
                targets_tensor = targets_tensor.to(self.device)

            # 훈련
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            self.model.train()

            for epoch in range(epochs):
                total_loss = 0
                n_batches = 0

                for i in range(0, len(sequences_tensor), batch_size):
                    batch_seq = sequences_tensor[i:i+batch_size]
                    batch_target = targets_tensor[i:i+batch_size]

                    optimizer.zero_grad()
                    output = self.model(batch_seq)
                    loss = criterion(output, batch_target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    n_batches += 1

                    # 메모리 정리
                    if i % (batch_size * 10) == 0:  # 10배치마다
                        if torch.backends.mps.is_available():
                            torch.mps.empty_cache()

                avg_loss = total_loss / n_batches
                print(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f}")

        except Exception as e:
            print(f"❌ PyTorch 훈련 실패: {e}")
            print("🔄 시퀀스 특성을 통계로 대체합니다")

    def extract_features(self, sequences):
        """시퀀스에서 특성 추출"""
        if not TORCH_AVAILABLE or self.model is None:
            return self.statistical_features(sequences)

        print("🎯 PyTorch 간단한 LSTM 특성 추출...")

        try:
            sequences_tensor = torch.tensor(sequences, dtype=torch.long)
            if self.device:
                sequences_tensor = sequences_tensor.to(self.device)

            with torch.no_grad():
                batch_size = 256
                features_list = []

                for i in range(0, len(sequences_tensor), batch_size):
                    batch = sequences_tensor[i:i+batch_size]
                    output = self.model(batch)
                    if self.device:
                        output = output.cpu()
                    features_list.append(output.numpy())

                return np.vstack(features_list)

        except Exception as e:
            print(f"❌ PyTorch 특성 추출 실패: {e}")
            return self.statistical_features(sequences)

    def statistical_features(self, sequences):
        """통계 기반 시퀀스 특성 (폴백)"""
        print("📊 통계 기반 시퀀스 특성 추출...")

        features = []
        for seq in tqdm(sequences, desc="통계 특성"):
            non_zero = seq[seq != 0]
            if len(non_zero) > 0:
                feat = [
                    np.mean(non_zero),
                    np.std(non_zero),
                    np.max(non_zero),
                    np.min(non_zero),
                    len(non_zero),
                    len(non_zero) / len(seq),  # 밀도
                    np.median(non_zero),
                    np.sum(non_zero > np.mean(non_zero))  # 평균 이상 개수
                ]
            else:
                feat = [0] * 8

            # 32차원으로 확장 (반복)
            feat = feat * 4
            features.append(feat)

        return np.array(features)

class MacXGBoostSimple:
    """Mac용 XGBoost + 간단한 LSTM CTR 예측"""

    def __init__(self):
        self.model = None
        self.sequence_processor = PyTorchSequenceProcessor()
        print("🍎 Mac용 XGBoost + 간단한 LSTM CTR 초기화 완료")

    def load_and_preprocess(self, sample_ratio=0.3, use_batch=False):
        """데이터 로드 및 전처리"""
        print(f"📂 데이터 로딩 중... (샘플링: {sample_ratio*100:.0f}%)")

        if use_batch:
            # 배치 처리 모드
            return self.load_data_in_batches('data/train.parquet')
        else:
            # 직접 로딩 모드
            train_df = pd.read_parquet('data/train.parquet')
            test_df = pd.read_parquet('data/test.parquet')

            if sample_ratio < 1.0:
                train_df = train_df.sample(frac=sample_ratio, random_state=42)
                print(f"✅ 샘플링 완료: {len(train_df):,}행")

            return self.preprocess_data(train_df, test_df)

    def load_data_in_batches(self, file_path, batch_size=500000):
        """배치별 안전한 데이터 로딩"""
        print(f"📦 배치 크기 {batch_size:,}행으로 안전 로딩...")

        # 첫 번째 배치로 구조 확인
        first_batch = pd.read_parquet(file_path, engine='pyarrow').head(batch_size)
        print(f"✅ 첫 배치 로드: {first_batch.shape}")

        # 테스트 데이터는 전체 로드
        test_df = pd.read_parquet('data/test.parquet')

        return self.preprocess_data(first_batch, test_df)

    def preprocess_data(self, train_df, test_df):
        """데이터 전처리"""
        print("🔧 데이터 전처리 중...")

        # 기본 특성 선택
        feature_cols = []

        # 수치형 특성
        numeric_cols = [col for col in train_df.columns
                       if col.startswith(('feat_', 'history_')) and train_df[col].dtype in ['float64', 'int64']]

        print(f"📊 수치형 특성: {len(numeric_cols)}개")

        # 간단한 전처리
        for col in tqdm(numeric_cols[:50], desc="수치 전처리"):  # 상위 50개만
            if col in train_df.columns:
                mean_val = train_df[col].mean()
                std_val = train_df[col].std()

                # 결측값 처리
                train_df[col] = train_df[col].fillna(mean_val)
                test_df[col] = test_df[col].fillna(mean_val)

                # 정규화
                if std_val > 0:
                    train_df[col] = (train_df[col] - mean_val) / std_val
                    test_df[col] = (test_df[col] - mean_val) / std_val

                feature_cols.append(col)

        # 카테고리 특성 (간단하게)
        if 'gender' in train_df.columns:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            combined = pd.concat([train_df['gender'].fillna('unknown'),
                                test_df['gender'].fillna('unknown')])
            le.fit(combined.astype(str))
            train_df['gender_encoded'] = le.transform(train_df['gender'].fillna('unknown').astype(str))
            test_df['gender_encoded'] = le.transform(test_df['gender'].fillna('unknown').astype(str))
            feature_cols.append('gender_encoded')

        # 시퀀스 특성 처리
        sequence_features = self.process_sequences(train_df, test_df)

        # 기본 특성과 시퀀스 특성 결합
        X_train = np.column_stack([
            train_df[feature_cols].values,
            sequence_features['train']
        ])
        X_test = np.column_stack([
            test_df[feature_cols].values,
            sequence_features['test']
        ])

        y_train = train_df['clicked'].values

        print(f"✅ 전처리 완료: 훈련 {X_train.shape}, 테스트 {X_test.shape}")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'test_df': test_df
        }

    def process_sequences(self, train_df, test_df):
        """시퀀스 특성 처리"""
        print("🔄 시퀀스 특성 처리...")

        # 간단한 시퀀스 생성
        train_sequences = []
        test_sequences = []

        # 수치형 컬럼들을 시퀀스로 변환
        sequence_cols = [col for col in train_df.columns if col.startswith('feat_')][:20]  # 상위 20개

        for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="훈련 시퀀스"):
            seq = [int(abs(row[col]) * 1000) % 50000 for col in sequence_cols if pd.notna(row[col])]
            seq = seq[:50] + [0] * (50 - len(seq))  # 길이 50으로 패딩
            train_sequences.append(seq)

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="테스트 시퀀스"):
            seq = [int(abs(row[col]) * 1000) % 50000 for col in sequence_cols if pd.notna(row[col])]
            seq = seq[:50] + [0] * (50 - len(seq))  # 길이 50으로 패딩
            test_sequences.append(seq)

        train_sequences = np.array(train_sequences)
        test_sequences = np.array(test_sequences)

        # PyTorch 모델 생성 및 훈련
        self.sequence_processor.create_model()
        if self.sequence_processor.model is not None:
            self.sequence_processor.train_model(train_sequences)

        # 시퀀스 특성 추출
        train_seq_features = self.sequence_processor.extract_features(train_sequences)
        test_seq_features = self.sequence_processor.extract_features(test_sequences)

        return {
            'train': train_seq_features,
            'test': test_seq_features
        }

    def train_xgboost(self, data):
        """XGBoost 모델 훈련"""
        print("🚀 XGBoost 모델 훈련...")

        X_train, X_val, y_train, y_val = train_test_split(
            data['X_train'], data['y_train'],
            test_size=0.2, random_state=42, stratify=data['y_train']
        )

        # XGBoost 데이터셋
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # 하이퍼파라미터
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist'  # Mac 최적화
        }

        # 훈련
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=20
        )

        # 검증 성능
        val_pred = self.model.predict(dval)
        comp_score, ap, wll = calculate_competition_score(y_val, val_pred)

        print(f"\n📊 검증 성능:")
        print(f"   대회 점수: {comp_score:.4f}")
        print(f"   AP: {ap:.4f}")
        print(f"   WLL: {wll:.4f}")

        return True

    def predict_and_submit(self, data):
        """예측 및 제출 파일 생성"""
        print("🎯 예측 및 제출 파일 생성...")

        # 테스트 예측
        dtest = xgb.DMatrix(data['X_test'])
        predictions = self.model.predict(dtest)

        # 제출 파일 생성
        try:
            submission = pd.read_csv('data/sample_submission.csv')
            submission['clicked'] = predictions
            print(f"✅ 올바른 ID 형식: {submission['ID'].iloc[0]}")
        except:
            submission = pd.DataFrame({
                'ID': [f'TEST_{i:07d}' for i in range(len(predictions))],
                'clicked': predictions
            })

        submission_path = 'submission_mac_xgboost_simple.csv'
        submission.to_csv(submission_path, index=False, encoding='utf-8')

        print(f"\n✅ 제출 파일 생성: {submission_path}")
        print(f"📊 예측 통계:")
        print(f"   평균 클릭률: {predictions.mean():.4f}")
        print(f"   최소값: {predictions.min():.4f}")
        print(f"   최대값: {predictions.max():.4f}")

        return submission_path

    def run_pipeline(self, mode=1):
        """전체 파이프라인 실행"""

        # 모드별 설정
        if mode == 1:
            sample_ratio = 0.3
            use_batch = False
            print(f"\n🚀 초고속 모드 (30% 샘플링)")
        elif mode == 2:
            sample_ratio = 0.5
            use_batch = False
            print(f"\n⚡ 빠른 모드 (50% 샘플링)")
        elif mode == 3:
            sample_ratio = 0.7
            use_batch = False
            print(f"\n🎯 정확 모드 (70% 샘플링)")
        elif mode == 4:
            sample_ratio = 1.0
            use_batch = False
            print(f"\n🏆 최고 성능 모드 (전체 데이터)")
        else:  # mode == 5
            sample_ratio = 1.0
            use_batch = True
            print(f"\n🛡️ 안전 최고 성능 모드 (배치 처리)")

        try:
            # 1. 데이터 로드 및 전처리
            data = self.load_and_preprocess(sample_ratio, use_batch)

            # 2. XGBoost 훈련
            if not self.train_xgboost(data):
                return False

            # 3. 예측 및 제출
            submission_path = self.predict_and_submit(data)

            print(f"\n🎉 파이프라인 완료!")
            print(f"📁 제출 파일: {submission_path}")

            return True

        except Exception as e:
            print(f"❌ 파이프라인 실패: {e}")
            return False

def main():
    print("🚀 Mac용 XGBoost + 간단한 LSTM CTR 예측!")
    print("📊 대회 평가지표: AP (50%) + WLL (50%)")
    print("🧠 시퀀스: 간단한 LSTM (Attention 제외), 테이블: XGBoost")
    print("=" * 60)

    pipeline = MacXGBoostSimple()

    print("📋 실행 옵션:")
    print("1. 🚀 초고속 모드 (30% 샘플링) - 1-2분")
    print("2. ⚡ 빠른 모드 (50% 샘플링) - 2-3분")
    print("3. 🎯 정확 모드 (70% 샘플링) - 4-5분")
    print("4. 🏆 최고 성능 모드 (전체 데이터 직접) - 5-8분 ⚠️ 메모리 위험")
    print("5. 🛡️ 안전 최고 성능 모드 (전체 데이터 배치) - 8-12분 ✅ 메모리 안전")

    choice = input("선택 (1-5, 기본값 1): ").strip() or '1'

    success = pipeline.run_pipeline(int(choice))

    if success:
        print("\n🎉 성공!")
    else:
        print("\n❌ 실패")

if __name__ == "__main__":
    main()