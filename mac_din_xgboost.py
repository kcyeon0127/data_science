#!/usr/bin/env python3
"""
Mac용 DIN + XGBoost 하이브리드 CTR 예측
Apple Silicon 최적화 + 빠른 처리 (2-5분 내 완료)
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Mac 친화적 라이브러리들
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
from tqdm.auto import tqdm
import gc
import time

# TensorFlow (DIN용) - Mac 최적화
print("🔄 TensorFlow 초기화 중... (mutex 메시지는 정상입니다)")
try:
    import tensorflow as tf
    print("📦 TensorFlow 임포트 완료")

    # Apple Silicon 최적화
    if hasattr(tf.config, 'experimental') and hasattr(tf.config.experimental, 'enable_mlcompute'):
        tf.config.experimental.enable_mlcompute()
        print("🍎 Apple MLCompute 활성화됨")

    # GPU 메모리 설정
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("💾 GPU 메모리 설정 완료")

    TF_AVAILABLE = True
    print("✅ TensorFlow Apple Silicon 최적화 활성화")
except ImportError:
    TF_AVAILABLE = False
    print("⚠️ TensorFlow 없음 - XGBoost 단독 모드")

class MacDINXGBoost:
    def __init__(self):
        self.din_model = None
        self.xgb_models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_cols = {
            'tabular': [],
            'sequence': []
        }

        print("🍎 Mac용 DIN + XGBoost 하이브리드 초기화")

    def load_data_efficiently(self, sample_ratio=0.3):
        """효율적 데이터 로딩 (샘플링으로 빠른 처리)"""
        print("📂 효율적 데이터 로딩 중...")

        # 1. 파일 크기 확인
        train_size = os.path.getsize('data/train.parquet') / (1024**3)  # GB
        print(f"📊 훈련 데이터 크기: {train_size:.1f}GB")

        if train_size > 3.0:  # 3GB 이상이면 샘플링
            print(f"🔄 대용량 데이터 감지 - {sample_ratio*100:.0f}% 균형 샘플링 적용")

            # 청크별로 읽으면서 샘플링
            chunks = []
            total_rows = 0

            for chunk in tqdm(pd.read_parquet('data/train.parquet', chunksize=200000),
                            desc="청크별 샘플링"):
                total_rows += len(chunk)

                # 균형 샘플링 (클릭/비클릭 비율 유지)
                clicked = chunk[chunk['clicked'] == 1]
                not_clicked = chunk[chunk['clicked'] == 0]

                sample_clicked = clicked.sample(
                    min(len(clicked), int(len(clicked) * sample_ratio)),
                    random_state=42
                )
                sample_not_clicked = not_clicked.sample(
                    min(len(not_clicked), int(len(not_clicked) * sample_ratio)),
                    random_state=42
                )

                chunk_sample = pd.concat([sample_clicked, sample_not_clicked])
                chunks.append(chunk_sample)

                # 메모리 관리
                if len(chunks) > 20:  # 너무 많은 청크 누적 방지
                    chunks = [pd.concat(chunks[-20:], ignore_index=True)]
                    gc.collect()

            self.train_df = pd.concat(chunks, ignore_index=True)
            print(f"📊 샘플링 결과: {total_rows:,} → {len(self.train_df):,}행")

        else:
            print("📂 전체 데이터 로딩 (크기 적당)")
            self.train_df = pd.read_parquet('data/train.parquet')

        # 테스트 데이터는 항상 전체
        self.test_df = pd.read_parquet('data/test.parquet')

        print(f"✅ 로딩 완료: 훈련 {self.train_df.shape}, 테스트 {self.test_df.shape}")
        print(f"📊 최종 클릭률: {self.train_df['clicked'].mean():.4f}")

        return True

    def preprocess_features(self):
        """특성 분리 및 전처리 (DIN용 vs XGBoost용)"""
        print("🔧 특성 분리 및 전처리...")

        # 1. 특성 분류
        all_cols = self.train_df.columns.tolist()

        # Tabular features (XGBoost용)
        tabular_features = [
            col for col in all_cols
            if col.startswith(('feat_', 'history_')) and col not in ['seq']
        ]

        # Categorical features
        categorical_features = ['gender', 'age_group', 'inventory_id']

        # Sequence features (DIN용)
        sequence_features = ['seq'] if 'seq' in all_cols else []

        print(f"📊 특성 분포:")
        print(f"  Tabular: {len(tabular_features)}개")
        print(f"  Categorical: {len(categorical_features)}개")
        print(f"  Sequence: {len(sequence_features)}개")

        # 2. Tabular features 전처리 (XGBoost용)
        print("🔧 Tabular features 전처리 중...")

        # 수치형 특성 정리
        for col in tqdm(tabular_features, desc="수치형 처리"):
            if col in self.train_df.columns:
                # 결측값 처리
                mean_val = self.train_df[col].fillna(0).mean()
                self.train_df[col] = self.train_df[col].fillna(mean_val)
                self.test_df[col] = self.test_df[col].fillna(mean_val)

        # 카테고리 특성 인코딩
        for col in categorical_features:
            if col in self.train_df.columns:
                print(f"카테고리 인코딩: {col}")
                le = LabelEncoder()

                # 훈련+테스트 데이터 결합하여 학습
                combined = pd.concat([
                    self.train_df[col].fillna('unknown'),
                    self.test_df[col].fillna('unknown')
                ]).astype(str)

                le.fit(combined)
                self.train_df[col] = le.transform(self.train_df[col].fillna('unknown').astype(str))
                self.test_df[col] = le.transform(self.test_df[col].fillna('unknown').astype(str))

                self.encoders[col] = le
                tabular_features.append(col)

        # 3. Sequence features 전처리 (DIN용)
        if sequence_features and TF_AVAILABLE:
            print("🔧 Sequence features 전처리 중...")
            for col in sequence_features:
                if col in self.train_df.columns:
                    # 시퀀스를 고정 길이로 패딩 (DIN에서 필요)
                    self.preprocess_sequence(col, max_len=50)

        # 4. 최종 특성 목록 저장
        self.feature_cols['tabular'] = [col for col in tabular_features if col in self.train_df.columns]
        self.feature_cols['sequence'] = [col for col in sequence_features if col in self.train_df.columns]

        print(f"✅ 전처리 완료:")
        print(f"  최종 Tabular: {len(self.feature_cols['tabular'])}개")
        print(f"  최종 Sequence: {len(self.feature_cols['sequence'])}개")

        return True

    def preprocess_sequence(self, col, max_len=50):
        """시퀀스 데이터 전처리 (DIN용)"""
        print(f"🔄 시퀀스 전처리: {col}")

        # 시퀀스를 리스트로 변환하고 패딩
        def pad_sequence(seq_str, max_len):
            if pd.isna(seq_str) or seq_str == '':
                return [0] * max_len

            try:
                # 문자열을 리스트로 변환 (예: "1,2,3" → [1,2,3])
                seq = [int(x) for x in str(seq_str).split(',')]

                # 패딩 또는 자르기
                if len(seq) >= max_len:
                    return seq[:max_len]
                else:
                    return seq + [0] * (max_len - len(seq))
            except:
                return [0] * max_len

        # 훈련 데이터
        train_seqs = [pad_sequence(seq, max_len) for seq in tqdm(self.train_df[col], desc="훈련 시퀀스")]
        self.train_df[f'{col}_padded'] = train_seqs

        # 테스트 데이터
        test_seqs = [pad_sequence(seq, max_len) for seq in tqdm(self.test_df[col], desc="테스트 시퀀스")]
        self.test_df[f'{col}_padded'] = test_seqs

        # 원본 컬럼 제거
        self.feature_cols['sequence'] = [f'{col}_padded' if c == col else c for c in self.feature_cols['sequence']]

    def create_din_model(self, sequence_vocab_size=10000, embedding_dim=64):
        """DIN 모델 생성 (시퀀스 특성용)"""
        if not TF_AVAILABLE or not self.feature_cols['sequence']:
            print("⚠️ DIN 모델 스킵 (TensorFlow 없음 또는 시퀀스 없음)")
            return None

        print("🧠 DIN 모델 생성 중...")

        # 입력 레이어
        sequence_input = tf.keras.layers.Input(shape=(50,), name='sequence')
        target_input = tf.keras.layers.Input(shape=(1,), name='target')

        # 임베딩 레이어
        item_embedding = tf.keras.layers.Embedding(
            sequence_vocab_size, embedding_dim, mask_zero=True
        )

        sequence_emb = item_embedding(sequence_input)  # (batch, 50, 64)
        target_emb = item_embedding(target_input)      # (batch, 1, 64)

        # DIN Attention
        target_expanded = tf.keras.layers.RepeatVector(50)(tf.squeeze(target_emb, axis=1))  # (batch, 50, 64)

        # Attention weights 계산
        attention_input = tf.keras.layers.Concatenate()([
            sequence_emb, target_expanded, sequence_emb * target_expanded
        ])  # (batch, 50, 192)

        attention_weights = tf.keras.layers.Dense(64, activation='relu')(attention_input)
        attention_weights = tf.keras.layers.Dense(1, activation='sigmoid')(attention_weights)  # (batch, 50, 1)

        # Attention 적용
        attended_sequence = tf.keras.layers.Multiply()([sequence_emb, attention_weights])  # (batch, 50, 64)
        sequence_repr = tf.keras.layers.GlobalAveragePooling1D()(attended_sequence)  # (batch, 64)

        # 최종 출력
        output = tf.keras.layers.Dense(32, activation='relu')(sequence_repr)
        output = tf.keras.layers.Dropout(0.3)(output)
        sequence_features = tf.keras.layers.Dense(16, activation='relu', name='sequence_features')(output)

        model = tf.keras.Model(inputs=[sequence_input, target_input], outputs=sequence_features)
        model.compile(optimizer='adam', loss='mse')

        print("✅ DIN 모델 생성 완료")
        return model

    def train_xgboost_ensemble(self, X_tabular, y, cv_folds=3):
        """XGBoost 앙상블 훈련 (Mac 최적화)"""
        print("🚀 XGBoost 앙상블 훈련 중...")

        # Mac 최적화 XGBoost 파라미터
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',  # Mac에서 안전한 방법
            'random_state': 42,
            'verbosity': 0,
            'n_jobs': -1  # Mac 멀티코어 활용
        }

        # 다양한 XGBoost 설정으로 앙상블
        model_configs = {
            'xgb_conservative': {
                **base_params,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'xgb_aggressive': {
                **base_params,
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 200,
                'subsample': 0.9,
                'colsample_bytree': 0.9
            }
        }

        # 클래스 불균형 처리
        pos_ratio = y.mean()
        scale_pos_weight = (1 - pos_ratio) / pos_ratio
        print(f"📊 클릭률: {pos_ratio:.4f}, Scale pos weight: {scale_pos_weight:.2f}")

        for config in model_configs.values():
            config['scale_pos_weight'] = scale_pos_weight

        # 교차 검증으로 모델 훈련
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        for name, params in model_configs.items():
            print(f"훈련 중: {name}")

            cv_scores = []
            models = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_tabular, y)):
                X_train_fold = X_tabular.iloc[train_idx]
                X_val_fold = X_tabular.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]

                # 모델 훈련
                model = xgb.XGBClassifier(**params)
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    early_stopping_rounds=10,
                    verbose=False
                )

                # 검증
                val_pred = model.predict_proba(X_val_fold)[:, 1]
                auc = roc_auc_score(y_val_fold, val_pred)
                cv_scores.append(auc)
                models.append(model)

            print(f"  {name} CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
            self.xgb_models[name] = models

        print("✅ XGBoost 앙상블 훈련 완료")

    def predict_and_submit(self, return_features=False):
        """예측 및 제출 파일 생성"""
        print("🎯 예측 시작...")

        # 1. Tabular features로 XGBoost 예측
        X_test_tabular = self.test_df[self.feature_cols['tabular']]

        xgb_predictions = []
        for name, models in self.xgb_models.items():
            fold_preds = []
            for model in models:
                pred = model.predict_proba(X_test_tabular)[:, 1]
                fold_preds.append(pred)

            avg_pred = np.mean(fold_preds, axis=0)
            xgb_predictions.append(avg_pred)
            print(f"{name} 예측 완료")

        # 2. XGBoost 앙상블 평균
        final_predictions = np.mean(xgb_predictions, axis=0)

        # 3. DIN 특성이 있다면 결합 (현재는 XGBoost 단독)
        if self.din_model and self.feature_cols['sequence']:
            print("🧠 DIN 특성 추가...")
            # DIN 예측 코드 (필요시 구현)
            pass

        # 4. 제출 파일 생성 (올바른 형식)
        try:
            submission = pd.read_csv('data/sample_submission.csv')
            submission['clicked'] = final_predictions
            print(f"✅ 올바른 ID 형식: {submission['ID'].iloc[0]}")
        except:
            submission = pd.DataFrame({
                'ID': [f'TEST_{i:07d}' for i in range(len(final_predictions))],
                'clicked': final_predictions
            })

        submission_path = 'submission_mac_din_xgboost.csv'
        submission.to_csv(submission_path, index=False, encoding='utf-8')

        print(f"✅ 제출 파일 생성: {submission_path}")
        print(f"📊 예측 통계:")
        print(f"  평균 클릭률: {final_predictions.mean():.4f}")
        print(f"  최소값: {final_predictions.min():.4f}")
        print(f"  최대값: {final_predictions.max():.4f}")

        return submission_path

    def run_pipeline(self, sample_ratio=0.3):
        """전체 파이프라인 실행"""
        print("🚀 Mac용 DIN + XGBoost 하이브리드 파이프라인 시작!")
        print("=" * 60)

        start_time = time.time()

        # 1. 데이터 로딩
        if not self.load_data_efficiently(sample_ratio):
            return False

        # 2. 특성 전처리
        if not self.preprocess_features():
            return False

        # 3. DIN 모델 생성 (시퀀스가 있는 경우)
        if self.feature_cols['sequence'] and TF_AVAILABLE:
            self.din_model = self.create_din_model()

        # 4. XGBoost 훈련
        X_tabular = self.train_df[self.feature_cols['tabular']]
        y = self.train_df['clicked']

        self.train_xgboost_ensemble(X_tabular, y)

        # 5. 예측 및 제출
        submission_path = self.predict_and_submit()

        elapsed = time.time() - start_time
        print("\n" + "🎉" * 60)
        print("파이프라인 완료!")
        print("🎉" * 60)
        print(f"⏱️ 총 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
        print(f"📁 제출 파일: {submission_path}")

        return True

def main():
    print("🍎 Mac용 DIN + XGBoost 하이브리드 CTR 예측")
    print("=" * 60)

    pipeline = MacDINXGBoost()

    # 빠른 실행을 위한 샘플링 옵션
    print("📋 실행 옵션:")
    print("1. 빠른 모드 (30% 샘플링) - 2-3분")
    print("2. 균형 모드 (50% 샘플링) - 4-5분")
    print("3. 전체 모드 (100% 데이터) - 10-15분")

    choice = input("선택 (1-3, 기본값 1): ").strip() or '1'

    sample_ratios = {'1': 0.3, '2': 0.5, '3': 1.0}
    sample_ratio = sample_ratios.get(choice, 0.3)

    print(f"🚀 {sample_ratio*100:.0f}% 데이터로 실행 시작...")

    success = pipeline.run_pipeline(sample_ratio)

    if success:
        print("\n✅ 성공! 제출 파일이 생성되었습니다.")
    else:
        print("\n❌ 실행 실패")

if __name__ == "__main__":
    main()