#!/usr/bin/env python3
"""
맥북 최적화 CTR 예측 파이프라인
Apple Silicon (M1/M2/M3) 및 Intel Mac 지원
"""

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# TensorFlow 맥 최적화
try:
    import tensorflow as tf

    # Apple Silicon 최적화 설정
    if hasattr(tf.config, 'experimental') and hasattr(tf.config.experimental, 'enable_mlcompute'):
        tf.config.experimental.enable_mlcompute()
        print("✅ Apple MLCompute 활성화")

    # Metal Performance Shaders 사용
    try:
        from tensorflow.python.framework.config import set_memory_growth
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("✅ Apple GPU 메모리 최적화")
    except:
        pass

    TF_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow 없음")
    TF_AVAILABLE = False

class MacOptimizedCTR:
    def __init__(self):
        self.model = None
        self.preprocessor = None

        # 맥 시스템 정보
        import platform
        self.system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'is_apple_silicon': 'arm64' in platform.machine().lower()
        }

        print(f"🍎 Mac 시스템 정보:")
        print(f"  플랫폼: {self.system_info['platform']}")
        print(f"  Apple Silicon: {self.system_info['is_apple_silicon']}")

    def load_and_preprocess_data(self):
        """맥북 친화적 데이터 로딩 및 전처리"""
        print("📂 데이터 로딩 중...")

        try:
            # 청크별로 안전하게 로드
            train_chunks = []
            chunk_size = 500000  # 50만행씩

            for chunk in tqdm(pd.read_parquet('data/train.parquet', chunksize=chunk_size),
                            desc="훈련 데이터 로드"):
                train_chunks.append(chunk)

            self.train_df = pd.concat(train_chunks, ignore_index=True)
            self.test_df = pd.read_parquet('data/test.parquet')

            print(f"✅ 데이터 로드 완료: 훈련 {self.train_df.shape}, 테스트 {self.test_df.shape}")

            return self.preprocess_features()

        except Exception as e:
            print(f"❌ 청크 로딩 실패, 전체 로드 시도...")

            # 폴백: 메모리 모니터링하며 전체 로드
            try:
                self.train_df = pd.read_parquet('data/train.parquet')
                self.test_df = pd.read_parquet('data/test.parquet')
                print(f"✅ 전체 로드 성공: 훈련 {self.train_df.shape}, 테스트 {self.test_df.shape}")
                return self.preprocess_features()
            except Exception as e2:
                print(f"❌ 데이터 로드 실패: {e2}")
                return False

    def preprocess_features(self):
        """맥북 최적화 특성 전처리"""
        print("🔧 특성 전처리 중...")

        try:
            # 메모리 효율적 전처리

            # 1. 수치형 특성 처리
            numeric_cols = [col for col in self.train_df.columns
                           if col.startswith(('feat_', 'history_')) and self.train_df[col].dtype in ['float64', 'int64']]

            print(f"📊 수치형 특성: {len(numeric_cols)}개")

            # 결측값 처리 및 정규화
            for col in tqdm(numeric_cols, desc="수치형 처리"):
                # 결측값 처리
                mean_val = self.train_df[col].mean()
                self.train_df[col] = self.train_df[col].fillna(mean_val)
                self.test_df[col] = self.test_df[col].fillna(mean_val)

                # 간단한 정규화 (메모리 효율적)
                std_val = self.train_df[col].std()
                if std_val > 0:
                    self.train_df[col] = (self.train_df[col] - mean_val) / std_val
                    self.test_df[col] = (self.test_df[col] - mean_val) / std_val

            # 2. 카테고리 특성 처리 (간단한 라벨 인코딩)
            categorical_cols = ['gender', 'age_group']

            for col in categorical_cols:
                if col in self.train_df.columns:
                    # 라벨 인코딩
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()

                    # 훈련+테스트 데이터 합쳐서 라벨 학습
                    combined_values = pd.concat([
                        self.train_df[col].fillna('unknown'),
                        self.test_df[col].fillna('unknown')
                    ]).astype(str)

                    le.fit(combined_values)

                    self.train_df[col] = le.transform(self.train_df[col].fillna('unknown').astype(str))
                    self.test_df[col] = le.transform(self.test_df[col].fillna('unknown').astype(str))

            # 특성 선택 (메모리 절약)
            feature_cols = numeric_cols + categorical_cols
            self.feature_cols = [col for col in feature_cols if col in self.train_df.columns]

            print(f"✅ 전처리 완료: {len(self.feature_cols)}개 특성 사용")
            return True

        except Exception as e:
            print(f"❌ 전처리 실패: {e}")
            return False

    def create_mac_optimized_model(self, input_dim):
        """맥북 최적화 모델 (TensorFlow Metal 활용)"""

        if not TF_AVAILABLE:
            return self.create_sklearn_model()

        print("🧠 TensorFlow 모델 생성 중...")

        try:
            # Apple Silicon 최적화 모델
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
                tf.keras.layers.Dropout(0.3),

                # Attention-like 레이어 (간소화)
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),

                # Self-attention 시뮬레이션
                tf.keras.layers.Dense(64, activation='tanh'),  # Query
                tf.keras.layers.Dense(64, activation='relu'),  # Key/Value
                tf.keras.layers.Dropout(0.1),

                # 출력 레이어
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            # 맥 최적화 컴파일
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['auc', 'precision', 'recall']
            )

            print("✅ TensorFlow 모델 생성 완료")
            return model

        except Exception as e:
            print(f"⚠️ TensorFlow 모델 실패: {e}")
            print("🔄 Scikit-learn 모델로 폴백...")
            return self.create_sklearn_model()

    def create_sklearn_model(self):
        """Scikit-learn 기반 폴백 모델"""
        print("🔧 Scikit-learn 모델 생성...")

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        # Apple Silicon에서 잘 작동하는 모델들
        models = {
            'rf': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1  # 멀티코어 활용
            ),
            'lr': LogisticRegression(
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            )
        }

        return models

    def train_model(self):
        """모델 훈련"""
        print("🚀 모델 훈련 시작...")

        try:
            # 특성과 타겟 분리
            X = self.train_df[self.feature_cols].values
            y = self.train_df['clicked'].values

            print(f"📊 훈련 데이터: {X.shape}")

            # Train/Val 분할
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self.model = self.create_mac_optimized_model(X.shape[1])

            if isinstance(self.model, dict):
                # Scikit-learn 모델들
                print("🔧 Scikit-learn 앙상블 훈련...")

                trained_models = {}
                for name, model in self.model.items():
                    print(f"훈련 중: {name}")
                    model.fit(X_train, y_train)

                    # 검증 성능
                    from sklearn.metrics import roc_auc_score
                    val_pred = model.predict_proba(X_val)[:, 1]
                    auc = roc_auc_score(y_val, val_pred)
                    print(f"  {name} 검증 AUC: {auc:.4f}")

                    trained_models[name] = model

                self.model = trained_models

            else:
                # TensorFlow 모델
                print("🚀 TensorFlow 모델 훈련...")

                # 콜백 설정
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=3, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.5, patience=2
                    )
                ]

                # 훈련
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=1024,  # 맥북 친화적 배치 크기
                    callbacks=callbacks,
                    verbose=1
                )

                print("✅ TensorFlow 모델 훈련 완료")

            return True

        except Exception as e:
            print(f"❌ 모델 훈련 실패: {e}")
            return False

    def predict_and_submit(self):
        """예측 및 제출 파일 생성"""
        print("🎯 예측 및 제출 파일 생성...")

        try:
            # 테스트 데이터 준비
            X_test = self.test_df[self.feature_cols].values

            if isinstance(self.model, dict):
                # Scikit-learn 앙상블 예측
                predictions = []
                for name, model in self.model.items():
                    pred = model.predict_proba(X_test)[:, 1]
                    predictions.append(pred)
                    print(f"{name} 예측 완료")

                # 앙상블 (평균)
                final_predictions = np.mean(predictions, axis=0)

            else:
                # TensorFlow 예측
                final_predictions = self.model.predict(X_test, batch_size=2048).flatten()

            # 제출 파일 생성 (올바른 형식)
            try:
                submission = pd.read_csv('data/sample_submission.csv')
                submission['clicked'] = final_predictions
                print(f"✅ 올바른 ID 형식: {submission['ID'].iloc[0]}")
            except:
                submission = pd.DataFrame({
                    'ID': [f'TEST_{i:07d}' for i in range(len(final_predictions))],
                    'clicked': final_predictions
                })

            submission_path = 'submission_mac_optimized.csv'
            submission.to_csv(submission_path, index=False, encoding='utf-8')

            print(f"✅ 제출 파일 생성: {submission_path}")
            print(f"📊 예측 통계:")
            print(f"  평균 클릭률: {final_predictions.mean():.4f}")
            print(f"  최소값: {final_predictions.min():.4f}")
            print(f"  최대값: {final_predictions.max():.4f}")

            return submission_path

        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            return None

    def run_pipeline(self):
        """전체 파이프라인 실행"""
        print("🍎 맥북 최적화 CTR 파이프라인 시작!")

        # 1. 데이터 로드 및 전처리
        if not self.load_and_preprocess_data():
            return False

        # 2. 모델 훈련
        if not self.train_model():
            return False

        # 3. 예측 및 제출
        submission_path = self.predict_and_submit()
        if submission_path is None:
            return False

        print("🎉 맥북 CTR 파이프라인 완료!")
        return True

def main():
    print("🍎 맥북 전용 CTR 예측 시스템")
    print("=" * 50)

    pipeline = MacOptimizedCTR()
    success = pipeline.run_pipeline()

    if success:
        print("\n🎉 성공!")
        print("📁 생성된 파일: submission_mac_optimized.csv")
    else:
        print("\n❌ 실패")

if __name__ == "__main__":
    main()