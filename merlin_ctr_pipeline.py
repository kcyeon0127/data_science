#!/usr/bin/env python3
"""
NVIDIA Merlin 기반 CTR 예측 파이프라인
Attention 메커니즘 포함 + 최종 제출 CSV 생성
"""

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Merlin 라이브러리
try:
    import cudf
    import nvtabular as nvt
    from nvtabular import ops
    import merlin.models.tf as mm
    from merlin.io import Dataset
    from merlin.schema import Tags
    import tensorflow as tf

    print("✅ Merlin 라이브러리 로드 성공")
    MERLIN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Merlin 라이브러리 없음: {e}")
    print("💡 설치 명령어: pip install merlin-dataloader merlin-models nvtabular cudf-cu11")
    MERLIN_AVAILABLE = False

class MerlinCTRPipeline:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, 'train.parquet')
        self.test_path = os.path.join(data_dir, 'test.parquet')
        self.model = None
        self.workflow = None

        # GPU 설정
        if MERLIN_AVAILABLE:
            try:
                # GPU 메모리 제한 설정
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    print("✅ GPU 메모리 설정 완료")
            except:
                print("⚠️ GPU 설정 실패, CPU 사용")

    def load_data(self):
        """데이터 로드 (Merlin 최적화)"""
        print("📂 데이터 로드 중...")

        if not os.path.exists(self.train_path):
            print(f"❌ {self.train_path} 파일이 없습니다.")
            return False

        try:
            if MERLIN_AVAILABLE:
                # CuDF로 로드 (GPU 가속)
                self.train_df = cudf.read_parquet(self.train_path)
                self.test_df = cudf.read_parquet(self.test_path)
                print(f"✅ GPU 가속 로드: 훈련 {self.train_df.shape}, 테스트 {self.test_df.shape}")
            else:
                # Pandas 폴백
                self.train_df = pd.read_parquet(self.train_path)
                self.test_df = pd.read_parquet(self.test_path)
                print(f"✅ CPU 로드: 훈련 {self.train_df.shape}, 테스트 {self.test_df.shape}")

            return True

        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False

    def create_nvtabular_workflow(self):
        """NVTabular 전처리 워크플로우 생성"""
        print("🔧 전처리 워크플로우 생성 중...")

        # 카테고리 컬럼들
        categorical_cols = ['gender', 'age_group', 'inventory_id', 'l_feat_14']

        # 연속형 컬럼들
        continuous_cols = [col for col in self.train_df.columns
                          if col.startswith(('feat_', 'history_')) and col not in categorical_cols]

        # 시퀀스 컬럼
        sequence_cols = ['seq'] if 'seq' in self.train_df.columns else []

        print(f"📊 컬럼 정보:")
        print(f"  카테고리: {len(categorical_cols)}개")
        print(f"  연속형: {len(continuous_cols)}개")
        print(f"  시퀀스: {len(sequence_cols)}개")

        # 전처리 파이프라인 구성
        cat_features = categorical_cols >> ops.Categorify(out_dtype="int32")
        cont_features = continuous_cols >> ops.FillMissing() >> ops.Normalize()

        # 시퀀스 처리 (있는 경우)
        if sequence_cols:
            # 시퀀스를 고정 길이로 패딩
            seq_features = sequence_cols >> ops.ListSlice(0, 50) >> ops.Categorify(out_dtype="int32")
            workflow_ops = cat_features + cont_features + seq_features
        else:
            workflow_ops = cat_features + cont_features

        # 타겟 컬럼
        target = ['clicked'] >> ops.LambdaOp(lambda x: x.astype('float32'))

        self.workflow = nvt.Workflow(workflow_ops + target)

        return True

    def preprocess_data(self):
        """데이터 전처리 실행"""
        print("⚡ GPU 가속 전처리 시작...")

        try:
            # 훈련 데이터 전처리
            train_dataset = Dataset(self.train_df)
            self.workflow.fit(train_dataset)

            train_processed = self.workflow.transform(train_dataset)
            test_dataset = Dataset(self.test_df)
            test_processed = self.workflow.transform(test_dataset)

            print("✅ 전처리 완료")

            return train_processed, test_processed

        except Exception as e:
            print(f"❌ 전처리 실패: {e}")
            return None, None

    def create_attention_model(self, schema):
        """Attention 기반 CTR 모델 생성"""
        print("🧠 Attention 기반 모델 생성 중...")

        # 입력 블록
        inputs = mm.InputBlockV2(schema)

        # 임베딩 레이어
        embeddings = mm.EmbeddingFeatures(inputs)

        # Multi-Head Attention 블록
        attention_output = mm.Block([
            mm.MLPBlock([512, 256], activation="relu", dropout=0.2),
            # 시퀀스가 있는 경우 Attention 적용
            mm.SequenceEmbeddingFeatures(
                aggregation="concat",
                schema=schema
            ) if any("seq" in col.name for col in schema) else mm.MLPBlock([256]),
        ])(embeddings)

        # Multi-Head Self-Attention (커스텀)
        attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            dropout=0.1,
            name="multi_head_attention"
        )

        # Attention 적용 (3D 텐서가 필요하므로 reshape)
        reshaped = tf.keras.layers.Reshape((-1, 1))(attention_output)
        attended = attention_layer(reshaped, reshaped)
        flattened = tf.keras.layers.Flatten()(attended)

        # 최종 예측 레이어
        dense_layers = mm.MLPBlock([256, 128, 64], activation="relu", dropout=0.3)(flattened)

        # 이진 분류 태스크
        predictions = mm.BinaryClassificationTask("clicked")(dense_layers)

        # 모델 생성
        model = mm.Model(inputs, predictions)

        return model

    def train_model(self, train_dataset, valid_dataset=None):
        """모델 훈련"""
        print("🚀 모델 훈련 시작...")

        try:
            schema = train_dataset.schema

            # Attention 모델 생성
            self.model = self.create_attention_model(schema)

            # 컴파일
            self.model.compile(
                optimizer='adam',
                run_eagerly=False,
                metrics=[
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
            )

            print("📋 모델 구조:")
            self.model.summary()

            # 콜백 설정
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if valid_dataset else 'loss',
                    patience=3,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if valid_dataset else 'loss',
                    factor=0.5,
                    patience=2
                )
            ]

            # 훈련
            history = self.model.fit(
                train_dataset,
                validation_data=valid_dataset,
                epochs=10,
                batch_size=4096,
                callbacks=callbacks,
                verbose=1
            )

            print("✅ 모델 훈련 완료")
            return history

        except Exception as e:
            print(f"❌ 모델 훈련 실패: {e}")
            return None

    def predict_and_submit(self, test_dataset):
        """예측 및 제출 파일 생성"""
        print("🎯 예측 시작...")

        try:
            # 예측
            predictions = self.model.predict(test_dataset, batch_size=8192)

            # 제출 파일 형태로 변환
            if hasattr(predictions, 'numpy'):
                pred_probs = predictions.numpy().flatten()
            else:
                pred_probs = predictions.flatten()

            # 테스트 데이터의 ID 가져오기
            if hasattr(self.test_df, 'to_pandas'):
                test_ids = self.test_df.to_pandas().index
            else:
                test_ids = self.test_df.index

            # 제출 파일 생성
            submission = pd.DataFrame({
                'id': test_ids,
                'clicked': pred_probs
            })

            # 저장
            submission_path = 'submission_merlin_attention.csv'
            submission.to_csv(submission_path, index=False)

            print(f"✅ 제출 파일 생성: {submission_path}")
            print(f"📊 예측 통계:")
            print(f"  평균 클릭률: {pred_probs.mean():.4f}")
            print(f"  최소값: {pred_probs.min():.4f}")
            print(f"  최대값: {pred_probs.max():.4f}")

            return submission_path

        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            return None

    def run_pipeline(self):
        """전체 파이프라인 실행"""
        print("🚀 Merlin CTR 파이프라인 시작!")

        # 1. 데이터 로드
        if not self.load_data():
            return False

        # 2. 전처리 워크플로우 생성
        if not self.create_nvtabular_workflow():
            return False

        # 3. 데이터 전처리
        train_processed, test_processed = self.preprocess_data()
        if train_processed is None:
            return False

        # 4. Train/Validation 분할
        train_len = len(train_processed)
        val_len = int(train_len * 0.2)

        valid_processed = train_processed.to_ddf().tail(val_len)
        train_processed = train_processed.to_ddf().head(train_len - val_len)

        print(f"📊 데이터 분할: 훈련 {train_len - val_len:,}, 검증 {val_len:,}")

        # 5. 모델 훈련
        history = self.train_model(train_processed, valid_processed)
        if history is None:
            return False

        # 6. 예측 및 제출 파일 생성
        submission_path = self.predict_and_submit(test_processed)
        if submission_path is None:
            return False

        print("🎉 파이프라인 완료!")
        return True

def fallback_baseline():
    """Merlin이 없을 때의 간단한 베이스라인"""
    print("🔄 베이스라인 모델로 폴백...")

    try:
        # 데이터 로드
        train_df = pd.read_parquet('data/train.parquet')
        test_df = pd.read_parquet('data/test.parquet')

        print(f"📊 데이터 크기: 훈련 {train_df.shape}, 테스트 {test_df.shape}")

        # 간단한 특성 선택
        numeric_cols = [col for col in train_df.columns if col.startswith('feat_')]

        X = train_df[numeric_cols].fillna(0)
        y = train_df['clicked']
        X_test = test_df[numeric_cols].fillna(0)

        # 간단한 모델 훈련
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_scaled, y)

        # 예측
        predictions = model.predict_proba(X_test_scaled)[:, 1]

        # 제출 파일
        submission = pd.DataFrame({
            'id': test_df.index,
            'clicked': predictions
        })

        submission.to_csv('submission_baseline.csv', index=False)
        print("✅ 베이스라인 제출 파일 생성: submission_baseline.csv")

        return True

    except Exception as e:
        print(f"❌ 베이스라인 실패: {e}")
        return False

def main():
    if not MERLIN_AVAILABLE:
        print("💡 Merlin을 설치하면 GPU 가속과 Attention 모델을 사용할 수 있습니다!")
        print("📝 설치 방법:")
        print("  conda install -c nvidia -c rapidsai -c conda-forge cudf nvtabular")
        print("  pip install merlin-models")
        print()

        choice = input("베이스라인 모델로 진행하시겠습니까? (y/N): ")
        if choice.lower() == 'y':
            return fallback_baseline()
        else:
            print("Merlin 설치 후 다시 실행해주세요.")
            return False

    # Merlin 파이프라인 실행
    pipeline = MerlinCTRPipeline()
    success = pipeline.run_pipeline()

    if success:
        print("🎉 Merlin CTR 파이프라인 완료!")
        print("📁 생성된 파일:")
        print("  - submission_merlin_attention.csv (최종 제출 파일)")
    else:
        print("❌ 파이프라인 실패")
        print("💡 베이스라인으로 폴백 시도...")
        fallback_baseline()

if __name__ == "__main__":
    main()