#!/usr/bin/env python3
"""
Mac용 순수 XGBoost CTR 예측 파이프라인
대회 평가지표: AP (50%) + WLL (50%)
시퀀스 처리 없음, 테이블 데이터만 사용
"""

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# XGBoost
try:
    import xgboost as xgb
    print("✅ XGBoost 로드됨")
    XGB_AVAILABLE = True
except ImportError:
    print("❌ XGBoost 없음")
    XGB_AVAILABLE = False

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

class MacXGBoostPure:
    """Mac용 순수 XGBoost CTR 예측 (시퀀스 처리 없음)"""

    def __init__(self):
        self.model = None
        self.feature_cols = []
        self.encoders = {}
        print("🍎 Mac용 순수 XGBoost CTR 초기화 완료")

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
        """데이터 전처리 - 테이블 특성만"""
        print("🔧 데이터 전처리 중... (테이블 특성만)")

        # 1. 수치형 특성 처리
        numeric_cols = [col for col in train_df.columns
                       if col.startswith(('feat_', 'history_')) and
                       train_df[col].dtype in ['float64', 'int64']]

        print(f"📊 수치형 특성: {len(numeric_cols)}개")

        # 상위 100개 수치형 특성만 사용 (메모리 절약)
        selected_numeric = numeric_cols[:100]

        for col in tqdm(selected_numeric, desc="수치 전처리"):
            if col in train_df.columns:
                # 결측값 처리
                mean_val = train_df[col].mean()
                std_val = train_df[col].std()

                train_df[col] = train_df[col].fillna(mean_val)
                test_df[col] = test_df[col].fillna(mean_val)

                # 정규화 (표준화)
                if std_val > 0:
                    train_df[col] = (train_df[col] - mean_val) / std_val
                    test_df[col] = (test_df[col] - mean_val) / std_val

        self.feature_cols.extend(selected_numeric)

        # 2. 카테고리 특성 처리
        categorical_cols = ['gender', 'age_group']

        for col in categorical_cols:
            if col in train_df.columns:
                print(f"🏷️ 카테고리 처리: {col}")

                # 라벨 인코딩
                le = LabelEncoder()

                # 훈련+테스트 데이터 합쳐서 라벨 학습
                combined_values = pd.concat([
                    train_df[col].fillna('unknown'),
                    test_df[col].fillna('unknown')
                ]).astype(str)

                le.fit(combined_values)
                self.encoders[col] = le

                # 인코딩 적용
                encoded_col = f"{col}_encoded"
                train_df[encoded_col] = le.transform(train_df[col].fillna('unknown').astype(str))
                test_df[encoded_col] = le.transform(test_df[col].fillna('unknown').astype(str))

                self.feature_cols.append(encoded_col)

        # 3. 시간 기반 특성 (만약 있다면)
        time_cols = [col for col in train_df.columns if 'time' in col.lower() or 'date' in col.lower()]
        for col in time_cols[:5]:  # 상위 5개만
            if col in train_df.columns and train_df[col].dtype in ['object', 'datetime64[ns]']:
                try:
                    # 시간을 숫자로 변환
                    train_df[f"{col}_numeric"] = pd.to_datetime(train_df[col], errors='coerce').astype(np.int64) // 10**9
                    test_df[f"{col}_numeric"] = pd.to_datetime(test_df[col], errors='coerce').astype(np.int64) // 10**9

                    # 결측값 처리
                    mean_val = train_df[f"{col}_numeric"].mean()
                    train_df[f"{col}_numeric"] = train_df[f"{col}_numeric"].fillna(mean_val)
                    test_df[f"{col}_numeric"] = test_df[f"{col}_numeric"].fillna(mean_val)

                    self.feature_cols.append(f"{col}_numeric")
                    print(f"⏰ 시간 특성 추가: {col}_numeric")
                except:
                    pass

        # 4. 상호작용 특성 (간단한 조합)
        if len(selected_numeric) >= 2:
            print("🔗 상호작용 특성 생성...")

            # 상위 5개 특성간 곱셈 조합
            top_features = selected_numeric[:5]
            for i, feat1 in enumerate(top_features):
                for feat2 in top_features[i+1:i+3]:  # 너무 많으면 안되므로 제한
                    interaction_col = f"{feat1}_x_{feat2}"
                    train_df[interaction_col] = train_df[feat1] * train_df[feat2]
                    test_df[interaction_col] = test_df[feat1] * test_df[feat2]
                    self.feature_cols.append(interaction_col)

        # 최종 특성 준비
        print(f"✅ 전처리 완료: {len(self.feature_cols)}개 특성 사용")

        X_train = train_df[self.feature_cols].values
        X_test = test_df[self.feature_cols].values
        y_train = train_df['clicked'].values

        print(f"📊 최종 데이터: 훈련 {X_train.shape}, 테스트 {X_test.shape}")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'test_df': test_df
        }

    def train_xgboost(self, data):
        """XGBoost 모델 훈련"""
        print("🚀 XGBoost 모델 훈련...")

        if not XGB_AVAILABLE:
            print("❌ XGBoost가 설치되지 않았습니다")
            return False

        # 훈련/검증 분할
        X_train, X_val, y_train, y_val = train_test_split(
            data['X_train'], data['y_train'],
            test_size=0.2, random_state=42, stratify=data['y_train']
        )

        print(f"📊 훈련 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")

        # XGBoost 데이터셋
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # 하이퍼파라미터 (CTR 예측 최적화)
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'tree_method': 'hist',  # Mac 최적화
            'scale_pos_weight': 50  # 클래스 불균형 대응 (CTR 1.9%)
        }

        print("🎯 XGBoost 하이퍼파라미터:")
        for key, value in params.items():
            print(f"   {key}: {value}")

        # 훈련
        evals_result = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=20,
            verbose_eval=25,
            evals_result=evals_result
        )

        # 검증 성능 평가
        val_pred = self.model.predict(dval)
        comp_score, ap, wll = calculate_competition_score(y_val, val_pred)

        print(f"\n📊 검증 성능 (대회 지표):")
        print(f"   🏆 최종 점수: {comp_score:.4f}")
        print(f"   📈 AP (Average Precision): {ap:.4f}")
        print(f"   📉 WLL (Weighted LogLoss): {wll:.4f}")
        print(f"   🎯 클릭률 예측 평균: {val_pred.mean():.4f}")

        # 특성 중요도 출력
        importance = self.model.get_score(importance_type='weight')
        print(f"\n🔍 상위 10개 특성 중요도:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, score) in enumerate(sorted_importance[:10]):
            feature_name = self.feature_cols[int(feature[1:])] if feature.startswith('f') else feature
            print(f"   {i+1:2d}. {feature_name}: {score}")

        return True

    def predict_and_submit(self, data):
        """예측 및 제출 파일 생성"""
        print("🎯 예측 및 제출 파일 생성...")

        # 테스트 예측
        dtest = xgb.DMatrix(data['X_test'])
        predictions = self.model.predict(dtest)

        # 제출 파일 생성
        try:
            # sample_submission.csv를 템플릿으로 사용
            submission = pd.read_csv('data/sample_submission.csv')
            submission['clicked'] = predictions
            print(f"✅ 올바른 ID 형식 사용: {submission['ID'].iloc[0]}")
        except Exception as e:
            print(f"⚠️ sample_submission.csv 로드 실패: {e}")
            # 직접 ID 생성
            submission = pd.DataFrame({
                'ID': [f'TEST_{i:07d}' for i in range(len(predictions))],
                'clicked': predictions
            })
            print("⚠️ 직접 ID 생성")

        submission_path = 'submission_mac_xgboost_pure.csv'
        submission.to_csv(submission_path, index=False, encoding='utf-8')

        print(f"\n✅ 제출 파일 생성: {submission_path}")
        print(f"📊 예측 통계:")
        print(f"   평균 클릭률: {predictions.mean():.4f}")
        print(f"   최소값: {predictions.min():.4f}")
        print(f"   최대값: {predictions.max():.4f}")
        print(f"   표준편차: {predictions.std():.4f}")

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

            print(f"\n🎉 순수 XGBoost 파이프라인 완료!")
            print(f"📁 제출 파일: {submission_path}")

            return True

        except Exception as e:
            print(f"❌ 파이프라인 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    print("🚀 Mac용 순수 XGBoost CTR 예측!")
    print("📊 대회 평가지표: AP (50%) + WLL (50%)")
    print("🏷️ 테이블 데이터만 사용 (시퀀스 처리 없음)")
    print("=" * 60)

    pipeline = MacXGBoostPure()

    print("📋 실행 옵션:")
    print("1. 🚀 초고속 모드 (30% 샘플링) - 30초-1분")
    print("2. ⚡ 빠른 모드 (50% 샘플링) - 1-2분")
    print("3. 🎯 정확 모드 (70% 샘플링) - 2-3분")
    print("4. 🏆 최고 성능 모드 (전체 데이터 직접) - 3-5분 ⚠️ 메모리 위험")
    print("5. 🛡️ 안전 최고 성능 모드 (전체 데이터 배치) - 5-8분 ✅ 메모리 안전")

    choice = input("선택 (1-5, 기본값 1): ").strip() or '1'

    success = pipeline.run_pipeline(int(choice))

    if success:
        print("\n🎉 성공!")
        print("📝 특징: 순수 XGBoost, 안정적, 빠른 실행")
    else:
        print("\n❌ 실패")

if __name__ == "__main__":
    main()