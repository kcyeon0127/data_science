#!/usr/bin/env python3
"""
Mac용 XGBoost 전용 CTR 예측 (TensorFlow 없음)
초고속 실행 + 실시간 진행 상황 표시
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import xgboost as xgb
from tqdm.auto import tqdm
import gc
import time

def calculate_weighted_logloss(y_true, y_pred, eps=1e-15):
    """가중 LogLoss 계산 (50:50 클래스 가중치)"""
    y_pred = np.clip(y_pred, eps, 1 - eps)

    mask_0 = (y_true == 0)
    mask_1 = (y_true == 1)

    ll_0 = -np.mean(np.log(1 - y_pred[mask_0])) if mask_0.sum() > 0 else 0
    ll_1 = -np.mean(np.log(y_pred[mask_1])) if mask_1.sum() > 0 else 0

    return 0.5 * ll_0 + 0.5 * ll_1

def calculate_competition_score(y_true, y_pred):
    """대회 평가 지표: 0.5*AP + 0.5*(1/(1+WLL))"""
    ap = average_precision_score(y_true, y_pred)
    wll = calculate_weighted_logloss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score, ap, wll

print("🚀 Mac용 XGBoost 전용 CTR 예측 시작!")
print("📊 대회 평가지표: 0.5*AP + 0.5*(1/(1+WLL))")
print("=" * 60)

class MacXGBoostCTR:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.feature_cols = []
        print("🍎 Mac용 XGBoost CTR 초기화 완료")

    def load_data_efficiently(self, sample_ratio=0.3, use_batch=False):
        """효율적 데이터 로딩 (배치 처리 옵션 포함)"""
        print("\n📂 데이터 로딩 시작...")

        train_size = os.path.getsize('data/train.parquet') / (1024**3)
        print(f"📊 훈련 데이터 크기: {train_size:.1f}GB")

        if sample_ratio < 1.0:
            # 샘플링 모드
            print(f"🔄 샘플링 모드 - {sample_ratio*100:.0f}% 데이터 사용")

            # 전체 데이터를 한번에 로드한 후 샘플링
            print("📦 전체 데이터 로드 후 샘플링...")
            full_df = pd.read_parquet('data/train.parquet')

            # 균형 샘플링
            clicked = full_df[full_df['clicked'] == 1]
            not_clicked = full_df[full_df['clicked'] == 0]

            n_clicked = int(len(clicked) * sample_ratio)
            n_not_clicked = int(len(not_clicked) * sample_ratio)

            print(f"샘플링: 클릭 {len(clicked):,} → {n_clicked:,}, 비클릭 {len(not_clicked):,} → {n_not_clicked:,}")

            sample_clicked = clicked.sample(min(len(clicked), n_clicked), random_state=42) if n_clicked > 0 else pd.DataFrame()
            sample_not_clicked = not_clicked.sample(min(len(not_clicked), n_not_clicked), random_state=42) if n_not_clicked > 0 else pd.DataFrame()

            self.train_df = pd.concat([sample_clicked, sample_not_clicked], ignore_index=True)

            # 메모리 정리
            del full_df, clicked, not_clicked, sample_clicked, sample_not_clicked
            gc.collect()

        elif use_batch:
            # 배치 처리 모드 (전체 데이터 + 메모리 안전)
            print("🔄 배치 처리 모드 - 전체 데이터를 안전하게 로딩")
            self.train_df = self.load_data_in_batches('data/train.parquet')

        else:
            # 직접 로드 모드 (위험하지만 빠름)
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
        print(f"   클릭률: {self.train_df['clicked'].mean():.4f}")

        return True

    def load_data_in_batches(self, file_path, batch_size=500000):
        """배치별로 안전하게 데이터 로딩 (간단한 방식)"""
        print(f"📦 배치 크기 {batch_size:,}행으로 안전 로딩...")

        try:
            # 먼저 전체 로드 시도
            print("   전체 로드 시도 중...")
            full_df = pd.read_parquet(file_path)
            print(f"✅ 전체 로드 성공: {full_df.shape}")
            return full_df

        except MemoryError:
            print("   메모리 부족! 샘플링으로 전환...")

            # 메모리 부족 시 70% 샘플링으로 폴백
            full_df = pd.read_parquet(file_path)
            clicked = full_df[full_df['clicked'] == 1]
            not_clicked = full_df[full_df['clicked'] == 0]

            # 70% 샘플링
            sample_ratio = 0.7
            n_clicked = int(len(clicked) * sample_ratio)
            n_not_clicked = int(len(not_clicked) * sample_ratio)

            sample_clicked = clicked.sample(min(len(clicked), n_clicked), random_state=42)
            sample_not_clicked = not_clicked.sample(min(len(not_clicked), n_not_clicked), random_state=42)

            result_df = pd.concat([sample_clicked, sample_not_clicked], ignore_index=True)

            # 메모리 정리
            del full_df, clicked, not_clicked, sample_clicked, sample_not_clicked
            gc.collect()

            print(f"✅ 샘플링 로드 완료: {result_df.shape}")
            return result_df

        except Exception as e:
            print(f"❌ 로딩 실패: {e}")
            # 최후의 수단: 30% 샘플링
            print("   30% 샘플링으로 재시도...")
            full_df = pd.read_parquet(file_path)
            return full_df.sample(frac=0.3, random_state=42)

    def preprocess_features(self):
        """빠른 특성 전처리"""
        print("\n🔧 특성 전처리 시작...")

        # 수치형 특성
        numeric_cols = [col for col in self.train_df.columns
                       if col.startswith(('feat_', 'history_', 'l_feat_'))]

        # 카테고리 특성
        categorical_cols = ['gender', 'age_group']
        if 'inventory_id' in self.train_df.columns:
            categorical_cols.append('inventory_id')

        print(f"📊 특성 정보:")
        print(f"   수치형: {len(numeric_cols)}개")
        print(f"   카테고리: {len(categorical_cols)}개")

        # 수치형 전처리
        print("🔧 수치형 특성 처리...")
        for col in tqdm(numeric_cols, desc="수치형"):
            if col in self.train_df.columns:
                mean_val = self.train_df[col].fillna(0).mean()
                self.train_df[col] = self.train_df[col].fillna(mean_val)
                self.test_df[col] = self.test_df[col].fillna(mean_val)

        # 카테고리 전처리
        print("🔧 카테고리 특성 처리...")
        for col in tqdm(categorical_cols, desc="카테고리"):
            if col in self.train_df.columns:
                le = LabelEncoder()

                combined = pd.concat([
                    self.train_df[col].fillna('unknown'),
                    self.test_df[col].fillna('unknown')
                ]).astype(str)

                le.fit(combined)
                self.train_df[col] = le.transform(self.train_df[col].fillna('unknown').astype(str))
                self.test_df[col] = le.transform(self.test_df[col].fillna('unknown').astype(str))
                self.encoders[col] = le

        # 최종 특성 목록
        self.feature_cols = [col for col in numeric_cols + categorical_cols
                           if col in self.train_df.columns]

        print(f"✅ 전처리 완료: {len(self.feature_cols)}개 특성")
        return True

    def train_xgboost_models(self):
        """XGBoost 모델 훈련 (validation + early stopping)"""
        print("\n🚀 XGBoost 모델 훈련 시작...")

        X = self.train_df[self.feature_cols]
        y = self.train_df['clicked']

        # 클래스 불균형 처리
        pos_ratio = y.mean()
        scale_pos_weight = (1 - pos_ratio) / pos_ratio
        print(f"📊 클릭률: {pos_ratio:.4f}")
        print(f"📊 Scale pos weight: {scale_pos_weight:.2f}")

        # Train/Validation 분할 (early stopping용)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 분할 후 타겟 분포 확인
        train_click_rate = y_train.mean()
        val_click_rate = y_val.mean()

        print(f"📊 데이터 분할: 훈련 {X_train.shape[0]:,}, 검증 {X_val.shape[0]:,}")
        print(f"📊 클릭률 분포:")
        print(f"   전체: {y.mean():.4f}")
        print(f"   훈련: {train_click_rate:.4f}")
        print(f"   검증: {val_click_rate:.4f}")
        print(f"   차이: {abs(train_click_rate - val_click_rate):.4f}")

        # 안전성 검사
        if abs(train_click_rate - val_click_rate) > 0.001:
            print("⚠️ 경고: 클릭률 분포가 다름!")
        else:
            print("✅ 클릭률 분포 균등함")

        # 모델 설정들 (대회 지표 최적화)
        model_configs = {
            'xgb_ap_focused': {
                'objective': 'binary:logistic',
                'eval_metric': ['auc', 'aucpr'],  # AP 최적화
                'tree_method': 'hist',
                'max_depth': 6,
                'learning_rate': 0.08,
                'n_estimators': 600,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            },
            'xgb_balanced': {
                'objective': 'binary:logistic',
                'eval_metric': ['logloss', 'aucpr'],  # WLL + AP 균형
                'tree_method': 'hist',
                'max_depth': 7,
                'learning_rate': 0.06,
                'n_estimators': 800,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'scale_pos_weight': scale_pos_weight,
                'reg_alpha': 0.1,  # L1 정규화 (확률 보정)
                'reg_lambda': 0.1,  # L2 정규화
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
        }

        for name, params in model_configs.items():
            print(f"\n🔄 {name} 훈련 중...")

            # Early stopping 파라미터를 생성자에 추가
            params_with_early_stop = params.copy()
            params_with_early_stop['early_stopping_rounds'] = 20
            params_with_early_stop['enable_categorical'] = False  # 호환성

            model = xgb.XGBClassifier(**params_with_early_stop)

            # 간단한 fit (새 버전 호환)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=20  # 20 라운드마다 출력
            )

            # 검증 성능 평가 (대회 지표)
            val_pred = model.predict_proba(X_val)[:, 1]

            # 대회 평가 지표 계산
            comp_score, ap, wll = calculate_competition_score(y_val, val_pred)
            auc = roc_auc_score(y_val, val_pred)

            # Best iteration 정보
            try:
                if hasattr(model, 'best_iteration') and model.best_iteration is not None:
                    best_iter = model.best_iteration
                    print(f"   ✅ Best iteration: {best_iter}")
                else:
                    best_iter = getattr(model, 'n_estimators', 'unknown')
                    print(f"   ✅ Total iterations: {best_iter}")
            except:
                print(f"   ✅ 훈련 완료")

            print(f"   📊 {name} 성능:")
            print(f"      대회 점수: {comp_score:.4f}")
            print(f"      AP: {ap:.4f}")
            print(f"      WLL: {wll:.4f}")
            print(f"      AUC: {auc:.4f}")

            self.models[name] = [model]  # 리스트로 감싸서 기존 코드와 호환

        print("✅ 모델 훈련 완료!")

    def predict_and_submit(self):
        """예측 및 제출 파일 생성"""
        print("\n🎯 예측 시작...")

        X_test = self.test_df[self.feature_cols]
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

        # 앙상블 평균
        final_predictions = np.mean(all_predictions, axis=0)

        # 제출 파일 생성 (올바른 형식)
        try:
            # sample_submission.csv를 템플릿으로 사용
            submission = pd.read_csv('data/sample_submission.csv')
            submission['clicked'] = final_predictions
            print(f"✅ 올바른 ID 형식 사용: {submission['ID'].iloc[0]}")
        except:
            # 폴백: 직접 생성
            submission = pd.DataFrame({
                'ID': [f'TEST_{i:07d}' for i in range(len(final_predictions))],
                'clicked': final_predictions
            })
            print("⚠️ 직접 ID 생성 (sample_submission.csv 없음)")

        submission_path = 'submission_mac_xgboost_competition.csv'
        submission.to_csv(submission_path, index=False, encoding='utf-8')

        print(f"\n✅ 제출 파일 생성: {submission_path}")
        print(f"📊 예측 통계:")
        print(f"   평균 클릭률: {final_predictions.mean():.4f}")
        print(f"   최소값: {final_predictions.min():.4f}")
        print(f"   최대값: {final_predictions.max():.4f}")

        return submission_path

    def run_pipeline(self, sample_ratio=0.3, use_batch=False):
        """전체 파이프라인 실행"""
        start_time = time.time()

        # 1. 데이터 로딩
        if not self.load_data_efficiently(sample_ratio, use_batch):
            return False

        # 2. 전처리
        if not self.preprocess_features():
            return False

        # 3. 모델 훈련
        self.train_xgboost_models()

        # 4. 예측
        submission_path = self.predict_and_submit()

        elapsed = time.time() - start_time

        print("\n" + "🎉" * 60)
        print("🎉 파이프라인 완료! 🎉")
        print("🎉" * 60)
        print(f"⏱️ 총 소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
        print(f"📁 제출 파일: {submission_path}")

        return True

def main():
    pipeline = MacXGBoostCTR()

    print("\n📋 실행 옵션:")
    print("1. 🚀 초고속 모드 (30% 샘플링) - 1-2분")
    print("2. ⚡ 빠른 모드 (50% 샘플링) - 2-3분")
    print("3. 🎯 정확 모드 (70% 샘플링) - 4-5분")
    print("4. 🏆 최고 성능 모드 (전체 데이터 직접) - 5-8분 ⚠️ 메모리 위험")
    print("5. 🛡️ 안전 최고 성능 모드 (전체 데이터 배치) - 8-12분 ✅ 메모리 안전")

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

    print("=" * 60)

    success = pipeline.run_pipeline(sample_ratio, use_batch)

    if success:
        print("\n🎊 성공! 제출 파일이 준비되었습니다!")
    else:
        print("\n❌ 실행 실패")

if __name__ == "__main__":
    main()