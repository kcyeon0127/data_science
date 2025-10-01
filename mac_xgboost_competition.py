#!/usr/bin/env python3
"""
Mac용 XGBoost 전용 CTR 예측 
초고속 실행 + 실시간 진행 상황 표시
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
try:
    import pyarrow.parquet as pq
except ImportError:  # pyarrow is optional for batch loading
    pq = None
import xgboost as xgb
from tqdm.auto import tqdm
import gc
import time

print("🚀 Mac용 XGBoost 전용 CTR 예측 시작!")
print("=" * 60)

class MacXGBoostCTR:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.feature_cols = []
        self.additional_numeric_cols = []
        print("🍎 Mac용 XGBoost CTR 초기화 완료")

    @staticmethod
    def compute_weighted_logloss(y_true, y_pred, eps=1e-15):
        """50:50 class-balanced logloss for CTR evaluation."""
        y_pred = np.clip(y_pred, eps, 1 - eps)
        y_true = np.asarray(y_true)

        pos_mask = y_true == 1
        neg_mask = ~pos_mask

        pos_count = pos_mask.sum()
        neg_count = neg_mask.sum()

        weights = np.zeros_like(y_pred, dtype=float)
        if pos_count > 0:
            weights[pos_mask] = 0.5 / pos_count
        if neg_count > 0:
            weights[neg_mask] = 0.5 / neg_count

        weighted_loss = -weights * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        weight_sum = weights.sum()
        if weight_sum == 0:
            return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return weighted_loss.sum() / weight_sum

    @staticmethod
    def blended_eval_metric(y_true, y_pred):
        """Custom metric: 0.5*AP + 0.5*(1-WLL) for sklearn API."""
        ap = average_precision_score(y_true, y_pred)
        wll = MacXGBoostCTR.compute_weighted_logloss(y_true, y_pred)
        return 0.5 * ap + 0.5 * (1 - wll)

    # Tell XGBoost to log a friendly metric name when using the sklearn wrapper.
    blended_eval_metric.__name__ = 'ap50_wll50'

    @staticmethod
    def blended_eval_metric_to_minimize(y_true, y_pred):
        """Inverted blended metric so XGBoost can minimize it during training."""
        return 1.0 - MacXGBoostCTR.blended_eval_metric(y_true, y_pred)

    blended_eval_metric_to_minimize.__name__ = 'ap50_wll50_inv'

    @staticmethod
    def optimize_chunk_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Downcast numeric columns to reduce memory footprint during batching."""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        return df

    @staticmethod
    def xgb_blended_feval(preds, dmatrix):
        """Custom evaluation for xgboost.train using blended AP/WLL."""
        labels = dmatrix.get_label()
        proba = preds
        if np.any((proba < 0) | (proba > 1)):
            proba = 1.0 / (1.0 + np.exp(-proba))
        score = MacXGBoostCTR.blended_eval_metric(labels, proba)
        return 'ap50_wll50', score

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

    def add_target_encoding_features(self, columns=None, prior=50):
        """Add smoothed target-encoding and frequency features for categorical columns."""
        if columns is None:
            columns = ['inventory_id', 'gender', 'age_group']

        available_cols = [col for col in columns if col in self.train_df.columns]
        if not available_cols:
            return

        print("\n🎯 타깃 인코딩 기반 파생 특성 생성...")
        global_mean = self.train_df['clicked'].mean()

        for col in tqdm(available_cols, desc="타깃 인코딩"):
            tqdm.write(f"   ➕ {col} 처리 중...")
            te_col = f"{col}_ctr_te"
            count_col = f"{col}_count"

            stats = self.train_df.groupby(col)['clicked'].agg(['sum', 'count'])
            ctr_map_full = (stats['sum'] + global_mean * prior) / (stats['count'] + prior)
            count_map_full = stats['count']

            sum_map = self.train_df[col].map(stats['sum'])
            count_map = self.train_df[col].map(stats['count'])
            numerator = sum_map - self.train_df['clicked'] + global_mean * prior
            denominator = count_map - 1 + prior
            self.train_df[te_col] = (numerator / denominator).fillna(global_mean)
            self.train_df[count_col] = count_map.fillna(0)

            if col in self.test_df.columns:
                self.test_df[te_col] = self.test_df[col].map(ctr_map_full).fillna(global_mean)
                self.test_df[count_col] = self.test_df[col].map(count_map_full).fillna(0)
            else:
                self.test_df[te_col] = global_mean
                self.test_df[count_col] = 0

            self.train_df[te_col] = self.train_df[te_col].astype(np.float32, copy=False)
            self.test_df[te_col] = self.test_df[te_col].astype(np.float32, copy=False)
            self.train_df[count_col] = self.train_df[count_col].astype(np.float32, copy=False)
            self.test_df[count_col] = self.test_df[count_col].astype(np.float32, copy=False)

            for new_col in (te_col, count_col):
                if new_col not in self.additional_numeric_cols:
                    self.additional_numeric_cols.append(new_col)

        gc.collect()

    def load_data_in_batches(self, file_path, batch_size=500000):
        """배치별로 안전하게 데이터 로딩 (pyarrow 기반)"""
        print(f"📦 배치 크기 {batch_size:,}행으로 안전 로딩...")

        if pq is None:
            print("⚠️ pyarrow가 없어 일반 로딩으로 전환합니다 (메모리 주의)")
            return pd.read_parquet(file_path)

        try:
            parquet_file = pq.ParquetFile(file_path)
            batches = []
            total_rows = 0

            for idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size), start=1):
                chunk_df = batch.to_pandas()
                chunk_df = self.optimize_chunk_memory(chunk_df)
                batches.append(chunk_df)
                total_rows += len(chunk_df)

                if idx % 5 == 0:
                    print(f"   ✅ {total_rows:,}행 누적 로딩 완료")

            if not batches:
                print("⚠️ 배치 로드 결과가 비어 있습니다")
                return pd.DataFrame()

            result_df = pd.concat(batches, ignore_index=True)
            print(f"✅ 배치 로드 완료: {result_df.shape}")
            return result_df

        except MemoryError:
            print("❌ 여전히 메모리 부족! 70% 샘플링으로 폴백합니다")
            full_df = pd.read_parquet(file_path)
            return full_df.sample(frac=0.7, random_state=42)

        except Exception as e:
            print(f"❌ 배치 로드 실패: {e}")
            print("   30% 샘플링으로 재시도...")
            full_df = pd.read_parquet(file_path)
            return full_df.sample(frac=0.3, random_state=42)

    def preprocess_features(self):
        """빠른 특성 전처리"""
        print("\n🔧 특성 전처리 시작...")

        # 수치형 특성
        numeric_cols = [col for col in self.train_df.columns
                       if col.startswith(('feat_', 'history_', 'l_feat_'))]
        extra_numeric = [col for col in self.additional_numeric_cols
                         if col in self.train_df.columns]
        # 순서를 보존하며 중복 제거
        numeric_cols = list(dict.fromkeys(numeric_cols + extra_numeric))

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

        # 메모리 최적화를 위해 다운캐스팅
        if numeric_cols:
            self.train_df.loc[:, numeric_cols] = self.train_df[numeric_cols].astype(np.float32, copy=False)
            self.test_df.loc[:, numeric_cols] = self.test_df[numeric_cols].astype(np.float32, copy=False)
        if categorical_cols:
            self.train_df.loc[:, categorical_cols] = self.train_df[categorical_cols].astype(np.int32, copy=False)
            self.test_df.loc[:, categorical_cols] = self.test_df[categorical_cols].astype(np.int32, copy=False)

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

        # 원본 데이터프레임은 더 이상 필요 없으므로 해제
        del X, y

        # numpy로 캐스팅하여 DMatrix 생성 후 메모리 절약
        X_train_np = X_train.to_numpy(dtype=np.float32, copy=True)
        X_val_np = X_val.to_numpy(dtype=np.float32, copy=True)
        y_train_np = y_train.to_numpy(dtype=np.float32, copy=True)
        y_val_np = y_val.to_numpy(dtype=np.float32, copy=True)

        QuantileDMatrix = getattr(xgb, 'QuantileDMatrix', None)
        if QuantileDMatrix is not None:
            try:
                dtrain_full = QuantileDMatrix(X_train_np, label=y_train_np)
                try:
                    dval_full = QuantileDMatrix(X_val_np, label=y_val_np, reference=dtrain_full)
                except TypeError:
                    try:
                        dval_full = QuantileDMatrix(X_val_np, label=y_val_np, ref=dtrain_full)
                    except TypeError:
                        dval_full = QuantileDMatrix(X_val_np, label=y_val_np)
            except TypeError:
                QuantileDMatrix = None
        if QuantileDMatrix is None:
            dtrain_full = xgb.DMatrix(X_train_np, label=y_train_np)
            dval_full = xgb.DMatrix(X_val_np, label=y_val_np)

        # 원본 DataFrame/Series 메모리 해제
        del X_train, X_val, y_train, y_train_np, X_train_np, X_val_np
        gc.collect()

        # 모델 설정들 (early stopping 포함)
        model_configs = {
            'xgb_fast': {
                'objective': 'binary:logistic',
                'tree_method': 'hist',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 500,  # 충분히 크게 설정 (early stopping으로 조절)
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            },
            'xgb_deep': {
                'objective': 'binary:logistic',
                'tree_method': 'hist',
                'max_depth': 8,
                'learning_rate': 0.05,
                'n_estimators': 800,  # 충분히 크게 설정
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'scale_pos_weight': scale_pos_weight,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
        }

        for name, params in model_configs.items():
            print(f"\n🔄 {name} 훈련 중...")

            train_params = params.copy()
            num_boost_round = train_params.pop('n_estimators', 500)
            train_params['enable_categorical'] = False
            if 'n_jobs' in train_params:
                train_params['nthread'] = train_params.pop('n_jobs')
            train_params.setdefault('subsample', 0.8)
            train_params.setdefault('colsample_bytree', 0.8)
            train_params.setdefault('max_bin', 256)

            booster = xgb.train(
                params=train_params,
                dtrain=dtrain_full,
                num_boost_round=num_boost_round,
                evals=[(dtrain_full, 'train'), (dval_full, 'validation')],
                custom_metric=MacXGBoostCTR.xgb_blended_feval,
                maximize=True,
                early_stopping_rounds=20,
                verbose_eval=20
            )

            best_iter = getattr(booster, 'best_iteration', None)
            if best_iter is not None and best_iter >= 0:
                booster_for_eval = booster[: best_iter + 1]
            else:
                booster_for_eval = booster

            val_pred = booster_for_eval.predict(dval_full)
            ap = average_precision_score(y_val_np, val_pred)
            wll = self.compute_weighted_logloss(y_val_np, val_pred)
            blended_score = 0.5 * ap + 0.5 * (1 - wll)

            if best_iter is not None:
                print(f"   ✅ Best iteration: {best_iter}")
            else:
                print(f"   ✅ 훈련 완료 (early stopping 미사용)")

            print(f"   📊 {name} Validation AP: {ap:.4f}")
            print(f"   📊 {name} Validation WLL: {wll:.4f}")
            print(f"   📊 {name} Validation Blended (50% AP, 50% WLL): {blended_score:.4f}")

            self.models[name] = [{'booster': booster_for_eval}]

        # 학습에 사용된 중간 자원 정리
        del dtrain_full, dval_full, y_val_np
        gc.collect()

        print("✅ 모델 훈련 완료!")

    def predict_and_submit(self):
        """예측 및 제출 파일 생성"""
        print("\n🎯 예측 시작...")

        X_test = self.test_df[self.feature_cols]
        dtest = xgb.DMatrix(X_test)
        all_predictions = []

        for name, fold_models in self.models.items():
            print(f"🔄 {name} 예측 중...")

            fold_preds = []
            for i, model in enumerate(fold_models):
                booster = model['booster']
                pred = booster.predict(dtest)
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

        # 1-1. 타깃 인코딩 기반 파생 특성 추가
        self.add_target_encoding_features()

        # 2. 전처리
        if not self.preprocess_features():
            return False

        # 3. 모델 훈련
        self.train_xgboost_models()
        self.train_df = None
        gc.collect()

        # 4. 예측
        submission_path = self.predict_and_submit()

        elapsed = time.time() - start_time

        print("\n" + "🎉" * 10)
        print("🎉 파이프라인 완료! 🎉")
        print("🎉" * 10)
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
