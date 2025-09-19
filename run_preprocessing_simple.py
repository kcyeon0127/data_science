#!/usr/bin/env python3
"""
간단한 CTR 예측 데이터 전처리 실행 스크립트
메모리 문제를 최소화한 버전
"""

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import gc
from ctr_preprocessing import CTRDataPreprocessor
from preprocessing_utils import CTRPreprocessingUtils

def load_data_safely(file_path, max_rows=None):
    """안전하게 데이터 로드"""
    print(f"데이터 로드 중: {file_path}")

    try:
        # 파일 크기 확인
        file_size_gb = os.path.getsize(file_path) / (1024**3)
        print(f"파일 크기: {file_size_gb:.2f} GB")

        # 작은 샘플로 구조 확인
        sample = pd.read_parquet(file_path, nrows=1000) if max_rows else pd.read_parquet(file_path)

        if max_rows and len(sample) > max_rows:
            print(f"데이터를 {max_rows:,}행으로 제한합니다.")
            sample = sample.head(max_rows)

        return sample

    except Exception as e:
        print(f"데이터 로드 오류: {e}")
        return None

def main():
    print("간단한 CTR 예측 데이터 전처리 시작...")

    # 메모리 체크
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    print(f"사용 가능 메모리: {available_memory_gb:.1f}GB")

    # 메모리에 따른 처리 방식 결정
    if available_memory_gb < 4:
        max_rows = 100000
        print(f"⚠️  메모리 부족: 샘플 데이터로 처리합니다 ({max_rows:,}행)")
    elif available_memory_gb < 8:
        max_rows = 500000
        print(f"🔧 메모리 제한적: 일부 데이터로 처리합니다 ({max_rows:,}행)")
    else:
        max_rows = None
        print("🚀 전체 데이터 처리를 시도합니다.")

    # 데이터 경로 설정
    train_path = 'data/train.parquet'
    test_path = 'data/test.parquet'

    # 파일 존재 확인
    if not os.path.exists(train_path):
        print(f"오류: {train_path} 파일을 찾을 수 없습니다.")
        return False

    # 전처리기 초기화
    preprocessor = CTRDataPreprocessor()
    utils = CTRPreprocessingUtils()

    # 출력 디렉토리 생성
    os.makedirs('processed_data', exist_ok=True)

    try:
        print("\n=== 1단계: 훈련 데이터 로드 ===")
        train_df = load_data_safely(train_path, max_rows)
        if train_df is None:
            return False

        print(f"훈련 데이터 형태: {train_df.shape}")
        print(f"CTR: {train_df['clicked'].mean():.4f}")
        print(f"메모리 사용량: {train_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")

        print("\n=== 2단계: 데이터 전처리 ===")
        with tqdm(total=6, desc="훈련 데이터 전처리") as pbar:
            train_processed = preprocessor.preprocess_pipeline(
                train_df,
                is_training=True,
                target_col='clicked',
                pbar=pbar
            )

        print(f"전처리 완료: {train_processed.shape}")

        # 원본 데이터 메모리 해제
        del train_df
        gc.collect()

        print("\n=== 3단계: 테스트 데이터 처리 ===")
        test_processed = None
        if os.path.exists(test_path):
            test_df = load_data_safely(test_path, max_rows)
            if test_df is not None:
                print(f"테스트 데이터 형태: {test_df.shape}")

                with tqdm(total=6, desc="테스트 데이터 전처리") as pbar:
                    test_processed = preprocessor.preprocess_pipeline(
                        test_df,
                        is_training=False,
                        pbar=pbar
                    )

                print(f"테스트 전처리 완료: {test_processed.shape}")
                del test_df
                gc.collect()

        print("\n=== 4단계: 훈련/검증 분할 ===")
        # 타겟 변수 분리
        feature_cols = [col for col in train_processed.columns if col != 'clicked']
        X = train_processed[feature_cols]
        y = train_processed['clicked']

        print(f"피처 수: {len(feature_cols)}")

        # 메모리 절약을 위한 분할
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"훈련 세트: {X_train.shape}")
        print(f"검증 세트: {X_val.shape}")

        # 메모리 정리
        del train_processed, X, y
        gc.collect()

        print("\n=== 5단계: 간단한 분석 ===")
        # 메모리를 절약하기 위해 작은 샘플로만 분석
        if len(X_train) > 10000:
            sample_X = X_train.sample(n=10000, random_state=42)
            sample_y = y_train.loc[sample_X.index]

            try:
                print("피처 중요도 분석 (샘플)...")
                feature_importance = utils.analyze_feature_importance(
                    sample_X, sample_y,
                    feature_names=sample_X.columns,
                    method='mutual_info',
                    k=min(20, len(sample_X.columns))
                )

                print("상위 5개 중요 피처:")
                for i, row in feature_importance.head(5).iterrows():
                    print(f"  {row['feature']}: {row['importance']:.4f}")

            except Exception as e:
                print(f"피처 중요도 분석 중 오류: {e}")

        print("\n=== 6단계: 데이터 저장 ===")
        save_files = [
            ("X_train", X_train),
            ("X_val", X_val),
            ("y_train", y_train),
            ("y_val", y_val)
        ]

        if test_processed is not None:
            # 테스트 데이터에서 타겟 컬럼 제거
            test_features = test_processed[feature_cols] if 'clicked' not in test_processed.columns else test_processed.drop('clicked', axis=1)
            save_files.append(("X_test", test_features))

        with tqdm(total=len(save_files), desc="데이터 저장") as pbar:
            for name, data in save_files:
                try:
                    # 데이터 타입 최적화
                    optimized_data = data.copy()

                    # 정수형 최적화
                    int_cols = optimized_data.select_dtypes(include=['int64']).columns
                    for col in int_cols:
                        optimized_data[col] = pd.to_numeric(optimized_data[col], downcast='integer')

                    # 실수형 최적화
                    float_cols = optimized_data.select_dtypes(include=['float64']).columns
                    for col in float_cols:
                        optimized_data[col] = pd.to_numeric(optimized_data[col], downcast='float')

                    # 저장
                    optimized_data.to_parquet(f'processed_data/{name}.parquet', compression='snappy')

                    memory_saved = (data.memory_usage(deep=True).sum() - optimized_data.memory_usage(deep=True).sum()) / 1024 / 1024
                    pbar.set_postfix_str(f"{name}: {optimized_data.shape}, {memory_saved:.1f}MB 절약")
                    pbar.update(1)

                    del optimized_data
                    gc.collect()

                except Exception as e:
                    print(f"{name} 저장 중 오류: {e}")
                    # 최적화 없이 저장 시도
                    try:
                        data.to_parquet(f'processed_data/{name}.parquet')
                        pbar.set_postfix_str(f"{name}: {data.shape} (최적화 실패)")
                        pbar.update(1)
                    except Exception as e2:
                        print(f"{name} 저장 완전 실패: {e2}")

        print("\n=== 7단계: 메타데이터 저장 ===")
        # 피처 정보 저장
        feature_info = {
            'total_features': len(feature_cols),
            'feature_names': feature_cols[:50],  # 처음 50개만 저장 (메모리 절약)
            'data_shape': {
                'train': list(X_train.shape),
                'val': list(X_val.shape),
                'test': list(test_processed.shape) if test_processed is not None else None
            },
            'memory_optimized': True,
            'sample_processed': max_rows is not None
        }

        import json
        with open('processed_data/feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)

        print("\n" + "="*60)
        print("✅ 간단한 전처리 완료!")
        print("생성된 파일들:")
        for name, _ in save_files:
            file_path = f'processed_data/{name}.parquet'
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / 1024 / 1024
                print(f"- {file_path} ({size_mb:.1f}MB)")
        print("- processed_data/feature_info.json")

        if max_rows:
            print(f"\n⚠️  메모리 제한으로 {max_rows:,}행만 처리되었습니다.")
            print("전체 데이터 처리를 위해서는 더 많은 메모리가 필요합니다.")

        print("="*60)
        return True

    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    import psutil
    process = psutil.Process()

    print(f"시작 메모리 사용량: {process.memory_info().rss / 1024 / 1024:.1f}MB")

    success = main()

    print(f"최종 메모리 사용량: {process.memory_info().rss / 1024 / 1024:.1f}MB")

    if success:
        print("처리가 성공적으로 완료되었습니다!")
    else:
        print("처리에 실패했습니다. 로그를 확인해주세요.")