#!/usr/bin/env python3
"""
CTR 예측을 위한 데이터 전처리 실행 스크립트
EDA 결과를 바탕으로 최적화된 전처리 파이프라인을 실행합니다.
"""

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from ctr_preprocessing import CTRDataPreprocessor
from preprocessing_utils import CTRPreprocessingUtils

def main():
    print("CTR 예측 데이터 전처리 시작...")

    # 데이터 경로 설정
    train_path = 'data/train.parquet'
    test_path = 'data/test.parquet'

    # 파일 존재 확인
    if not os.path.exists(train_path):
        print(f"오류: {train_path} 파일을 찾을 수 없습니다.")
        return

    if not os.path.exists(test_path):
        print(f"경고: {test_path} 파일을 찾을 수 없습니다. 훈련 데이터만 처리합니다.")
        test_path = None

    # 전처리기 초기화
    preprocessor = CTRDataPreprocessor()
    utils = CTRPreprocessingUtils()

    # 1. 데이터 로드
    print("\n1. 데이터 로딩...")
    with tqdm(total=2, desc="데이터 로딩") as pbar:
        preprocessor.load_data(train_path, test_path)
        pbar.update(2)

    # 2. 기본 데이터 정보 출력
    print(f"\n훈련 데이터 기본 정보:")
    print(f"- 형태: {preprocessor.train_df.shape}")
    print(f"- 메모리 사용량: {preprocessor.train_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print(f"- CTR: {preprocessor.train_df['clicked'].mean():.4f}")

    # 3. 훈련 데이터 전처리
    print("\n2. 훈련 데이터 전처리...")
    with tqdm(total=6, desc="훈련 데이터 전처리") as pbar:
        train_processed = preprocessor.preprocess_pipeline(
            preprocessor.train_df,
            is_training=True,
            target_col='clicked',
            pbar=pbar
        )

    # 4. 테스트 데이터 전처리 (있는 경우)
    test_processed = None
    if test_path:
        print("\n3. 테스트 데이터 전처리...")
        with tqdm(total=6, desc="테스트 데이터 전처리") as pbar:
            test_processed = preprocessor.preprocess_pipeline(
                preprocessor.test_df,
                is_training=False,
                pbar=pbar
            )

    # 5. 고급 피처 엔지니어링
    print("\n4. 고급 피처 엔지니어링...")

    feature_engineering_steps = []
    if 'seq' in train_processed.columns:
        feature_engineering_steps.append("시퀀스 피처")

    numeric_cols = [col for col in train_processed.columns
                   if col.startswith(('feat_', 'history_')) and '_bin' not in col]
    if len(numeric_cols) > 1:
        feature_engineering_steps.append("통계적 피처")

    feature_engineering_steps.append("인터랙션 피처")

    with tqdm(total=len(feature_engineering_steps), desc="고급 피처 엔지니어링") as pbar:
        # 시퀀스 피처 처리
        if 'seq' in train_processed.columns:
            train_processed = utils.handle_sequence_features(train_processed)
            if test_processed is not None:
                test_processed = utils.handle_sequence_features(test_processed)
            pbar.set_postfix_str("시퀀스 피처 처리 완료")
            pbar.update(1)

        # 숫자형 피처의 통계적 특성 추가
        if len(numeric_cols) > 1:
            train_processed = utils.create_statistical_features(train_processed, numeric_cols)
            if test_processed is not None:
                test_processed = utils.create_statistical_features(test_processed, numeric_cols)
            pbar.set_postfix_str("통계적 피처 생성 완료")
            pbar.update(1)

        # 인터랙션 피처 생성 (주요 피처들만)
        important_features = ['gender', 'age_group', 'hour', 'day_of_week']
        feature_pairs = [(feat1, feat2) for i, feat1 in enumerate(important_features)
                        for feat2 in important_features[i+1:]]

        train_processed = utils.create_interaction_features(train_processed, feature_pairs, max_combinations=20)
        if test_processed is not None:
            test_processed = utils.create_interaction_features(test_processed, feature_pairs, max_combinations=20)
        pbar.set_postfix_str("인터랙션 피처 생성 완료")
        pbar.update(1)

    # 6. 훈련/검증 데이터 분할
    print("\n5. 훈련/검증 데이터 분할...")

    # 타겟 변수 분리
    feature_cols = [col for col in train_processed.columns if col != 'clicked']
    X = train_processed[feature_cols]
    y = train_processed['clicked']

    # 분할
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✓ 훈련 세트: {X_train.shape}")
    print(f"✓ 검증 세트: {X_val.shape}")
    if test_processed is not None:
        print(f"✓ 테스트 세트: {test_processed.shape}")

    # 7. 피처 중요도 분석
    print("\n6. 피처 중요도 분석...")
    try:
        with tqdm(total=3, desc="피처 중요도 분석") as pbar:
            # 샘플링으로 계산 속도 향상
            sample_size = min(50000, len(X_train))
            sample_idx = np.random.choice(len(X_train), sample_size, replace=False)
            pbar.set_postfix_str("데이터 샘플링")
            pbar.update(1)

            X_sample = X_train.iloc[sample_idx]
            y_sample = y_train.iloc[sample_idx]
            pbar.set_postfix_str("샘플 데이터 준비")
            pbar.update(1)

            feature_importance = utils.analyze_feature_importance(
                X_sample, y_sample,
                feature_names=X_sample.columns,
                method='mutual_info',
                k=30
            )
            pbar.set_postfix_str("중요도 계산 완료")
            pbar.update(1)

        print("상위 10개 중요 피처:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

    except Exception as e:
        print(f"피처 중요도 분석 중 오류: {e}")

    # 8. 데이터 드리프트 검사 (테스트 데이터가 있는 경우)
    if test_processed is not None:
        print("\n7. 데이터 드리프트 검사...")
        try:
            with tqdm(total=2, desc="데이터 드리프트 검사") as pbar:
                common_features = list(set(X_train.columns) & set(test_processed.columns))[:20]  # 상위 20개만
                pbar.set_postfix_str("공통 피처 선별")
                pbar.update(1)

                drift_results = utils.detect_data_drift(
                    X_train[common_features],
                    test_processed[common_features],
                    common_features
                )
                pbar.set_postfix_str("드리프트 검사 완료")
                pbar.update(1)

            drift_features = drift_results[drift_results['has_drift']].shape[0]
            print(f"✓ 드리프트 감지된 피처 수: {drift_features}/{len(common_features)}")

            if drift_features > 0:
                print("드리프트가 감지된 상위 5개 피처:")
                for i, row in drift_results[drift_results['has_drift']].head(5).iterrows():
                    print(f"  {row['feature']}: KS통계량={row['ks_statistic']:.4f}")

        except Exception as e:
            print(f"데이터 드리프트 검사 중 오류: {e}")

    # 9. 결과 저장
    print("\n8. 전처리된 데이터 저장...")

    # 출력 디렉토리 생성
    os.makedirs('processed_data', exist_ok=True)

    # 저장할 파일 목록 준비
    save_files = [
        ("X_train", X_train),
        ("X_val", X_val),
        ("y_train", y_train),
        ("y_val", y_val)
    ]

    if test_processed is not None:
        test_features = test_processed[feature_cols] if 'clicked' not in test_processed.columns else test_processed.drop('clicked', axis=1)
        save_files.append(("X_test", test_features))

    # 데이터 저장
    with tqdm(total=len(save_files), desc="데이터 저장") as pbar:
        for name, data in save_files:
            data.to_parquet(f'processed_data/{name}.parquet')
            pbar.set_postfix_str(f"{name}.parquet 저장 완료")
            pbar.update(1)

    print("✓ 전처리된 데이터가 'processed_data' 디렉토리에 저장되었습니다.")

    # 10. 전처리 리포트 생성
    print("\n9. 전처리 리포트 생성...")
    utils.generate_preprocessing_report(preprocessor.train_df, train_processed)

    # 피처 리스트 저장
    feature_info = {
        'total_features': len(feature_cols),
        'feature_names': feature_cols,
        'categorical_features': [col for col in feature_cols if col in ['gender', 'age_group', 'inventory_id', 'l_feat_14']],
        'numeric_features': [col for col in feature_cols if col.startswith(('feat_', 'history_'))],
        'engineered_features': [col for col in feature_cols if any(suffix in col for suffix in ['_sin', '_cos', '_log1p', '_sqrt', '_bin', '_enc', '_mult', '_add', '_ratio'])]
    }

    import json
    with open('processed_data/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)

    print(f"✓ 피처 정보가 'processed_data/feature_info.json'에 저장되었습니다.")
    print(f"✓ 총 {len(feature_cols)}개의 피처가 생성되었습니다.")

    print("\n" + "="*60)
    print("전처리 완료!")
    print("다음 파일들이 생성되었습니다:")
    print("- processed_data/X_train.parquet (훈련 피처)")
    print("- processed_data/X_val.parquet (검증 피처)")
    print("- processed_data/y_train.parquet (훈련 타겟)")
    print("- processed_data/y_val.parquet (검증 타겟)")
    if test_processed is not None:
        print("- processed_data/X_test.parquet (테스트 피처)")
    print("- processed_data/feature_info.json (피처 정보)")
    print("="*60)

    return {
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'X_test': test_processed[feature_cols] if test_processed is not None else None,
        'preprocessor': preprocessor,
        'feature_info': feature_info
    }

if __name__ == "__main__":
    results = main()