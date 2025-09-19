#!/usr/bin/env python3
"""
'unknown' 라벨이 생성되는 원인 추적 스크립트
"""

import pandas as pd
import numpy as np

def debug_unknown_issue():
    print("=== 'unknown' 라벨 발생 원인 추적 ===")

    try:
        # 작은 샘플로 먼저 테스트
        print("작은 샘플 로딩 중...")
        df = pd.read_parquet('data/train.parquet')

        # 첫 10만행과 두 번째 10만행 비교
        chunk1 = df.iloc[:100000].copy()
        chunk2 = df.iloc[100000:200000].copy()

        print(f"청크1 크기: {chunk1.shape}")
        print(f"청크2 크기: {chunk2.shape}")

        # 범주형 변수들을 확인
        categorical_cols = ['gender', 'age_group', 'inventory_id', 'l_feat_14']

        print("\n=== 각 청크별 범주형 변수 비교 ===")
        for col in categorical_cols:
            if col in df.columns:
                print(f"\n--- {col} ---")

                # 청크1 고유값
                chunk1_unique = set(chunk1[col].dropna().astype(str))
                chunk2_unique = set(chunk2[col].dropna().astype(str))

                print(f"청크1 고유값 개수: {len(chunk1_unique)}")
                print(f"청크2 고유값 개수: {len(chunk2_unique)}")

                # 차이점 확인
                only_in_chunk1 = chunk1_unique - chunk2_unique
                only_in_chunk2 = chunk2_unique - chunk1_unique

                if only_in_chunk1:
                    print(f"청크1에만 있는 값: {list(only_in_chunk1)[:10]}")
                if only_in_chunk2:
                    print(f"청크2에만 있는 값: {list(only_in_chunk2)[:10]}")

                # 결측값 확인
                chunk1_nulls = chunk1[col].isnull().sum()
                chunk2_nulls = chunk2[col].isnull().sum()
                print(f"청크1 결측값: {chunk1_nulls}")
                print(f"청크2 결측값: {chunk2_nulls}")

        print("\n=== 전처리 중 'unknown' 생성 과정 추적 ===")

        # CTR 전처리 과정 시뮬레이션
        from ctr_preprocessing import CTRDataPreprocessor

        preprocessor = CTRDataPreprocessor()

        # 1단계: 첫 번째 청크로 전처리기 학습
        print("\n1. 첫 번째 청크로 전처리기 학습...")
        chunk1_processed = preprocessor.preprocess_pipeline(
            chunk1,
            is_training=True,
            target_col='clicked'
        )

        print("전처리 완료!")
        print(f"처리된 청크1 크기: {chunk1_processed.shape}")

        # 2단계: 두 번째 청크 처리 (여기서 오류 발생할 것으로 예상)
        print("\n2. 두 번째 청크 처리 시도...")
        try:
            chunk2_processed = preprocessor.preprocess_pipeline(
                chunk2,
                is_training=False,
                target_col='clicked'
            )
            print("두 번째 청크도 성공!")
        except Exception as e:
            print(f"예상된 오류 발생: {e}")

            # 오류가 발생한 지점에서 상세 분석
            print("\n=== 상세 분석 ===")

            # 범주형 변수에서 어떤 값이 'unknown'으로 변경되었는지 확인
            for col in categorical_cols:
                if col in chunk2.columns:
                    print(f"\n{col} 분석:")

                    # 전처리 전 값들
                    original_values = set(chunk2[col].dropna().astype(str))

                    # 학습된 인코더의 클래스들
                    if col in preprocessor.encoders:
                        encoder_classes = set(preprocessor.encoders[col].classes_)

                        # 인코더에 없는 값들 (unknown이 될 값들)
                        unknown_values = original_values - encoder_classes

                        print(f"  원본 고유값 개수: {len(original_values)}")
                        print(f"  인코더 클래스 개수: {len(encoder_classes)}")
                        print(f"  unknown이 될 값들: {list(unknown_values)[:10]}")

                        if unknown_values:
                            print(f"  🚨 {col}에서 {len(unknown_values)}개 값이 unknown으로 변환됨!")

    except Exception as e:
        print(f"전체 오류: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    debug_unknown_issue()