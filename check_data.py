#!/usr/bin/env python3
"""
데이터 샘플 확인 스크립트
"""

import pandas as pd
import numpy as np

def check_data():
    print("=== 데이터 샘플 확인 ===")

    try:
        # 작은 샘플 데이터 로드
        df = pd.read_parquet('data/train.parquet')
        sample = df.head(1000)

        print(f"데이터 형태: {sample.shape}")
        print(f"컬럼 수: {len(sample.columns)}")
        print(f"전체 데이터 크기: {df.shape}")

        print("\n=== 컬럼 목록 ===")
        print(sample.columns.tolist())

        print("\n=== 범주형 변수 상세 확인 ===")
        categorical_cols = ['gender', 'age_group', 'inventory_id', 'l_feat_14']

        for col in categorical_cols:
            if col in sample.columns:
                unique_vals = sample[col].unique()
                print(f"\n{col}:")
                print(f"  - 유니크 값 개수: {len(unique_vals)}")
                print(f"  - 데이터 타입: {sample[col].dtype}")
                print(f"  - 결측값 개수: {sample[col].isnull().sum()}")
                print(f"  - 샘플 값들: {unique_vals[:10]}")

                # NaN 여부 확인
                if sample[col].isnull().any():
                    print(f"  - 결측값 있음!")
                    print(f"  - 결측값 인덱스: {sample[sample[col].isnull()].index.tolist()[:5]}")

        print("\n=== seq 컬럼 확인 ===")
        if 'seq' in sample.columns:
            print(f"seq 데이터 타입: {sample['seq'].dtype}")
            print(f"seq 샘플 값: {sample['seq'].iloc[0]}")
            print(f"seq 길이 (첫 번째): {len(str(sample['seq'].iloc[0]).split(',')) if pd.notna(sample['seq'].iloc[0]) else 'NaN'}")

        print("\n=== clicked 타겟 변수 확인 ===")
        if 'clicked' in sample.columns:
            print(f"clicked 값 분포: {sample['clicked'].value_counts()}")
            print(f"clicked 데이터 타입: {sample['clicked'].dtype}")

        print("\n=== 수치형 변수 예시 ===")
        numeric_cols = [col for col in sample.columns if col.startswith(('feat_', 'history_'))][:5]
        for col in numeric_cols:
            print(f"{col}: 타입={sample[col].dtype}, 샘플값={sample[col].iloc[0]}")

        # 메모리 정보
        print(f"\n=== 메모리 사용량 ===")
        print(f"샘플 메모리: {sample.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")

        return True

    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    check_data()