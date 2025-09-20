#!/usr/bin/env python3
"""
테스트용 작은 CSV 파일 생성
"""

import pandas as pd

def create_test_csv():
    """작은 테스트 CSV 파일 생성"""

    try:
        # 원본에서 10만 행만 가져와서 테스트 CSV 생성
        df = pd.read_parquet('data/train.parquet')
        test_df = df.head(100000).copy()

        test_csv_path = 'data/train_test.csv'
        test_df.to_csv(test_csv_path, index=False)

        print(f"✅ 테스트 CSV 생성: {test_csv_path}")
        print(f"📊 크기: {test_df.shape}")

        return test_csv_path

    except Exception as e:
        print(f"❌ 테스트 CSV 생성 실패: {e}")
        return None

if __name__ == "__main__":
    test_path = create_test_csv()
    if test_path:
        print(f"\n🧪 테스트 명령어:")
        print(f"python run_preprocessing.py --data-path {test_path} --chunk-size 25000")