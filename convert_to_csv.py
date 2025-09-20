#!/usr/bin/env python3
"""
Parquet을 CSV로 변환 - 스트리밍 처리 가능하게 만들기
"""

import pandas as pd
import os

def convert_parquet_to_csv():
    """Parquet 파일을 CSV로 변환"""

    parquet_path = 'data/train.parquet'
    csv_path = 'data/train.csv'

    if not os.path.exists(parquet_path):
        print(f"❌ {parquet_path} 파일이 없습니다.")
        return False

    print("🔄 Parquet → CSV 변환 시작...")
    print("⚠️ 이 과정은 시간이 걸릴 수 있습니다 (5-10분)")

    try:
        # 전체 파일 로드 (마지막으로 한 번만)
        print("📂 Parquet 파일 로딩 중...")
        df = pd.read_parquet(parquet_path)
        print(f"📊 데이터 크기: {df.shape}")

        # CSV로 저장 (진행률 표시)
        print("💾 CSV 저장 중...")
        df.to_csv(csv_path, index=False)
        print(f"✅ 변환 완료: {csv_path}")

        # 메모리 정리
        del df
        import gc
        gc.collect()

        # 파일 크기 비교
        parquet_size = os.path.getsize(parquet_path) / 1024 / 1024  # MB
        csv_size = os.path.getsize(csv_path) / 1024 / 1024  # MB

        print(f"📁 파일 크기 비교:")
        print(f"  Parquet: {parquet_size:.1f}MB")
        print(f"  CSV: {csv_size:.1f}MB ({csv_size/parquet_size:.1f}배)")

        return True

    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        return False

if __name__ == "__main__":
    success = convert_parquet_to_csv()
    if success:
        print("\n🎉 이제 CSV 기반 처리가 가능합니다!")
        print("다음 명령어로 실행하세요:")
        print("python run_preprocessing.py --data-path data/train.csv --chunk-size 100000")
    else:
        print("❌ 변환 실패")