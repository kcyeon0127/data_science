#!/usr/bin/env python3
"""
청크별 Parquet → CSV 변환 (메모리 안전)
"""

import pandas as pd
import os

def convert_parquet_to_csv_chunked():
    """Parquet을 청크별로 CSV로 변환"""

    parquet_path = 'data/train.parquet'
    csv_path = 'data/train.csv'

    if not os.path.exists(parquet_path):
        print(f"❌ {parquet_path} 파일이 없습니다.")
        return False

    print("🔄 청크별 Parquet → CSV 변환 시작...")

    # 먼저 총 행 수 확인
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(parquet_path)
        total_rows = parquet_file.metadata.num_rows
        print(f"📊 총 데이터 행 수: {total_rows:,}행")
    except:
        print("❌ PyArrow로 메타데이터 읽기 실패")
        return False

    chunk_size = 500000  # 50만 행씩 처리
    total_chunks = (total_rows // chunk_size) + 1

    print(f"📦 {total_chunks}개 청크로 분할 처리 (청크당 {chunk_size:,}행)")

    # CSV 파일 초기화 (헤더 먼저 쓰기)
    header_written = False

    try:
        for chunk_idx in range(total_chunks):
            start_row = chunk_idx * chunk_size
            end_row = min(start_row + chunk_size, total_rows)

            if start_row >= total_rows:
                break

            print(f"🔄 청크 {chunk_idx+1}/{total_chunks}: {start_row:,} ~ {end_row:,}")

            # 전체 파일 로드 후 청크 추출
            full_df = pd.read_parquet(parquet_path)
            chunk_df = full_df.iloc[start_row:end_row].copy()
            del full_df

            # CSV 쓰기
            if not header_written:
                chunk_df.to_csv(csv_path, mode='w', index=False, header=True)
                header_written = True
            else:
                chunk_df.to_csv(csv_path, mode='a', index=False, header=False)

            del chunk_df
            import gc
            gc.collect()

            progress = ((chunk_idx + 1) / total_chunks) * 100
            print(f"✅ 청크 {chunk_idx+1} 완료 ({progress:.1f}%)")

        print(f"✅ 변환 완료: {csv_path}")

        # 파일 크기 확인
        parquet_size = os.path.getsize(parquet_path) / 1024 / 1024  # MB
        csv_size = os.path.getsize(csv_path) / 1024 / 1024  # MB

        print(f"📁 파일 크기:")
        print(f"  Parquet: {parquet_size:.1f}MB")
        print(f"  CSV: {csv_size:.1f}MB ({csv_size/parquet_size:.1f}배)")

        return True

    except Exception as e:
        print(f"❌ 변환 실패: {e}")
        return False

if __name__ == "__main__":
    success = convert_parquet_to_csv_chunked()
    if success:
        print("\n🎉 CSV 변환 완료!")
        print("이제 다음 명령어로 실행하세요:")
        print("python run_preprocessing.py --data-path data/train.csv --resume --chunk-size 100000")
    else:
        print("❌ 변환 실패")