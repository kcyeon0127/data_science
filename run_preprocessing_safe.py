#!/usr/bin/env python3
"""
메모리 안전한 CTR 예측 데이터 전처리 실행 스크립트
실제 청크 단위로 안전하게 처리
"""

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import gc
from ctr_preprocessing import CTRDataPreprocessor
from preprocessing_utils import CTRPreprocessingUtils

def process_single_chunk_file(file_path, start_row, chunk_size, preprocessor, is_training, target_col):
    """단일 청크를 파일에서 읽어서 처리"""
    try:
        # pandas의 skiprows와 nrows를 사용해 청크만 읽기
        chunk = pd.read_parquet(file_path)

        # 실제 청크 크기 계산
        total_rows = len(chunk)
        end_row = min(start_row + chunk_size, total_rows)

        if start_row >= total_rows:
            return None, True  # 완료 신호

        # 청크 분할
        chunk_data = chunk.iloc[start_row:end_row].copy()
        del chunk
        gc.collect()

        # 전처리 적용
        if preprocessor:
            processed_chunk = preprocessor.preprocess_pipeline(
                chunk_data,
                is_training=is_training,
                target_col=target_col
            )
        else:
            processed_chunk = chunk_data

        return processed_chunk, False  # 완료되지 않음

    except Exception as e:
        print(f"청크 처리 오류: {e}")
        return None, True

def safe_chunked_preprocessing():
    """안전한 청크 기반 전처리"""
    print("메모리 안전한 CTR 예측 데이터 전처리 시작...")

    # 메모리 체크
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    print(f"사용 가능 메모리: {available_memory_gb:.1f}GB")

    # 매우 안전한 청크 크기 설정
    if available_memory_gb < 4:
        chunk_size = 25000
        print("⚠️ 매우 작은 청크로 처리")
    elif available_memory_gb < 8:
        chunk_size = 50000
        print("🔧 작은 청크로 처리")
    else:
        chunk_size = 100000
        print("🚀 표준 청크로 처리")

    # 데이터 경로
    train_path = 'data/train.parquet'
    test_path = 'data/test.parquet'

    if not os.path.exists(train_path):
        print(f"오류: {train_path} 파일을 찾을 수 없습니다.")
        return False

    # 전처리기 초기화
    preprocessor = CTRDataPreprocessor()

    try:
        print("\n=== 1단계: 데이터 크기 확인 ===")
        # 작은 샘플로 크기 추정
        sample = pd.read_parquet(train_path)
        total_rows = len(sample)
        print(f"총 데이터 크기: {total_rows:,}행")

        num_chunks = (total_rows // chunk_size) + 1
        print(f"예상 청크 수: {num_chunks}")

        # 샘플 일부로 전처리기 학습
        print("\n=== 2단계: 전처리기 학습 ===")
        sample_small = sample.head(10000)
        del sample
        gc.collect()

        # 카테고리 스캔 (작은 샘플로)
        scan_success = preprocessor.scan_categorical_values(train_path, chunk_size=chunk_size//2)

        # 전처리기 학습
        with tqdm(total=6, desc="전처리기 학습") as pbar:
            sample_processed = preprocessor.preprocess_pipeline(
                sample_small,
                is_training=True,
                target_col='clicked',
                pbar=pbar
            )

        print(f"전처리기 학습 완료! 결과: {sample_processed.shape}")
        del sample_small, sample_processed
        gc.collect()

        print("\n=== 3단계: 청크별 처리 및 저장 ===")
        processed_files = []

        with tqdm(total=num_chunks, desc="청크 처리") as pbar:
            current_row = 0
            chunk_count = 0

            while current_row < total_rows:
                chunk_count += 1

                # 단일 청크 처리
                processed_chunk, is_done = process_single_chunk_file(
                    train_path, current_row, chunk_size,
                    preprocessor, False, 'clicked'
                )

                if processed_chunk is not None:
                    # 처리된 청크를 바로 파일로 저장
                    output_file = f'processed_chunk_{chunk_count}.parquet'
                    processed_chunk.to_parquet(output_file, compression='snappy')
                    processed_files.append(output_file)

                    memory_mb = processed_chunk.memory_usage(deep=True).sum() / 1024 / 1024
                    pbar.set_postfix_str(f"청크 {chunk_count}: {len(processed_chunk):,}행, {memory_mb:.1f}MB")

                    del processed_chunk
                    gc.collect()

                current_row += chunk_size
                pbar.update(1)

                if is_done:
                    break

        print(f"\n총 {len(processed_files)}개 청크 파일 생성됨")

        print("\n=== 4단계: 최종 결합 ===")
        # 작은 배치로 결합
        batch_size = 5
        final_data_parts = []

        for i in range(0, len(processed_files), batch_size):
            batch_files = processed_files[i:i+batch_size]

            print(f"배치 {i//batch_size + 1} 처리 중... ({len(batch_files)}개 파일)")

            batch_data = []
            for file in batch_files:
                chunk_data = pd.read_parquet(file)
                batch_data.append(chunk_data)

            # 배치 결합
            if batch_data:
                batch_combined = pd.concat(batch_data, ignore_index=True)

                # 배치 결과 저장
                batch_output = f'batch_combined_{i//batch_size}.parquet'
                batch_combined.to_parquet(batch_output, compression='snappy')
                final_data_parts.append(batch_output)

                del batch_data, batch_combined
                gc.collect()

        print("\n=== 5단계: Train/Val 분할 ===")
        # 첫 번째 배치만 사용해서 분할 (메모리 절약)
        if final_data_parts:
            print("첫 번째 배치로 train/val 분할...")
            combined_data = pd.read_parquet(final_data_parts[0])

            # 타겟 분리
            feature_cols = [col for col in combined_data.columns if col != 'clicked']
            X = combined_data[feature_cols]
            y = combined_data['clicked']

            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            print(f"분할 결과:")
            print(f"  훈련: {X_train.shape}")
            print(f"  검증: {X_val.shape}")

            # 결과 저장
            os.makedirs('processed_data', exist_ok=True)
            X_train.to_parquet('processed_data/X_train.parquet')
            X_val.to_parquet('processed_data/X_val.parquet')
            y_train.to_parquet('processed_data/y_train.parquet')
            y_val.to_parquet('processed_data/y_val.parquet')

            print("✅ 전처리 완료!")

        # 임시 파일 정리
        print("\n=== 6단계: 임시 파일 정리 ===")
        for file in processed_files + final_data_parts:
            try:
                os.remove(file)
            except:
                pass

        print("임시 파일 정리 완료")
        return True

    except Exception as e:
        print(f"처리 중 오류: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = safe_chunked_preprocessing()
    if success:
        print("전처리가 성공적으로 완료되었습니다!")
    else:
        print("전처리 실패")