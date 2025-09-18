#!/usr/bin/env python3
"""
메모리 효율적인 CTR 예측 데이터 전처리 실행 스크립트
큰 데이터셋을 청크 단위로 처리하여 메모리 사용량을 줄입니다.
"""

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import gc
from ctr_preprocessing import CTRDataPreprocessor
from preprocessing_utils import CTRPreprocessingUtils

def process_data_in_chunks(file_path, chunk_size=100000, preprocessor=None, is_training=True, target_col='clicked'):
    """데이터를 청크 단위로 처리"""

    # 전체 행 수 먼저 확인
    temp_df = pd.read_parquet(file_path, columns=['clicked'] if 'train' in file_path else [])
    total_rows = len(temp_df)
    del temp_df
    gc.collect()

    print(f"총 {total_rows:,}행을 {chunk_size:,}개씩 처리합니다.")

    processed_chunks = []

    # 청크별 처리
    with tqdm(total=total_rows//chunk_size + 1, desc=f"청크 처리 ({chunk_size:,}행씩)") as pbar:
        for chunk in pd.read_parquet(file_path, chunksize=chunk_size):
            # 메모리 사용량 모니터링
            pbar.set_postfix_str(f"메모리: {chunk.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB")

            # 전처리 적용
            try:
                if preprocessor:
                    chunk_processed = preprocessor.preprocess_pipeline(
                        chunk,
                        is_training=is_training,
                        target_col=target_col
                    )
                else:
                    chunk_processed = chunk

                processed_chunks.append(chunk_processed)

            except Exception as e:
                print(f"청크 처리 중 오류: {e}")
                # 실패한 청크는 기본 처리만 수행
                processed_chunks.append(chunk)

            pbar.update(1)

            # 메모리 정리
            del chunk
            gc.collect()

    return processed_chunks

def combine_chunks_efficiently(chunks, output_path):
    """청크들을 효율적으로 결합"""
    print(f"청크 결합 중... 총 {len(chunks)}개 청크")

    with tqdm(total=len(chunks), desc="청크 결합") as pbar:
        # 첫 번째 청크로 시작
        combined = chunks[0].copy()
        pbar.update(1)

        # 나머지 청크들을 순차적으로 결합
        for i, chunk in enumerate(chunks[1:], 1):
            try:
                combined = pd.concat([combined, chunk], ignore_index=True)
                pbar.set_postfix_str(f"현재 크기: {len(combined):,}행")
                pbar.update(1)

                # 메모리 정리
                del chunks[i-1]  # 이미 사용한 청크 삭제
                if i % 5 == 0:  # 5개마다 가비지 컬렉션
                    gc.collect()

            except MemoryError:
                print(f"메모리 부족으로 {i}번째 청크에서 중간 저장합니다.")
                # 중간 저장
                temp_path = output_path.replace('.parquet', f'_temp_{i}.parquet')
                combined.to_parquet(temp_path)
                del combined
                gc.collect()
                combined = chunk.copy()

    return combined

def main():
    print("메모리 효율적인 CTR 예측 데이터 전처리 시작...")
    print(f"현재 사용 가능 메모리를 확인하여 청크 크기를 조정합니다.")

    # 시스템 메모리 확인
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    print(f"사용 가능 메모리: {available_memory_gb:.1f}GB")

    # 메모리에 따른 청크 크기 조정
    if available_memory_gb < 4:
        chunk_size = 50000
        print("⚠️  메모리가 부족합니다. 작은 청크 크기로 처리합니다.")
    elif available_memory_gb < 8:
        chunk_size = 100000
        print("🔧 중간 청크 크기로 처리합니다.")
    else:
        chunk_size = 200000
        print("🚀 큰 청크 크기로 빠르게 처리합니다.")

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

    # 출력 디렉토리 생성
    os.makedirs('processed_data', exist_ok=True)

    try:
        print("\n=== 1단계: 샘플 데이터로 전처리기 학습 ===")
        # 작은 샘플로 전처리기 학습 (인코더, 스케일러 등)
        sample_df = pd.read_parquet(train_path, nrows=10000)
        print(f"샘플 데이터 로드: {sample_df.shape}")

        # 샘플로 전처리기 학습
        with tqdm(total=6, desc="전처리기 학습") as pbar:
            sample_processed = preprocessor.preprocess_pipeline(
                sample_df,
                is_training=True,
                target_col='clicked',
                pbar=pbar
            )

        print(f"전처리기 학습 완료! 샘플 결과: {sample_processed.shape}")
        del sample_df, sample_processed
        gc.collect()

        print("\n=== 2단계: 훈련 데이터 청크 처리 ===")
        # 전체 훈련 데이터 처리
        train_chunks = process_data_in_chunks(
            train_path,
            chunk_size=chunk_size,
            preprocessor=preprocessor,
            is_training=False,  # 이미 학습된 전처리기 사용
            target_col='clicked'
        )

        print("\n=== 3단계: 훈련 데이터 결합 및 분할 ===")
        # 청크 결합
        train_combined = combine_chunks_efficiently(train_chunks, 'processed_data/train_combined.parquet')

        # 타겟 분리
        feature_cols = [col for col in train_combined.columns if col != 'clicked']
        X = train_combined[feature_cols]
        y = train_combined['clicked']

        # 메모리 정리
        del train_combined, train_chunks
        gc.collect()

        print("훈련/검증 분할 중...")
        from sklearn.model_selection import train_test_split

        # 메모리 효율적인 분할
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"분할 완료:")
        print(f"  훈련 세트: {X_train.shape}")
        print(f"  검증 세트: {X_val.shape}")

        # 메모리 정리
        del X, y
        gc.collect()

        print("\n=== 4단계: 테스트 데이터 처리 ===")
        X_test = None
        if test_path:
            test_chunks = process_data_in_chunks(
                test_path,
                chunk_size=chunk_size,
                preprocessor=preprocessor,
                is_training=False
            )

            X_test = combine_chunks_efficiently(test_chunks, 'processed_data/test_combined.parquet')

            # 테스트 데이터에서 타겟 컬럼 제거 (있다면)
            if 'clicked' in X_test.columns:
                X_test = X_test.drop('clicked', axis=1)

            print(f"  테스트 세트: {X_test.shape}")
            del test_chunks
            gc.collect()

        print("\n=== 5단계: 결과 저장 ===")
        save_files = [
            ("X_train", X_train),
            ("X_val", X_val),
            ("y_train", y_train),
            ("y_val", y_val)
        ]

        if X_test is not None:
            save_files.append(("X_test", X_test))

        with tqdm(total=len(save_files), desc="데이터 저장") as pbar:
            for name, data in save_files:
                try:
                    data.to_parquet(f'processed_data/{name}.parquet')
                    pbar.set_postfix_str(f"{name}: {data.shape}")
                    pbar.update(1)

                    # 저장 후 메모리 정리
                    del data
                    gc.collect()

                except Exception as e:
                    print(f"{name} 저장 중 오류: {e}")

        print("\n=== 6단계: 피처 정보 저장 ===")
        # 피처 정보 저장
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

        print("\n" + "="*60)
        print("✅ 메모리 효율적 전처리 완료!")
        print("생성된 파일들:")
        print("- processed_data/X_train.parquet")
        print("- processed_data/X_val.parquet")
        print("- processed_data/y_train.parquet")
        print("- processed_data/y_val.parquet")
        if X_test is not None:
            print("- processed_data/X_test.parquet")
        print("- processed_data/feature_info.json")
        print(f"총 {len(feature_cols)}개의 피처가 생성되었습니다.")
        print("="*60)

        return True

    except Exception as e:
        print(f"처리 중 오류 발생: {e}")
        print("메모리 부족일 가능성이 높습니다. 더 작은 청크 크기로 다시 시도해보세요.")
        return False

if __name__ == "__main__":
    # 메모리 사용량 모니터링
    import psutil
    process = psutil.Process()

    print(f"시작 메모리 사용량: {process.memory_info().rss / 1024 / 1024:.1f}MB")

    success = main()

    if success:
        print(f"최종 메모리 사용량: {process.memory_info().rss / 1024 / 1024:.1f}MB")
    else:
        print("처리 실패. 시스템 리소스를 확인해주세요.")