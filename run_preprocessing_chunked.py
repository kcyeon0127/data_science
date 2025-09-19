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
    """데이터를 청크 단위로 처리 - 메모리 안전한 방식"""

    print(f"청크 방식으로 데이터 처리: 청크 크기 {chunk_size:,}행")

    processed_chunks = []
    chunk_count = 0
    temp_files = []

    try:
        # 파일 크기만 먼저 확인 (전체 로드 없이)
        sample_df = pd.read_parquet(file_path, nrows=1000)
        file_size = os.path.getsize(file_path)
        sample_memory = sample_df.memory_usage(deep=True).sum()
        estimated_rows = int((file_size / sample_memory) * 1000)

        del sample_df
        gc.collect()

        num_chunks = (estimated_rows // chunk_size) + 1
        print(f"예상 {estimated_rows:,}행을 {chunk_size:,}개씩 {num_chunks}개 청크로 처리합니다.")

        # 파일을 작은 단위로 읽어가며 처리
        with tqdm(total=num_chunks, desc=f"청크 처리 ({chunk_size:,}행씩)") as pbar:
            offset = 0

            while True:
                try:
                    # 청크 단위로 읽기
                    chunk = pd.read_parquet(file_path)

                    # 실제 청크 분할
                    start_idx = offset
                    end_idx = min(offset + chunk_size, len(chunk))

                    if start_idx >= len(chunk):
                        break

                    current_chunk = chunk.iloc[start_idx:end_idx].copy()
                    del chunk  # 즉시 메모리 해제
                    gc.collect()

                    chunk_count += 1

                    # 메모리 사용량 모니터링
                    memory_mb = current_chunk.memory_usage(deep=True).sum() / 1024 / 1024
                    pbar.set_postfix_str(f"청크 {chunk_count}, 메모리: {memory_mb:.1f}MB")

                    # 전처리 적용
                    if preprocessor:
                        chunk_processed = preprocessor.preprocess_pipeline(
                            current_chunk,
                            is_training=is_training,
                            target_col=target_col
                        )
                    else:
                        chunk_processed = current_chunk.copy()

                    # 처리된 청크를 임시 파일로 저장 (메모리 절약)
                    temp_file = f'temp_chunk_{chunk_count}.parquet'
                    chunk_processed.to_parquet(temp_file, compression='snappy')
                    temp_files.append(temp_file)

                    # 메모리 정리
                    del current_chunk, chunk_processed
                    gc.collect()

                    offset += chunk_size
                    pbar.update(1)

                    # 중간 진행 상황 출력
                    if chunk_count % 10 == 0:
                        print(f"\n청크 {chunk_count}개 처리 완료, 메모리 정리 중...")
                        gc.collect()

                except Exception as e:
                    print(f"\n청크 {chunk_count} 처리 중 오류: {e}")
                    break

        # 임시 파일들을 다시 로드하여 processed_chunks 생성
        print(f"\n임시 파일들을 로드 중... ({len(temp_files)}개)")
        for temp_file in tqdm(temp_files, desc="청크 로드"):
            try:
                chunk_data = pd.read_parquet(temp_file)
                processed_chunks.append(chunk_data)
            except Exception as e:
                print(f"임시 파일 {temp_file} 로드 오류: {e}")

        # 임시 파일 정리
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

        print(f"총 {len(processed_chunks)}개 청크 처리 완료")
        return processed_chunks

    except Exception as e:
        print(f"청크 처리 전체 오류: {e}")

        # 임시 파일 정리
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

        return []

def process_data_simple_chunks(file_path, chunk_size, preprocessor, is_training, target_col):
    """간단한 청크 처리 (PyArrow 없을 때)"""
    print("간단한 청크 처리 방식을 사용합니다...")

    # 전체 데이터 로드 후 분할
    print("전체 데이터를 로드합니다...")
    full_df = pd.read_parquet(file_path)
    total_rows = len(full_df)

    print(f"총 {total_rows:,}행을 {chunk_size:,}개씩 처리합니다.")

    processed_chunks = []
    num_chunks = (total_rows // chunk_size) + (1 if total_rows % chunk_size else 0)

    with tqdm(total=num_chunks, desc=f"청크 처리 ({chunk_size:,}행씩)") as pbar:
        for i in range(0, total_rows, chunk_size):
            try:
                end_idx = min(i + chunk_size, total_rows)
                chunk = full_df.iloc[i:end_idx].copy()

                memory_mb = chunk.memory_usage(deep=True).sum() / 1024 / 1024
                pbar.set_postfix_str(f"메모리: {memory_mb:.1f}MB, 행: {len(chunk):,}")

                # 전처리 적용
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
                print(f"청크 {i//chunk_size + 1} 처리 중 오류: {e}")
                continue

            pbar.update(1)

            # 메모리 정리
            del chunk
            if 'chunk_processed' in locals():
                del chunk_processed
            gc.collect()

    # 원본 데이터프레임 삭제
    del full_df
    gc.collect()

    return processed_chunks

def combine_chunks_efficiently(chunks, output_path):
    """청크들을 배치별로 효율적으로 결합"""
    print(f"청크 결합 중... 총 {len(chunks)}개 청크")

    if len(chunks) == 0:
        raise ValueError("결합할 청크가 없습니다.")

    if len(chunks) == 1:
        return chunks[0]

    batch_size = 5  # 한번에 5개씩 결합
    temp_files = []

    try:
        # 배치별 결합
        with tqdm(total=(len(chunks) // batch_size) + 1, desc="배치별 결합") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i+batch_size]

                pbar.set_postfix_str(f"배치 {i//batch_size + 1}: {len(batch_chunks)}개 청크")

                # 배치 내 청크들 결합
                if len(batch_chunks) == 1:
                    batch_combined = batch_chunks[0]
                else:
                    batch_combined = pd.concat(batch_chunks, ignore_index=True)

                # 임시 파일로 저장
                temp_file = output_path.replace('.parquet', f'_batch_{i//batch_size}.parquet')
                batch_combined.to_parquet(temp_file, compression='snappy')
                temp_files.append(temp_file)

                # 메모리 정리
                del batch_chunks, batch_combined
                gc.collect()

                pbar.update(1)

        print(f"\n배치별 결합 완료. {len(temp_files)}개 배치 파일 생성.")

        # 최종 결합
        print("최종 배치 파일들을 결합 중...")
        final_chunks = []

        with tqdm(total=len(temp_files), desc="최종 결합") as pbar:
            for temp_file in temp_files:
                chunk = pd.read_parquet(temp_file)
                final_chunks.append(chunk)
                pbar.set_postfix_str(f"로드: {len(chunk):,}행")
                pbar.update(1)

        # 최종 결합
        print("전체 데이터 결합 중...")
        final_combined = pd.concat(final_chunks, ignore_index=True)

        # 임시 파일 삭제
        print("임시 파일 정리 중...")
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

        print(f"결합 완료! 최종 크기: {final_combined.shape}")
        return final_combined

    except Exception as e:
        print(f"청크 결합 중 오류: {e}")

        # 임시 파일 정리
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass

        # 폴백: 첫 번째 청크만 반환
        print("폴백: 첫 번째 청크만 사용합니다.")
        return chunks[0]

def main():
    print("메모리 효율적인 CTR 예측 데이터 전처리 시작...")
    print(f"현재 사용 가능 메모리를 확인하여 청크 크기를 조정합니다.")

    # 시스템 메모리 확인
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    print(f"사용 가능 메모리: {available_memory_gb:.1f}GB")

    # 청크 크기 설정 (커맨드 라인 인자나 환경 변수로 조정 가능)
    import sys
    if len(sys.argv) > 1:
        try:
            chunk_size = int(sys.argv[1])
            print(f"📋 사용자 지정 청크 크기: {chunk_size:,}행")
        except ValueError:
            chunk_size = 100000
            print("⚠️  잘못된 청크 크기입니다. 기본값을 사용합니다.")
    else:
        # 메모리에 따른 자동 청크 크기 조정
        if available_memory_gb < 4:
            chunk_size = 50000
            print("⚠️  메모리가 부족합니다. 작은 청크 크기로 처리합니다.")
        elif available_memory_gb < 8:
            chunk_size = 100000
            print("🔧 중간 청크 크기로 처리합니다.")
        else:
            chunk_size = 200000
            print("🚀 큰 청크 크기로 빠르게 처리합니다.")

    print(f"💾 청크 크기: {chunk_size:,}행")

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
        print("\n=== 1단계: 전체 데이터 카테고리 스캔 ===")
        # 전체 데이터의 카테고리 값들을 미리 스캔
        scan_success = preprocessor.scan_categorical_values(train_path, chunk_size=chunk_size//2)

        if not scan_success:
            print("⚠️  카테고리 스캔 실패. 기본 방식으로 처리합니다.")

        print("\n=== 2단계: 샘플 데이터로 전처리기 학습 ===")
        # 작은 샘플로 전처리기 학습 (인코더, 스케일러 등)
        sample_df = pd.read_parquet(train_path)
        sample_df = sample_df.head(10000)  # 첫 10000행만 사용
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

        print("\n=== 3단계: 훈련 데이터 청크 처리 ===")
        # 전체 훈련 데이터 처리
        train_chunks = process_data_in_chunks(
            train_path,
            chunk_size=chunk_size,
            preprocessor=preprocessor,
            is_training=False,  # 이미 학습된 전처리기 사용
            target_col='clicked'
        )

        print("\n=== 4단계: 훈련 데이터 결합 및 분할 ===")
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

        print("\n=== 5단계: 테스트 데이터 처리 ===")
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

        print("\n=== 6단계: 결과 저장 ===")
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

        print("\n=== 7단계: 피처 정보 저장 ===")
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