#!/usr/bin/env python3
"""
체크포인트 기능이 있는 CTR 예측 데이터 전처리 스크립트
전처리기 저장/로드 및 중간 재시작 기능 포함
"""

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import gc
import pickle
import json
from ctr_preprocessing import CTRDataPreprocessor
from preprocessing_utils import CTRPreprocessingUtils

class CheckpointManager:
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_preprocessor(self, preprocessor, filename='preprocessor.pkl'):
        """전처리기 저장"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor, f)
        print(f"✅ 전처리기 저장: {filepath}")

    def load_preprocessor(self, filename='preprocessor.pkl'):
        """전처리기 로드"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                preprocessor = pickle.load(f)
            print(f"✅ 전처리기 로드: {filepath}")
            return preprocessor
        return None

    def save_progress(self, progress_info):
        """진행 상황 저장"""
        filepath = os.path.join(self.checkpoint_dir, 'progress.json')
        with open(filepath, 'w') as f:
            json.dump(progress_info, f, indent=2)
        print(f"📊 진행상황 저장: {progress_info['current_chunk']}/{progress_info['total_chunks']}")

    def load_progress(self):
        """진행 상황 로드"""
        filepath = os.path.join(self.checkpoint_dir, 'progress.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                progress = json.load(f)
            print(f"📊 진행상황 로드: {progress['current_chunk']}/{progress['total_chunks']}")
            return progress
        return None

    def list_processed_chunks(self):
        """처리된 청크 파일 목록"""
        chunk_files = []
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith('chunk_') and file.endswith('.parquet'):
                chunk_files.append(os.path.join(self.checkpoint_dir, file))
        return sorted(chunk_files)

def prepare_preprocessor(train_path, checkpoint_manager, chunk_size=50000, force_retrain=False):
    """전처리기 준비 (로드 또는 새로 학습)"""

    if not force_retrain:
        # 기존 전처리기 로드 시도
        preprocessor = checkpoint_manager.load_preprocessor()
        if preprocessor is not None:
            print("✅ 기존 전처리기를 사용합니다.")
            return preprocessor

    print("🔧 새로운 전처리기를 학습합니다...")

    # 새 전처리기 생성
    preprocessor = CTRDataPreprocessor()

    print("\n=== 1단계: 카테고리 스캔 ===")
    scan_success = preprocessor.scan_categorical_values(train_path, chunk_size=chunk_size)

    if not scan_success:
        print("⚠️ 카테고리 스캔 실패")
        return None

    print("\n=== 2단계: 전처리기 학습 ===")
    # 작은 샘플로 전처리기 학습
    sample_df = pd.read_parquet(train_path)
    sample_df = sample_df.head(10000)

    with tqdm(total=6, desc="전처리기 학습") as pbar:
        sample_processed = preprocessor.preprocess_pipeline(
            sample_df,
            is_training=True,
            target_col='clicked',
            pbar=pbar
        )

    print(f"전처리기 학습 완료! 결과: {sample_processed.shape}")

    # 전처리기 저장
    checkpoint_manager.save_preprocessor(preprocessor)

    # 메모리 정리
    del sample_df, sample_processed
    gc.collect()

    return preprocessor

def process_chunks_with_checkpoint(train_path, preprocessor, checkpoint_manager, chunk_size=100000):
    """체크포인트 기능으로 청크 처리"""

    # 진행 상황 확인
    progress = checkpoint_manager.load_progress()

    # 전체 데이터 크기 확인
    temp_df = pd.read_parquet(train_path)
    total_rows = len(temp_df)
    total_chunks = (total_rows // chunk_size) + 1
    del temp_df
    gc.collect()

    # 시작 지점 결정
    if progress:
        start_chunk = progress['current_chunk']
        print(f"🔄 청크 {start_chunk}부터 재시작합니다.")
    else:
        start_chunk = 0
        print(f"🚀 처음부터 청크 처리를 시작합니다.")

    print(f"총 {total_chunks}개 청크 중 {start_chunk}부터 처리")

    # 청크별 처리
    with tqdm(total=total_chunks, initial=start_chunk, desc="청크 처리") as pbar:
        for chunk_idx in range(start_chunk, total_chunks):
            try:
                # 청크 범위 계산
                start_row = chunk_idx * chunk_size
                end_row = min(start_row + chunk_size, total_rows)

                # 이미 처리된 청크인지 확인
                chunk_file = os.path.join(checkpoint_manager.checkpoint_dir, f'chunk_{chunk_idx:04d}.parquet')

                if os.path.exists(chunk_file):
                    pbar.set_postfix_str(f"청크 {chunk_idx} 스킵 (이미 처리됨)")
                    pbar.update(1)
                    continue

                # 청크 로드
                full_df = pd.read_parquet(train_path)
                chunk_df = full_df.iloc[start_row:end_row].copy()
                del full_df
                gc.collect()

                # 전처리 적용
                processed_chunk = preprocessor.preprocess_pipeline(
                    chunk_df,
                    is_training=False,
                    target_col='clicked'
                )

                # 청크 저장
                processed_chunk.to_parquet(chunk_file, compression='snappy')

                # 진행 상황 저장
                progress_info = {
                    'current_chunk': chunk_idx + 1,
                    'total_chunks': total_chunks,
                    'total_rows': total_rows,
                    'chunk_size': chunk_size
                }
                checkpoint_manager.save_progress(progress_info)

                # 메모리 정리
                del chunk_df, processed_chunk
                gc.collect()

                memory_mb = chunk_df.memory_usage(deep=True).sum() / 1024 / 1024 if 'chunk_df' in locals() else 0
                pbar.set_postfix_str(f"청크 {chunk_idx}: {end_row-start_row:,}행 완료")
                pbar.update(1)

            except Exception as e:
                print(f"\n❌ 청크 {chunk_idx} 처리 중 오류: {e}")
                print(f"💡 다음 명령어로 재시작 가능: python run_preprocessing_checkpoint.py --resume")
                return False

    print("✅ 모든 청크 처리 완료!")
    return True

def combine_chunks(checkpoint_manager):
    """처리된 청크들을 결합"""

    print("\n=== 청크 결합 ===")
    chunk_files = checkpoint_manager.list_processed_chunks()

    if not chunk_files:
        print("❌ 처리된 청크가 없습니다.")
        return False

    print(f"📁 {len(chunk_files)}개 청크를 결합합니다.")

    # 배치별로 결합 (메모리 안전)
    batch_size = 5
    combined_parts = []

    for i in range(0, len(chunk_files), batch_size):
        batch_files = chunk_files[i:i+batch_size]
        print(f"배치 {i//batch_size + 1} 처리 중... ({len(batch_files)}개)")

        batch_chunks = []
        for file in tqdm(batch_files, desc="청크 로드"):
            chunk = pd.read_parquet(file)
            batch_chunks.append(chunk)

        # 배치 결합
        batch_combined = pd.concat(batch_chunks, ignore_index=True)

        # 배치 저장
        batch_file = os.path.join(checkpoint_manager.checkpoint_dir, f'batch_{i//batch_size}.parquet')
        batch_combined.to_parquet(batch_file, compression='snappy')
        combined_parts.append(batch_file)

        del batch_chunks, batch_combined
        gc.collect()

    # 최종 결합
    print("최종 결합 중...")
    final_chunks = []
    for part_file in tqdm(combined_parts, desc="배치 로드"):
        chunk = pd.read_parquet(part_file)
        final_chunks.append(chunk)

    final_data = pd.concat(final_chunks, ignore_index=True)
    print(f"최종 데이터 크기: {final_data.shape}")

    return final_data

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='중간부터 재시작')
    parser.add_argument('--retrain', action='store_true', help='전처리기 재학습')
    parser.add_argument('--chunk-size', type=int, default=100000, help='청크 크기')
    args = parser.parse_args()

    print("🔄 체크포인트 기능이 있는 CTR 전처리 시작")

    # 경로 설정
    train_path = 'data/train.parquet'
    checkpoint_manager = CheckpointManager()

    if not os.path.exists(train_path):
        print(f"❌ {train_path} 파일이 없습니다.")
        return

    try:
        # 1. 전처리기 준비
        print("\n🔧 전처리기 준비...")
        preprocessor = prepare_preprocessor(
            train_path,
            checkpoint_manager,
            chunk_size=args.chunk_size//2,
            force_retrain=args.retrain
        )

        if preprocessor is None:
            print("❌ 전처리기 준비 실패")
            return

        # 2. 청크 처리
        print("\n📦 청크 처리...")
        success = process_chunks_with_checkpoint(
            train_path,
            preprocessor,
            checkpoint_manager,
            chunk_size=args.chunk_size
        )

        if not success:
            print("❌ 청크 처리 실패")
            return

        # 3. 청크 결합 및 분할
        print("\n🔗 청크 결합...")
        combined_data = combine_chunks(checkpoint_manager)

        if combined_data is None:
            print("❌ 청크 결합 실패")
            return

        # 4. Train/Val 분할
        print("\n✂️ 데이터 분할...")
        feature_cols = [col for col in combined_data.columns if col != 'clicked']
        X = combined_data[feature_cols]
        y = combined_data['clicked']

        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 5. 결과 저장
        print("\n💾 결과 저장...")
        os.makedirs('processed_data', exist_ok=True)

        save_files = [
            ("X_train", X_train),
            ("X_val", X_val),
            ("y_train", y_train),
            ("y_val", y_val)
        ]

        for name, data in tqdm(save_files, desc="저장"):
            data.to_parquet(f'processed_data/{name}.parquet')

        print("✅ 전처리 완료!")
        print(f"📊 결과: 훈련 {X_train.shape}, 검증 {X_val.shape}")

        # 체크포인트 정리 여부 묻기
        response = input("\n🗑️ 체크포인트 파일들을 삭제하시겠습니까? (y/N): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(checkpoint_manager.checkpoint_dir)
            print("✅ 체크포인트 파일 삭제 완료")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("\n💡 다음 명령어로 재시작 가능:")
        print("python run_preprocessing_checkpoint.py --resume")

if __name__ == "__main__":
    main()