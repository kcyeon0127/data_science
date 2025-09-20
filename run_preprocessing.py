#!/usr/bin/env python3
"""
CTR 예측 데이터 전처리 실행 스크립트
체크포인트 기능으로 안전한 대용량 데이터 처리
"""

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import gc
from ctr_preprocessing import CTRDataPreprocessor
from checkpoint_manager import CheckpointManager

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
    # 적절한 크기의 샘플로 전처리기 학습 (청크 크기에 비례)
    sample_size = min(50000, chunk_size * 2)  # 청크 크기의 2배 또는 최대 50k
    print(f"📊 학습 샘플 크기: {sample_size:,}행")

    sample_df = pd.read_parquet(train_path)
    sample_df = sample_df.head(sample_size)

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

    # 전체 데이터 크기 확인 (파일 포맷에 따라)
    if train_path.endswith('.csv'):
        # CSV: 행 수 효율적 계산
        try:
            import subprocess
            result = subprocess.run(['wc', '-l', train_path], capture_output=True, text=True)
            total_rows = int(result.stdout.split()[0]) - 1  # 헤더 제외
            print(f"📊 총 데이터 행 수: {total_rows:,}행 (CSV)")
        except:
            # 폴백: pandas로 확인
            temp_df = pd.read_csv(train_path, nrows=1)  # 헤더만 읽기
            with open(train_path, 'r') as f:
                total_rows = sum(1 for _ in f) - 1  # 헤더 제외
            print(f"📊 총 데이터 행 수: {total_rows:,}행 (CSV 카운트)")
    else:
        # Parquet: 기존 방식
        import pyarrow.parquet as pq
        try:
            parquet_file = pq.ParquetFile(train_path)
            total_rows = parquet_file.metadata.num_rows
            print(f"📊 총 데이터 행 수: {total_rows:,}행 (Parquet 메타데이터)")
        except:
            temp_df = pd.read_parquet(train_path)
            total_rows = len(temp_df)
            del temp_df
            gc.collect()
            print(f"📊 총 데이터 행 수: {total_rows:,}행 (Parquet 전체로드)")

    total_chunks = (total_rows // chunk_size) + 1

    # 시작 지점 결정 및 청크 크기 호환성 확인
    if progress:
        old_chunk_size = progress.get('chunk_size', chunk_size)
        if old_chunk_size != chunk_size:
            print(f"⚠️ 청크 크기 변경 감지: {old_chunk_size:,} → {chunk_size:,}")
            print(f"🔄 새로운 청크 크기로 처음부터 재시작합니다.")
            start_chunk = 0
            # 기존 청크 파일들과 충돌 방지를 위해 백업
            import time
            backup_dir = f"checkpoints_backup_{int(time.time())}"
            import shutil
            if os.path.exists(checkpoint_manager.checkpoint_dir):
                shutil.copytree(checkpoint_manager.checkpoint_dir, backup_dir)
                print(f"📦 기존 청크들을 {backup_dir}에 백업했습니다.")
        else:
            start_chunk = progress['current_chunk']
            print(f"🔄 청크 {start_chunk}부터 재시작합니다. (청크 크기: {chunk_size:,})")
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

                # 청크 로드 (파일 포맷에 따라 다른 방식)
                try:
                    if train_path.endswith('.csv'):
                        # CSV: 메모리 효율적 청크 로드
                        chunk_df = pd.read_csv(
                            train_path,
                            skiprows=range(1, start_row + 1) if start_row > 0 else None,
                            nrows=chunk_size,
                            low_memory=False
                        )

                        if len(chunk_df) == 0:
                            print(f"⚠️ 청크 {chunk_idx}: 빈 청크, 완료")
                            break

                        print(f"📊 청크 {chunk_idx} 로드 완료: {len(chunk_df):,}행 (CSV 스트리밍)")

                    else:
                        # Parquet: 전체 파일 로드 (기존 방식)
                        print(f"⚠️ Parquet 방식: 전체 파일 로드 중...")

                        # 메모리 체크
                        import psutil
                        memory_before = psutil.virtual_memory().percent
                        if memory_before > 70:
                            print(f"⚠️ 메모리 사용률 높음 ({memory_before:.1f}%) - CSV 변환 권장")

                        full_df = pd.read_parquet(train_path)

                        # 청크 범위 확인
                        if start_row >= len(full_df):
                            print(f"⚠️ 청크 {chunk_idx}: 범위 초과, 스킵")
                            del full_df
                            break

                        actual_end = min(end_row, len(full_df))
                        chunk_df = full_df.iloc[start_row:actual_end].copy()

                        # 즉시 전체 데이터 삭제
                        del full_df
                        gc.collect()

                        if len(chunk_df) == 0:
                            print(f"⚠️ 청크 {chunk_idx}: 빈 청크, 스킵")
                            continue

                        print(f"📊 청크 {chunk_idx} 로드 완료: {len(chunk_df):,}행 (Parquet 전체로드)")

                except Exception as e:
                    print(f"❌ 청크 {chunk_idx} 로드 실패: {e}")
                    continue

                # 전처리 적용 (상세 메모리 추적)
                try:
                    import psutil
                    process = psutil.Process()

                    memory_before = psutil.virtual_memory().percent
                    process_memory_before = process.memory_info().rss / 1024 / 1024  # MB

                    processed_chunk = preprocessor.preprocess_pipeline(
                        chunk_df,
                        is_training=False,
                        target_col='clicked'
                    )

                    memory_after = psutil.virtual_memory().percent
                    process_memory_after = process.memory_info().rss / 1024 / 1024  # MB

                    memory_diff = process_memory_after - process_memory_before

                    print(f"🔍 청크 {chunk_idx} 메모리: {process_memory_before:.1f}MB → {process_memory_after:.1f}MB (차이: {memory_diff:+.1f}MB)")

                    if memory_after > 85:
                        print(f"⚠️ 시스템 메모리 사용률 높음: {memory_after:.1f}%")

                    if memory_diff > 100:  # 100MB 이상 증가시 누수 의심
                        print(f"🚨 메모리 누수 의심! 청크당 {memory_diff:.1f}MB 증가")

                except Exception as e:
                    print(f"❌ 청크 {chunk_idx} 전처리 실패: {e}")
                    # 메모리 정리 후 재시도
                    del chunk_df
                    gc.collect()
                    continue

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

                pbar.set_postfix_str(f"청크 {chunk_idx}: {end_row-start_row:,}행 완료")
                pbar.update(1)

            except Exception as e:
                print(f"\n❌ 청크 {chunk_idx} 처리 중 오류: {e}")
                print(f"💡 다음 명령어로 재시작 가능: python run_preprocessing.py --resume")
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

    # 최종 결합 (메모리 안전)
    print("최종 결합 중...")
    if len(combined_parts) == 1:
        # 배치가 하나뿐이면 바로 로드
        final_data = pd.read_parquet(combined_parts[0])
    else:
        # 여러 배치를 순차적으로 결합 (메모리 안전)
        final_data = None
        for i, part_file in enumerate(tqdm(combined_parts, desc="배치 순차 결합")):
            chunk = pd.read_parquet(part_file)
            if final_data is None:
                final_data = chunk
            else:
                final_data = pd.concat([final_data, chunk], ignore_index=True)
                del chunk
                gc.collect()

                # 메모리 체크
                import psutil
                memory_percent = psutil.virtual_memory().percent
                if memory_percent > 85:
                    print(f"⚠️ 메모리 사용률 높음: {memory_percent:.1f}%")

    print(f"최종 데이터 크기: {final_data.shape}")
    return final_data

def main():
    import argparse
    parser = argparse.ArgumentParser(description='CTR 예측 데이터 전처리')
    parser.add_argument('--resume', action='store_true', help='중간부터 재시작')
    parser.add_argument('--retrain', action='store_true', help='전처리기 재학습')
    parser.add_argument('--chunk-size', type=int, default=100000, help='청크 크기')
    parser.add_argument('--data-path', default='data/train.parquet', help='데이터 파일 경로')
    args = parser.parse_args()

    print("🔄 CTR 예측 데이터 전처리 시작")

    # 메모리 체크 및 청크 크기 자동 조정
    import psutil
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    print(f"💾 사용 가능 메모리: {available_memory_gb:.1f}GB")

    # 메모리가 부족하면 청크 크기 자동 감소
    if available_memory_gb < 2 and args.chunk_size > 10000:
        suggested_chunk_size = 10000
        print(f"⚠️ 메모리 부족 감지! 청크 크기를 {suggested_chunk_size:,}로 조정하는 것을 권장합니다.")
        print(f"현재 설정: {args.chunk_size:,} → 권장: {suggested_chunk_size:,}")
    elif available_memory_gb < 4 and args.chunk_size > 25000:
        suggested_chunk_size = 25000
        print(f"⚠️ 메모리 제한적! 청크 크기를 {suggested_chunk_size:,}로 조정하는 것을 권장합니다.")
        print(f"현재 설정: {args.chunk_size:,} → 권장: {suggested_chunk_size:,}")

    # 경로 설정
    checkpoint_manager = CheckpointManager()

    if not os.path.exists(args.data_path):
        print(f"❌ {args.data_path} 파일이 없습니다.")
        return

    try:
        # 1. 전처리기 준비
        print("\n🔧 전처리기 준비...")
        preprocessor = prepare_preprocessor(
            args.data_path,
            checkpoint_manager,
            chunk_size=args.chunk_size,
            force_retrain=args.retrain
        )

        if preprocessor is None:
            print("❌ 전처리기 준비 실패")
            return

        # 2. 청크 처리
        print("\n📦 청크 처리...")
        success = process_chunks_with_checkpoint(
            args.data_path,
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
            checkpoint_manager.clear_checkpoints()

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("\n💡 다음 명령어로 재시작 가능:")
        print("python run_preprocessing.py --resume")

if __name__ == "__main__":
    main()