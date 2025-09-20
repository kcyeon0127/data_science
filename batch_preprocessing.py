#!/usr/bin/env python3
"""
배치 작업으로 분할 처리 - 메모리 안전한 대용량 데이터 처리
"""

import os
import pandas as pd
import subprocess
import time
from tqdm.auto import tqdm

class BatchPreprocessor:
    def __init__(self):
        self.batch_dir = 'batch_data'
        self.output_dir = 'processed_data'
        os.makedirs(self.batch_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def step1_split_data(self, input_file='data/train.parquet', batch_size=500000):
        """1단계: 대용량 파일을 배치로 분할"""

        print("🔄 1단계: 데이터 분할 시작...")
        print(f"📦 배치 크기: {batch_size:,}행")

        try:
            # 전체 파일 로드 (한 번만)
            print("📂 전체 파일 로딩 중...")
            df = pd.read_parquet(input_file)
            total_rows = len(df)

            print(f"📊 총 데이터: {total_rows:,}행")

            num_batches = (total_rows // batch_size) + 1
            print(f"📦 생성할 배치 수: {num_batches}개")

            # 배치별로 분할 저장
            batch_files = []

            for i in tqdm(range(num_batches), desc="배치 분할"):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, total_rows)

                if start_idx >= total_rows:
                    break

                batch_data = df.iloc[start_idx:end_idx].copy()
                batch_file = os.path.join(self.batch_dir, f'batch_{i:03d}.parquet')
                batch_data.to_parquet(batch_file, compression='snappy')
                batch_files.append(batch_file)

                print(f"📁 배치 {i}: {len(batch_data):,}행 → {batch_file}")

            # 메모리 정리
            del df
            import gc
            gc.collect()

            print(f"✅ 분할 완료: {len(batch_files)}개 배치 파일")
            return batch_files

        except Exception as e:
            print(f"❌ 분할 실패: {e}")
            return []

    def step2_process_batches(self, batch_files, chunk_size=100000):
        """2단계: 각 배치를 개별 프로세스로 처리"""

        print(f"\n🔄 2단계: 배치별 처리 시작 ({len(batch_files)}개)")

        processed_files = []

        for i, batch_file in enumerate(batch_files):
            print(f"\n📦 배치 {i+1}/{len(batch_files)} 처리: {batch_file}")

            output_file = os.path.join(self.output_dir, f'processed_batch_{i:03d}.parquet')

            # 이미 처리된 파일이 있으면 스킵
            if os.path.exists(output_file):
                print(f"✅ 이미 처리됨: {output_file}")
                processed_files.append(output_file)
                continue

            # 개별 배치 처리 (새 프로세스)
            cmd = [
                'python', 'run_preprocessing.py',
                '--data-path', batch_file,
                '--chunk-size', str(chunk_size),
                '--retrain'  # 각 배치마다 전처리기 재학습
            ]

            print(f"🚀 실행: {' '.join(cmd)}")

            try:
                # 새 프로세스로 실행 (메모리 완전 분리)
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

                if result.returncode == 0:
                    print(f"✅ 배치 {i} 처리 완료")

                    # 처리된 결과를 지정된 위치로 이동
                    if os.path.exists('processed_data/X_train.parquet'):
                        # 결과 파일들을 배치별로 이름 변경
                        import shutil
                        shutil.move('processed_data/X_train.parquet', output_file)
                        processed_files.append(output_file)

                else:
                    print(f"❌ 배치 {i} 처리 실패:")
                    print(result.stderr)

            except subprocess.TimeoutExpired:
                print(f"⏰ 배치 {i} 타임아웃")
            except Exception as e:
                print(f"❌ 배치 {i} 실행 오류: {e}")

        return processed_files

    def step3_combine_results(self, processed_files):
        """3단계: 처리된 배치들을 최종 결합"""

        print(f"\n🔄 3단계: 결과 결합 ({len(processed_files)}개 파일)")

        if not processed_files:
            print("❌ 처리된 파일이 없습니다.")
            return False

        try:
            # 배치별로 로드하여 결합
            combined_data = []

            for file in tqdm(processed_files, desc="배치 결합"):
                batch_data = pd.read_parquet(file)
                combined_data.append(batch_data)
                print(f"📁 로드: {file} ({batch_data.shape})")

            # 최종 결합
            final_data = pd.concat(combined_data, ignore_index=True)
            print(f"📊 최종 데이터 크기: {final_data.shape}")

            # Train/Val 분할
            feature_cols = [col for col in final_data.columns if col != 'clicked']
            X = final_data[feature_cols]
            y = final_data['clicked']

            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # 최종 저장
            X_train.to_parquet('processed_data/X_train_final.parquet')
            X_val.to_parquet('processed_data/X_val_final.parquet')
            y_train.to_parquet('processed_data/y_train_final.parquet')
            y_val.to_parquet('processed_data/y_val_final.parquet')

            print("✅ 최종 결과 저장 완료!")
            print(f"📊 훈련: {X_train.shape}, 검증: {X_val.shape}")

            return True

        except Exception as e:
            print(f"❌ 결합 실패: {e}")
            return False

    def run_full_pipeline(self, input_file='data/train.parquet', batch_size=500000, chunk_size=100000):
        """전체 파이프라인 실행"""

        print("🚀 배치 처리 파이프라인 시작!")
        print(f"📂 입력 파일: {input_file}")
        print(f"📦 배치 크기: {batch_size:,}행")
        print(f"🔧 청크 크기: {chunk_size:,}행")

        # 1단계: 분할
        batch_files = self.step1_split_data(input_file, batch_size)
        if not batch_files:
            return False

        # 2단계: 배치별 처리
        processed_files = self.step2_process_batches(batch_files, chunk_size)
        if not processed_files:
            return False

        # 3단계: 결합
        success = self.step3_combine_results(processed_files)

        if success:
            print("\n🎉 배치 처리 완료!")
            print("💡 임시 파일 정리는 수동으로 하세요:")
            print(f"rm -rf {self.batch_dir}")

        return success

def main():
    processor = BatchPreprocessor()

    print("📋 배치 처리 옵션:")
    print("1. 소형 배치 (50만행) - 안전")
    print("2. 중형 배치 (100만행) - 균형")
    print("3. 대형 배치 (200만행) - 빠름")

    choice = input("선택 (1-3): ").strip()

    batch_sizes = {'1': 500000, '2': 1000000, '3': 2000000}
    batch_size = batch_sizes.get(choice, 500000)

    success = processor.run_full_pipeline(
        input_file='data/train.parquet',
        batch_size=batch_size,
        chunk_size=100000
    )

    if success:
        print("✅ 전체 파이프라인 성공!")
    else:
        print("❌ 파이프라인 실패")

if __name__ == "__main__":
    main()