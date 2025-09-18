#!/usr/bin/env python3
"""
시스템 메모리 상태 확인 및 적절한 전처리 방법 추천
"""

import psutil
import os

def check_system_resources():
    """시스템 리소스 확인"""
    print("=" * 50)
    print("시스템 리소스 확인")
    print("=" * 50)

    # 메모리 정보
    memory = psutil.virtual_memory()
    print(f"총 메모리: {memory.total / (1024**3):.1f} GB")
    print(f"사용 가능: {memory.available / (1024**3):.1f} GB ({memory.percent:.1f}% 사용 중)")

    # 디스크 정보
    disk = psutil.disk_usage('.')
    print(f"디스크 여유: {disk.free / (1024**3):.1f} GB")

    # 데이터 파일 크기
    train_path = 'data/train.parquet'
    test_path = 'data/test.parquet'

    if os.path.exists(train_path):
        train_size = os.path.getsize(train_path) / (1024**3)
        print(f"train.parquet 크기: {train_size:.2f} GB")
    else:
        print("❌ train.parquet 파일을 찾을 수 없습니다.")
        return False

    if os.path.exists(test_path):
        test_size = os.path.getsize(test_path) / (1024**3)
        print(f"test.parquet 크기: {test_size:.2f} GB")
    else:
        print("⚠️  test.parquet 파일을 찾을 수 없습니다.")

    print("\n" + "=" * 50)
    print("전처리 방법 추천")
    print("=" * 50)

    available_gb = memory.available / (1024**3)

    if available_gb < 4:
        print("🔴 메모리 부족 (4GB 미만)")
        print("권장 방법:")
        print("  1. 청크 처리 방식 사용")
        print("  2. python run_preprocessing_chunked.py")
        print("  3. 다른 프로그램 종료 후 재시도")
        return "chunked"

    elif available_gb < 8:
        print("🟡 메모리 제한적 (4-8GB)")
        print("권장 방법:")
        print("  1. 청크 처리 방식 사용 (안전)")
        print("  2. python run_preprocessing_chunked.py")
        print("  또는")
        print("  3. python run_preprocessing.py (위험할 수 있음)")
        return "chunked_recommended"

    else:
        print("🟢 메모리 충분 (8GB 이상)")
        print("권장 방법:")
        print("  1. python run_preprocessing.py (빠름)")
        print("  2. python run_preprocessing_chunked.py (안전)")
        return "normal"

def estimate_processing_time():
    """예상 처리 시간 계산"""
    print("\n" + "=" * 50)
    print("예상 처리 시간")
    print("=" * 50)

    train_path = 'data/train.parquet'
    if os.path.exists(train_path):
        import pandas as pd
        # 작은 샘플로 행 수 추정
        try:
            sample = pd.read_parquet(train_path, nrows=1000)
            file_size = os.path.getsize(train_path)
            sample_size = sample.memory_usage(deep=True).sum()
            estimated_rows = (file_size / sample_size) * 1000
            print(f"예상 데이터 행 수: {estimated_rows:,.0f}")

            # CPU 정보
            cpu_count = psutil.cpu_count()
            print(f"CPU 코어 수: {cpu_count}")

            # 처리 시간 추정 (매우 대략적)
            if estimated_rows > 10_000_000:
                print("예상 처리 시간:")
                print("  - 일반 방식: 15-30분")
                print("  - 청크 방식: 20-40분")
            else:
                print("예상 처리 시간:")
                print("  - 일반 방식: 5-15분")
                print("  - 청크 방식: 10-20분")

        except Exception as e:
            print(f"행 수 추정 실패: {e}")

def main():
    print("CTR 데이터 전처리 시스템 체크")
    recommendation = check_system_resources()
    estimate_processing_time()

    print("\n" + "=" * 50)
    print("실행 명령어")
    print("=" * 50)

    if recommendation == "chunked":
        print("🔴 메모리가 부족합니다. 반드시 청크 방식을 사용하세요:")
        print("python run_preprocessing_chunked.py")

    elif recommendation == "chunked_recommended":
        print("🟡 안전한 청크 방식을 권장합니다:")
        print("python run_preprocessing_chunked.py")
        print("\n일반 방식을 시도하려면:")
        print("python run_preprocessing.py")

    else:
        print("🟢 두 방식 모두 사용 가능합니다:")
        print("빠른 처리: python run_preprocessing.py")
        print("안전한 처리: python run_preprocessing_chunked.py")

    print("\n필요한 라이브러리 설치:")
    print("pip install pandas numpy scikit-learn tqdm psutil")

if __name__ == "__main__":
    main()