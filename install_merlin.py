#!/usr/bin/env python3
"""
NVIDIA Merlin 설치 스크립트
"""

import subprocess
import sys
import os

def check_gpu():
    """GPU 확인"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU 감지됨")
            return True
        else:
            print("⚠️ NVIDIA GPU 없음 - CPU 버전으로 설치")
            return False
    except FileNotFoundError:
        print("⚠️ nvidia-smi 없음 - CPU 버전으로 설치")
        return False

def install_merlin():
    """Merlin 설치"""

    print("🔧 NVIDIA Merlin 설치 시작...")

    has_gpu = check_gpu()

    # 기본 라이브러리들
    basic_packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'tensorflow>=2.9',
        'tqdm'
    ]

    print("📦 기본 패키지 설치...")
    for package in basic_packages:
        print(f"설치 중: {package}")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', package],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️ {package} 설치 실패: {result.stderr}")

    if has_gpu:
        print("\n🚀 GPU 버전 Merlin 설치...")

        # RAPIDS 설치 (CuDF 등)
        conda_commands = [
            'conda install -y -c nvidia -c rapidsai -c conda-forge cudf',
            'conda install -y -c nvidia -c conda-forge nvtabular'
        ]

        pip_commands = [
            'pip install merlin-models',
            'pip install merlin-dataloader',
            'pip install merlin-systems'
        ]

        print("🔧 Conda 패키지 설치 중...")
        for cmd in conda_commands:
            print(f"실행: {cmd}")
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️ 명령어 실패: {result.stderr}")

        print("🔧 Pip 패키지 설치 중...")
        for cmd in pip_commands:
            print(f"실행: {cmd}")
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️ 명령어 실패: {result.stderr}")

    else:
        print("\n💻 CPU 버전으로 설치...")
        cpu_packages = [
            'merlin-models',
            'merlin-dataloader',
            'nvtabular-cpu'
        ]

        for package in cpu_packages:
            print(f"설치 중: {package}")
            result = subprocess.run([sys.executable, '-m', 'pip', 'install', package],
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️ {package} 설치 실패 (정상일 수 있음)")

def test_installation():
    """설치 테스트"""
    print("\n🧪 설치 테스트...")

    test_imports = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', None),
        ('tensorflow', 'tf'),
    ]

    advanced_imports = [
        ('cudf', None),
        ('nvtabular', 'nvt'),
        ('merlin.models.tf', 'mm'),
        ('merlin.io', None)
    ]

    # 기본 패키지 테스트
    for module, alias in test_imports:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")

    # Merlin 패키지 테스트
    merlin_available = True
    for module, alias in advanced_imports:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✅ {module}")
        except ImportError as e:
            print(f"⚠️ {module}: {e}")
            merlin_available = False

    return merlin_available

def main():
    print("🚀 NVIDIA Merlin 설치 도구")
    print("=" * 50)

    # 현재 환경 확인
    print(f"🐍 Python: {sys.version}")
    print(f"📂 현재 디렉토리: {os.getcwd()}")

    # 설치 시작
    install_merlin()

    # 테스트
    merlin_ok = test_installation()

    print("\n" + "=" * 50)
    if merlin_ok:
        print("🎉 Merlin 설치 완료!")
        print("💡 다음 명령어로 실행하세요:")
        print("   python merlin_ctr_pipeline.py")
    else:
        print("⚠️ Merlin 설치 불완전")
        print("💡 베이스라인 모드로 실행 가능:")
        print("   python merlin_ctr_pipeline.py")
        print("   (베이스라인 선택 시 'y' 입력)")

if __name__ == "__main__":
    main()