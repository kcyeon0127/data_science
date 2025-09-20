# 🚀 NVIDIA Merlin CTR 예측 파이프라인

NVIDIA Merlin을 사용한 고성능 CTR(Click-Through Rate) 예측 시스템입니다.
Attention 메커니즘을 포함하여 최신 딥러닝 기술을 적용했습니다.

## ✨ 주요 특징

- 🎯 **CTR 전용 최적화**: NVIDIA Merlin으로 클릭률 예측에 특화
- 🧠 **Multi-Head Attention**: 최신 어텐션 메커니즘 적용
- ⚡ **GPU 가속**: CUDA 기반 초고속 데이터 처리
- 📊 **대용량 처리**: 메모리 문제 없이 10GB+ 데이터 처리
- 🎁 **원클릭 실행**: 설치부터 제출파일 생성까지 자동화

## 🚀 빠른 시작

### 1단계: Merlin 설치
```bash
# 자동 설치 (GPU/CPU 자동 감지)
python install_merlin.py

# 또는 수동 설치
conda install -c nvidia -c rapidsai -c conda-forge cudf nvtabular
pip install merlin-models merlin-dataloader
```

### 2단계: 파이프라인 실행
```bash
# 전체 파이프라인 실행 (데이터 전처리 → 모델 훈련 → 예측 → 제출파일)
python merlin_ctr_pipeline.py
```

### 3단계: 제출
```bash
# 생성된 제출 파일 확인
ls -la submission_merlin_attention.csv
```

## 📋 실행 옵션

### 🎯 Merlin 모드 (권장)
- GPU 가속 데이터 처리
- Multi-Head Attention 모델
- 최고 성능

### 💻 베이스라인 모드
- Merlin 설치 없이 실행 가능
- Scikit-learn 기반 간단 모델
- 빠른 테스트용

## 🏗️ 아키텍처

```
📂 데이터 로드 (CuDF/GPU)
    ↓
⚡ NVTabular 전처리
    ↓
🧠 Attention 기반 딥러닝 모델
    ↓
🎯 CTR 예측
    ↓
📄 제출 파일 생성
```

### 모델 구조
```python
Input → Embedding → Multi-Head Attention → MLP → Binary Classification
```

## 📊 성능 비교

| 방식 | 처리 속도 | 메모리 사용 | 정확도 | GPU 활용 |
|------|-----------|-------------|--------|----------|
| **Merlin** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ |
| Pandas+Sklearn | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ❌ |

## 🔧 상세 설정

### GPU 메모리 설정
```python
# 메모리 증가 방식 설정 (자동)
tf.config.experimental.set_memory_growth(gpu, True)
```

### 모델 하이퍼파라미터
```python
# Attention 설정
num_heads = 8           # Multi-Head 수
key_dim = 64           # Attention 차원
dropout = 0.1          # 드롭아웃 비율

# 훈련 설정
epochs = 10            # 에포크 수
batch_size = 4096      # 배치 크기
learning_rate = 0.001  # 학습률
```

## 📁 생성되는 파일들

```
📄 submission_merlin_attention.csv  # 최종 제출 파일
📄 submission_baseline.csv          # 베이스라인 (폴백)
📂 model_checkpoints/               # 모델 체크포인트
📂 preprocessed_data/               # 전처리된 데이터
```

## 🔍 트러블슈팅

### GPU 인식 안됨
```bash
# NVIDIA 드라이버 확인
nvidia-smi

# CUDA 설치 확인
nvcc --version
```

### Merlin 설치 실패
```bash
# CPU 모드로 실행
python merlin_ctr_pipeline.py
# 베이스라인 선택: y
```

### 메모리 부족
```python
# 배치 크기 감소
batch_size = 2048  # 기본값: 4096
```

## 📈 예상 성능

### 처리 속도
- **GPU**: 10GB 데이터 → 15-30분
- **CPU**: 10GB 데이터 → 2-4시간

### 메모리 사용량
- **GPU**: VRAM 4-8GB
- **CPU**: RAM 4-6GB

### 예상 AUC
- **Merlin+Attention**: 0.75-0.80
- **베이스라인**: 0.70-0.75

## 🎯 성능 최적화 팁

### 1. 데이터 최적화
```python
# 카테고리 압축
categorical_cols >> ops.Categorify(dtype="int16")

# 연속형 정규화
continuous_cols >> ops.Normalize(out_dtype="float16")
```

### 2. 모델 최적화
```python
# Mixed Precision 훈련
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### 3. 배치 최적화
```python
# 동적 배치 크기
if gpu_memory > 8:
    batch_size = 8192
else:
    batch_size = 4096
```

## 🚀 고급 사용법

### 커스텀 Attention
```python
# 더 복잡한 Attention 구조
attention_layers = [
    MultiHeadAttention(num_heads=8, key_dim=64),
    MultiHeadAttention(num_heads=4, key_dim=128),
]
```

### 앙상블 모델
```python
# 여러 모델 조합
models = [attention_model, deepfm_model, dcn_model]
ensemble_pred = tf.reduce_mean([m.predict(x) for m in models], axis=0)
```

## 📞 지원

### 문제 신고
- GitHub Issues
- 로그 파일: `merlin_pipeline.log`

### 참고 자료
- [NVIDIA Merlin 공식 문서](https://nvidia-merlin.github.io/Merlin/)
- [Attention 메커니즘 설명](https://arxiv.org/abs/1706.03762)
- [CTR 예측 베스트 프랙티스](https://github.com/NVIDIA-Merlin/models)

---

🔥 **GPU가 있다면 Merlin을 강력 추천합니다!**
🖥️ **GPU가 없어도 베이스라인으로 실행 가능합니다!**