# CTR 예측 데이터 전처리 가이드

## 📋 개요
CTR(Click-Through Rate) 예측을 위한 데이터 전처리 파이프라인입니다. EDA 분석 결과를 바탕으로 최적화된 전처리 과정을 제공합니다.

## 📁 파일 구조
```
class_datas/
├── data/
│   ├── train.parquet          # 훈련 데이터
│   ├── test.parquet           # 테스트 데이터
│   └── sample_submission.csv  # 제출 샘플
├── out_eda/                   # EDA 결과 파일들
├── ctr_preprocessing.py       # 메인 전처리 클래스
├── preprocessing_utils.py     # 고급 유틸리티 함수
├── run_preprocessing.py       # 실행 스크립트
└── processed_data/            # 전처리 결과 (생성됨)
```

## 🚀 빠른 시작

### 1. 필요한 라이브러리 설치
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy tqdm
```

### 2. 전체 전처리 파이프라인 실행
```bash
cd /Users/gimchaeyeon/Documents/2025/class_datas
python run_preprocessing.py
```

### 3. 개별 전처리 사용법
```python
from ctr_preprocessing import CTRDataPreprocessor

# 전처리기 초기화
preprocessor = CTRDataPreprocessor()

# 데이터 로드
preprocessor.load_data('data/train.parquet', 'data/test.parquet')

# 훈련 데이터 전처리
train_processed = preprocessor.preprocess_pipeline(
    preprocessor.train_df,
    is_training=True,
    target_col='clicked'
)

# 테스트 데이터 전처리
test_processed = preprocessor.preprocess_pipeline(
    preprocessor.test_df,
    is_training=False
)
```

## 📊 전처리 과정 상세

### 1. 기본 전처리 (`ctr_preprocessing.py`)

#### 결측값 처리
- **gender, age_group**: 0.16% 결측률 → 최빈값으로 대체
- **수치형 변수**: 중앙값으로 대체

#### 피처 엔지니어링
- **시간 피처**: 순환 인코딩 (sin, cos 변환)
  ```python
  hour_sin = sin(2π × hour / 24)
  hour_cos = cos(2π × hour / 24)
  ```
- **시퀀스 피처**: 길이 기반 비닝 (EDA 분위수 활용)
- **타겟 인코딩**: 고카디널리티 변수(inventory_id) 처리
- **인터랙션 피처**: gender × age_group

#### 스케일링 및 인코딩
- **수치형**: RobustScaler (이상치에 강함)
- **범주형**: LabelEncoder
- **이상치**: IQR 방법으로 캐핑

### 2. 고급 전처리 (`preprocessing_utils.py`)

#### 피처 중요도 분석
```python
from preprocessing_utils import CTRPreprocessingUtils

utils = CTRPreprocessingUtils()
importance = utils.analyze_feature_importance(X, y, method='mutual_info')
```

#### 데이터 드리프트 감지
```python
drift_results = utils.detect_data_drift(train_df, test_df, features)
```

#### 고급 피처 생성
- **통계적 피처**: 평균, 표준편차, 왜도, 첨도
- **인터랙션 피처**: 곱셈, 덧셈, 비율
- **시퀀스 고급 처리**: 트렌드, 반복률, 최빈값

## 📈 실행 결과

### 생성되는 파일들
```
processed_data/
├── X_train.parquet      # 훈련 피처 (80%)
├── X_val.parquet        # 검증 피처 (20%)
├── y_train.parquet      # 훈련 타겟
├── y_val.parquet        # 검증 타겟
├── X_test.parquet       # 테스트 피처
└── feature_info.json    # 피처 메타데이터
```

### 예상 처리 시간
- 훈련 데이터 (1천만 행): 약 5-10분
- 메모리 사용량: 약 4-8GB

## 🔧 고급 사용법

### 1. 맞춤형 전처리 파이프라인
```python
# 개별 단계별 실행
preprocessor = CTRDataPreprocessor()
preprocessor.load_data('data/train.parquet')

# 1단계: 결측값 처리
df = preprocessor.handle_missing_values(train_df)

# 2단계: 피처 엔지니어링
df = preprocessor.engineer_features(df)

# 3단계: 타겟 인코딩
df = preprocessor.create_target_encoding(df, 'clicked', is_training=True)

# 4단계: 이상치 처리
df = preprocessor.handle_outliers(df)

# 5단계: 인코딩 및 스케일링
df = preprocessor.encode_categorical_features(df, is_training=True)
df = preprocessor.scale_numeric_features(df, is_training=True)
```

### 2. 피처 선택 및 차원 축소
```python
from preprocessing_utils import CTRPreprocessingUtils

utils = CTRPreprocessingUtils()

# PCA 적용
X_train_pca, X_test_pca, pca = utils.apply_pca_to_features(
    X_train, X_test,
    n_components=0.95,
    feature_prefix='feat_'
)

# 피처 중요도 기반 선택
top_features = utils.analyze_feature_importance(
    X_train, y_train,
    method='mutual_info',
    k=100
)
```

### 3. 전처리 결과 시각화
```python
# 전처리 전후 비교
utils.plot_preprocessing_results(original_df, processed_df)

# 종합 리포트
report = utils.generate_preprocessing_report(original_df, processed_df)
```

## 🐛 문제 해결

### 일반적인 오류들

#### 1. 메모리 부족 오류
```python
# 청크 단위로 처리
chunk_size = 100000
for chunk in pd.read_parquet('data/train.parquet', chunksize=chunk_size):
    processed_chunk = preprocessor.preprocess_pipeline(chunk)
```

#### 2. 파일 경로 오류
```bash
# 현재 디렉토리 확인
pwd
ls data/

# 절대 경로 사용
python -c "import os; print(os.path.abspath('data/train.parquet'))"
```

#### 3. 라이브러리 버전 호환성
```bash
# 권장 버전
pip install pandas==1.5.3 scikit-learn==1.3.0 numpy==1.24.3
```

### 성능 최적화 팁

#### 1. 메모리 사용량 줄이기
```python
# 데이터 타입 최적화
def optimize_dtypes(df):
    for col in df.select_dtypes(include=['int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df
```

#### 2. 병렬 처리
```python
# multiprocessing 사용
from multiprocessing import Pool

def process_chunk(chunk):
    return preprocessor.preprocess_pipeline(chunk)

with Pool(processes=4) as pool:
    results = pool.map(process_chunk, chunks)
```

## 📚 참고 자료

### EDA 기반 전처리 결정사항
- **CTR**: 1.9% (불균형 데이터)
- **시퀀스 길이**: 중간값 439, 95% 분위수 1345
- **결측률**: gender/age_group 0.16%
- **수치형 피처**: 48개 (feat_a~e, history_a)

### 추가 개선 아이디어
1. **앙상블 타겟 인코딩**: 여러 폴드의 평균 사용
2. **지연된 피처**: 이전 시점의 CTR 정보 활용
3. **임베딩**: 범주형 변수의 dense representation
4. **GBM 기반 피처 선택**: 트리 모델의 피처 중요도 활용

## 📞 지원

문제가 발생하거나 개선 사항이 있다면:
1. 로그 파일 확인: `preprocessing.log`
2. 메모리 사용량 모니터링: `htop` 또는 `Activity Monitor`
3. 전처리 리포트 확인: `processed_data/feature_info.json`

---
**Created by**: CTR Preprocessing Pipeline v1.0
**Last Updated**: 2025-09-17