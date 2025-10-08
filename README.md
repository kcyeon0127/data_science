# CTR 예측 데이터 파이프라인

## 📜 정책 및 평가 기준
- 외부 구현을 사용하면 반드시 출처를 명시합니다. 미기재 시 20점 감점됩니다.
- 평가 항목: Novelty(20점), EDA 수행(10점), Feature selection 설명(5점), Model selection 설명(5점), 랭킹 점수
- Novelty 세부 배점: 모델 변경(2점), 피처 분석 및 신규 피처(3~6점), 앙상블(2점), 하이퍼파라미터 변경(3점), WOW 요소(0~10점)

## ✅ 우리가 한 일 정리
1. **EDA**
   - 데이터 규모, 클릭률(약 2%), 범주 분포(성별, 연령대, inventory_id 등), 시퀀스 길이, 결측치 등을 분석.
   - 이상치/결측치에 대한 처리 방법을 정리.
2. **Feature Engineering**
   - 기본 수치/카테고리 피처 정리 및 결측치 보정.
   - 타깃 인코딩(`inventory_id`, `age_group`, `gender`) 적용.
   - 시퀀스 파생 변수 추가(`length`, `mean`, `std`, `last`, `recent_mean`).
   - LSTM/Transformer 임베딩을 추가 피처로 생성.
3. **Model Selection**
   - XGBoost 베이스라인 → LSTM + XGBoost 하이브리드 → Transformer + XGBoost 하이브리드 순으로 확장.
   - 각 모델 구조 선택 이유와 기대 효과를 문서화.
4. **Novelty 항목 충족**
   - 모델 변경, 신규 피처 생성, 앙상블(여러 XGBoost 설정 평균), 하이퍼파라미터 튜닝, Transformer/LSTM 임베딩 등 WOW 요소까지 도전.
5. **실험 기록/랭킹 관리**
   - 각 실험(점수, 설정)을 로그로 남김 → 재현성 확보.
   - 최종 제출 파일 및 랭킹 결과를 기록.

## 📈 주요 모델 & 실험
- `mac_xgboost_competition.py`: XGBoost 베이스라인.
- `xg_lstm.py`: LSTM + CrossNetwork로 시퀀스 임베딩을 생성하고, XGBoost에 공급.
- `xg_trans.py`: Transformer 기반 시퀀스 임베딩 + 시퀀스/타깃 파생 변수 추가 후 XGBoost 학습.

## 📁 디렉터리 구조 (핵심)
- `challenge_submission/`: 평가 정책 및 체크리스트, 모델 위치 안내.
- `add_features.py`: train/test에 시퀀스 파생·타깃 인코딩 피처를 추가해 저장하는 스크립트.
- `xg_lstm.py`: LSTM + XGBoost 하이브리드 모델.
- `xg_trans.py`: Transformer + XGBoost 하이브리드 모델.
- `mac_xgboost_competition.py`: 기본 XGBoost 베이스라인 스크립트.
- `data/`: train/test/sample_submission 파일 (경로 자동 탐지).

## 📦 제출/체크포인트
- LSTM/Transformer 모델 학습 후 체크포인트(`wd_lstm_checkpoint.pt`, `wd_transformer_checkpoint.pt`) 저장.
- 제출 파일: `submission_mac_xgboost_competition.csv`, `submission_xg_lstm.csv`, `submission_xg_trans.csv`.

## ⚙️ 최신 스크립트 및 추가 사항
- `add_features.py`: train/test에 시퀀스 기반 파생 변수(길이/평균/표준편차/최근 평균 등)와 타깃 인코딩(`inventory_id`, `age_group`, `gender`)을 추가하고 `train_enriched.parquet`, `test_enriched.parquet`로 저장합니다.
- `xg_trans.py`: 위에서 생성한 파생 피처를 포함해 Transformer + XGBoost 하이브리드 학습을 수행하며, 최근 Validation AP 약 0.082로 상승했습니다(제출 예시 `submission_xg_trans.csv`).
- `tune_xg_lstm.py`: Optuna 기반으로 LSTM + XGBoost 모델의 하이퍼파라미터(배치 크기, epoch, 학습률 등)를 탐색합니다.
- 제출 파일(예시):
  * `submission_xg_lstm.csv`: 약 0.3447 AP (현재 최고)
  * `submission_xg_trans.csv`: 약 0.3433 AP (Transformer 하이브리드)
  * `submission_xg_trans_plus_param.csv`: 약 0.2669 AP (새 파생/튜닝 실험)

## 🚀 다음 단계 아이디어
- Transformer/LSTM 임베딩을 LightGBM, CatBoost와 앙상블.
- Optuna를 적용해 더 넓은 하이퍼파라미터 탐색.
- 시간 기반 validation, Hard negative sampling 등 추가 실험.
