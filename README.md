# 철강 제조 공정 7종 표면 결함 다중 클래스 분류

UCI Steel Plates Faults 데이터셋을 활용하여 철판 표면 결함 유형을 자동 분류하는 머신러닝 모델을 개발했습니다. Other_Faults 클래스 제거와 Feature Engineering을 통해 F1-Macro **0.70 → 0.95** 수준으로 성능을 개선했습니다.

## 1. 문제 정의 (필수)

- **비즈니스 목표**: 철판 표면 결함을 자동 분류하여 육안 검사 의존도를 줄이고, 공정 중 조기 개입으로 불량품 누적을 방지
- **ML 목표**: 27개 공정·품질 지표를 입력으로 받아 7종(→ 모델2: 6종) 결함 유형을 다중 분류
- **성공 기준**: F1-Macro 0.90 이상

## 2. 데이터

- **출처**: [UCI Steel Plates Faults Dataset](https://archive.ics.uci.edu/dataset/198/steel+plates+faults)
- **규모**: 1,941행 × (Features 27개 + Targets 7개)
- **타깃**: Pastry, Z_Scratch, K_Scratch, Stains, Dirtiness, Bumps, Other_Faults (7종 결함)
- **결측치**: 없음

```
data/
├── steel_plates_faults_features.csv   # 입력 피처 (27개 변수)
└── steel_plates_faults_targets.csv    # 타깃 레이블 (7종 결함 One-Hot)
```

## 3. 접근 방법 (필수)

| 단계 | 핵심 결정 | 근거 |
|------|----------|------|
| EDA | 클래스 분포 확인, 상관관계 분석 | 최대 vs 최소 클래스 약 12배 불균형 확인 |
| 전처리 | TypeOfSteel 다중공선성 제거, 극단값 수동 제거 | 상관계수 -1.0 / IQR 일괄 제거 시 K_Scratch 80% 손실 |
| Feature Engineering | 기하·밝기·위치 파생변수 12개 추가 (27 → 37개) | 결함 형태·비율·상호작용 포착 |
| 오버샘플링 | SMOTE (CV 폴드 내 적용) | Data Leakage 방지 |
| 모델 1 | RF·GB·ET·XGB·LGBM·Voting·Stacking + Optuna 튜닝 | 7종 전체 분류 |
| 모델 2 | Other_Faults 제거 후 동일 파이프라인 재실행 | Bumps↔Other_Faults 분류 불가 확인 후 잡음 클래스 제거 |
| 검증 | Stratified 5-Fold CV | 클래스 비율 유지, 정보 누출 방지 |

노트북: [소스_코드_10팀_철판_결함_예측_분석_모델_개발.ipynb](소스_코드_10팀_철판_결함_예측_분석_모델_개발.ipynb)

## 4. 결과 (필수)

| 모델 | Base line F1-Macro | 모델 1 F1-Macro | 모델 2 F1-Macro | 변화 |
|------|-------------------|----------------|----------------|------|
| Random Forest | 0.7005 | 0.8144 | 0.9206 | +0.1201 |
| GradientBoosting | 0.7039 | 0.8413 | 0.9495 | +0.1456 |
| ExtraTrees | 0.7080 | 0.7986 | **0.9513** | +0.2433 |
| XGBoost | 0.7096 | 0.8148 | 0.9175 | +0.2079 |
| LightGBM | 0.6950 | 0.8197 | 0.9495 | +0.2545 |
| Voting | 0.7012 | 0.8373 | 0.9464 | +0.2452 |
| Stacking | 0.7133 | 0.8124 | 0.9216 | +0.2083 |

**최종 선택 모델**: ExtraTrees (F1-Macro **0.9513**)

**비즈니스 번역**: Other_Faults 제거 이전(모델1)에도 F1-Macro 0.84 수준을 달성했으나, PCA·t-SNE 분석 결과 Bumps와 Other_Faults의 피처 분포가 크게 겹쳐 분류 경계 설정이 불가능한 것으로 확인되었습니다. Other_Faults를 제거한 모델2에서 ExtraTrees가 F1-Macro 0.95를 기록하였으며, 이는 공정 데이터 기반의 6종 결함 자동 분류를 실용 수준으로 구현한 것입니다. 향후 Other_Faults 내 세부 결함 유형 레이블링이 이루어진다면 7종 전체 분류 성능도 대폭 향상될 것으로 기대됩니다.

## 5. 재현 방법 (필수)

```bash
git clone [레포 주소]
cd steel-fault-classification
pip install -r requirements.txt

# 데이터 확인: 아래 두 파일이 노트북과 같은 폴더에 있는지 확인
# steel_plates_faults_features.csv
# steel_plates_faults_targets.csv

# 노트북 실행
jupyter notebook 소스_코드_10팀_철판_결함_예측_분석_모델_개발.ipynb
```

> **한글 폰트**: Windows 환경은 `Malgun Gothic`이 기본 설정되어 있습니다. Linux/Mac 환경에서는 노트북 폰트 설정 셀을 `NanumBarunGothic`으로 변경하거나 `koreanize-matplotlib`을 활용하세요.

## 레포 구조

```
├── data/
│   ├── steel_plates_faults_features.csv    # 입력 피처
│   └── steel_plates_faults_targets.csv     # 타깃 레이블
├── 소스_코드_10팀_철판_결함_예측_분석_모델_개발.ipynb  # 분석 노트북
├── requirements.txt                         # 필요 라이브러리
└── README.md                                # 프로젝트 요약 (← 이 파일)
```

## 배운 것 / 다음 시도

- **데이터 품질이 모델 구조보다 중요**: Other_Faults라는 잡음 클래스 하나를 제거하는 것이 어떤 모델 튜닝보다 성능에 큰 영향을 미쳤습니다.
- **이상치 처리는 도메인 맥락이 필요**: IQR 일괄 제거 시 특정 클래스가 80% 이상 손실되는 문제를 경험하며, 통계적 기준과 도메인 지식을 함께 고려해야 함을 배웠습니다.
- **다음 시도**: Other_Faults 내 세부 유형(롤마크, 흠, 덴트 등) 레이블링 후 재분류, 실시간 추론을 위한 경량 모델 탐색
