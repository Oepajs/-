# Architecture Style Classification Project
### 3-Class Automated Classification: Hanok vs. Modern vs. Gothic
**Course:** 인공지능개론 (Introduction to Artificial Intelligence)  
**Professor:** 정계동 
**Submission Date:** June 4, 2026  

---

## Personal Information & Course Details
* **이름 (Name):** Çimen Yakup Emre
* **학번 (Student ID):** 2555040
* **학과 (Department):** 빅데이터과 (Department of Big Data)
* **강의 제목 (Course Title):** 인공지능개론 (Introduction to Artificial Intelligence)

---

## I. 프로젝트 주제 (Project Topic)
건축 양식 자동 분류 및 빅데이터 분석 프로젝트  
*(Automated Architectural Style Classification Big Data Analysis Project)*

---

## II. 프로젝트 목적 (Project Objectives)
본 프로젝트는 Orange3 기반의 시각적 데이터 분석과 독립형 파이썬(Python) 자동화 스크립트 파이프라인을 상호 연동하여, 단순한 객체 인식을 넘어 건축물이 가진 고유한 기하학적·형태학적 시각 패턴을 심층 분석합니다. 약 100장의 고해상도 건축 이미지 데이터를 활용해 한국의 전통 **한옥(Hanok)**, 미니멀한 기하학적 구조를 가진 **현대식(Modern)**, 그리고 유럽의 **고딕(Gothic)** 건축 양식의 특징 공간(Feature Space)을 모델들이 스스로 학습하게 하고, 이를 정밀하게 판별·분류할 수 있는 고성능 인공지능 분류기를 구축하는 것을 목적으로 합니다.

---

## III. 처리 순서 및 파이프라인 (Processing Steps)
본 프로젝트는 설계 명세 및 수립된 파이썬 자동화 파이프라인(`architecture_classifier.py`)에 따라 총 5개의 핵심 단계를 거쳐 동적으로 안전하게 수행됩니다.

* **3.1 데이터 수집 (Data Collection):** 전통 한옥(Hanok), 현대식 건축(Modern), 고딕 양식 건축(Gothic) 이미지를 각 카테고리별로 고해상도로 확보하여 총 100여 장 규모의 이미지 데이터셋(`arch/` 폴더)을 정돈하였습니다.
* **3.2 동적 디렉터리 스캔 및 이미지 전처리 (Dynamic Scan & OpenCV Preprocessing):** * 스크립트 실행 시 현재 경로 내의 한글/영문 서브 디렉터리명을 실시간으로 추적하여 한옥, 현대, 고딕 폴더를 동적으로 자동 매칭(`find_actual_folders`)합니다.
  * 데이터 일관성을 위해 공식 OpenCV(cv2) 라이브러리를 연동하여 다중 확장자 파일들을 정밀 인덱싱하고, BGR 채널을 RGB 채널로 정상 복원하였습니다.
  * 선형 보간법(`cv2.INTER_LINEAR`)을 적용해 모든 그래픽 규격을 128x128 픽셀 해상도로 통일한 후, `float32` 타입 변환 및 [0, 1] 범위의 픽셀 스케일링(Normalization)을 거쳐 1차원 벡터로 평탄화(Flattening) 처리를 완료했습니다.
* **3.3 다중 모델 설계 및 하이퍼파라미터 구성 (Model Selection):** 추출된 이미지 특징 임베딩의 수치 특성을 정밀 학습하기 위해 서로 다른 수학적 패러다임을 가진 3가지 핵심 머신러닝 분류 알고리즘을 빌드하였습니다.
  1. **Neural Network (MLP):** 500개의 뉴런으로 구성된 은닉층과 `max_iter=500` 옵션을 통해 고차원 비선형 조합을 안정적으로 탐색합니다.
  2. **SVM (Support Vector Machine):** 고차원 공간 상에서 최적의 방사형 결정 경계선을 도출하기 위해 `rbf` 커널 함수를 채택하고 확률 추정 기능을 활성화했습니다.
  3. **Random Forest:** 100개의 독립적인 결정 트리를 앙상블로 조합하고 병렬 연산(`n_jobs=-1`)을 적용해 특징 중요도를 분석합니다.
* **3.4 엄격한 모델 평가 및 교차 검증 (Model Evaluation & 5-Fold Stratified CV):** 데이터의 무작위 추출 편향 및 테스트 클래스 비율 불균형으로 인한 데이터 유출(Data Leakage)을 완벽하게 차단하기 위해 **5-Fold Stratified Cross-Validation (층화 교차 검증)** 전략을 엄격히 구현했습니다. 표준화 스케일러(`StandardScaler`)와의 파이프라인 조합을 통해 공정한 환경에서 모델별 Accuracy, Macro F1-Score, One-vs-Rest(OvR) 기준 Multi-Class ROC Curves 및 AUC 성능을 독립 산출하였습니다.
* **3.5 자동화 결과 플로팅 및 예측 레포팅 (Automated Plotting & Prediction Report):** 검증이 완료되면 `matplotlib`과 `seaborn` 라이브러리를 통해 시각화 자료들을 300 DPI 고해상도로 자동 저장하며, 각 양식별 Precision, Recall 지표가 정밀하게 기록된 종합 분석 보고서 파일을 파일 시스템에 자동으로 출력합니다.

---

## IV. 사용 기술 및 환경 (Technologies Used)
* **프로그램 및 라이브러리:** Orange3 Data Mining Tool, Python 3.x, OpenCV (cv2), scikit-learn, PIL, Matplotlib, Seaborn
* **형상 관리:** GitHub (코드, 데이터셋 및 최종 분석 결과서 통합 배포 관리)

---

## V. 핵심 실행 결과 요약 (Key Evaluation Results)

교차 검증 데이터 유출(Data Leakage)이 없는 엄격한 검증 상태에서 도출된 핵심 성능 지표는 다음과 같습니다.

### 1. 분류기별 성능 비교 (Test & Score)
* **Neural Network:** Classification Accuracy (**CA: 0.970**) / **AUC: 0.998** —— **최우수 모델**
* **Support Vector Machine (SVM):** Classification Accuracy (**CA: 0.960**) / **AUC: 0.999**
* **Random Forest:** Classification Accuracy (**CA: 0.910**) / **AUC: 0.990**

### 2. 오분류 및 잠재 구조 진단 결과
* **혼동 행렬(Confusion Matrix) 및 MDS 분석:** **Gothic** 양식의 경우 특징 공간상에서 매우 조밀하고 독자적인 클러스터를 형성하여 오분류율이 가장 낮았으나(**평균 오류율 0.025**), **Modern** 양식은 미니멀한 기하학적 공통성으로 인해 일부 **Hanok** 양식과 경계면이 인접해 있어 상대적으로 높은 모호성(**평균 오류율 0.082**)을 보였습니다. (일원 분산 분석 결과: $ANOVA = 2.655$, $p = 0.075$)
* **설명 가능한 AI (SHAP 기여도):** `Modern` 클래스 분류 시, 특징 특성 $n1201$의 높은 값은 약 **+600**의 긍정적인 기여도를 주도한 반면, 특징 $n538$의 존재는 약 **-400**의 페널티를 부여하는 결정적 지표임을 확인하였습니다.

---

## 리포지토리 디렉터리 구조 (Directory Structure)
GitHub 저장소는 아래와 같이 깔끔하게 정돈되어 제출됩니다.

```text
인공지능개론/
├── arch/                    <-- 3.1 수집 데이터셋 폴더
│   ├── 한옥(Hanok)/         
│   ├── 현대(Modern)/         
│   └── Gothic/               
├── 기말 과제.ows             <-- 설계 완료된 Orange3 워크플로우 파일
├── test/                    <-- 3.5 모델 예측을 위한 독립형 테스트 이미지 폴더
├── architecture_classifier.py <-- 교차 검증 및 플로팅 자동화 파이썬 스크립트
├── requirements.txt           <-- 설치 필요 외부 패키지 명세서
├── 다중 분류기 벤치마킹 및 건축 양식의 특징 공간 분석.docx <-- 최종 기말 보고서 파일
├── confusion_matrices.png     <-- 자동 생성된 혼동 행렬 시각화 그래프
├── roc_curves.png             <-- 자동 생성된 OvR ROC-AUC 성능 평가 곡선
├── evaluation_report.txt      <-- 자동 생성된 최종 텍스트 스코어 레포트
└── README.md                  <-- 현재 메인 안내 파일 (본 문서)
└── README.txt                 <-- GITHUB 없이 열기 가능한 txt 안내 파일
