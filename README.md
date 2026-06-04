# Architecture Style Classification Project
### 3-Class Automated Classification: Hanok vs. Modern vs. Gothic
**Course:** 인공지능개론 (Introduction to Artificial Intelligence)  
**Professor:** [Instructor Name]  
**Submission Date:** May 28, 2026  

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
본 프로젝트는 Orange Data Mining의 'Image Analytics' 및 'Machine Learning' 위젯과 파이썬(Python) 연동을 활용하여, 단순한 사물 인식을 넘어 건축물이 가진 고유한 기하학적 형태학적 시각 패턴을 깊이 있게 분석합니다. 약 100장의 고해상도 건축 이미지 데이터를 사용해 한국의 전통 **한옥(Hanok)**, contemporary **현대식(Modern)**, 그리고 유럽의 **고딕(Gothic)** 건축 양식의 특징을 스스로 학습시키고 정밀하게 판별·분류할 수 있는 고성능 인공지능 파이프라인을 구축하는 것을 목적으로 합니다.

---

## III. 처리 순서 및 파이프라인 (Processing Steps)
본 프로젝트는 설계 명세에 따라 총 5개의 핵심 단계를 거쳐 순차적으로 수행되었습니다.

* **3.1 데이터 수집 (Data Collection):** 전통 한옥(Hanok), 현대식 건축(Modern), 고딕 양식 건축(Gothic) 이미지를 각 카테고리별로 고해상도로 확보하여 총 100여 장 규모의 데이터셋(`arch/` 폴더)을 구축하였습니다.
* **3.2 데이터 전처리 (Data Preprocessing):** PIL 및 OpenCV 라이브러리를 활용하여 다양한 크기와 포맷의 이미지를 데이터 일관성을 위해 128x128 차원으로 균일하게 정규화(Pixel Scaling [0, 1]) 조정하였습니다.
* **3.3 모델 선택 (Model Selection):** 이미지 특징 임베딩 처리에 적합한 3가지 핵심 백엔드 분류 알고리즘을 도입해 상호 비교군을 구성했습니다.
  1. **Neural Network:** 은닉층 뉴런 500개 구성을 통해 정밀한 비선형 조합 탐색
  2. **Random Forest:** 100개의 결정 트리를 조합해 텍스처 요소 중요도 연산
  3. **SVM (Support Vector Machine):** 고차원 공간 상의 최적 선형/방사형 결정 경계선 도출
* **3.4 모델 평가 (Model Evaluation):** 데이터 무작위 편향을 완벽히 방지하기 위해 **5-Fold Stratified Cross-Validation (층화 교차 검증)** 알고리즘을 설계하여 Accuracy 및 F1-Score를 산출하고 혼동 행렬(Confusion Matrix)과 Multi-Class ROC Curves를 도출했습니다.
* **3.5 모델 예측 (Model Prediction):** 최종 가중치가 학습된 모형을 활용해 새롭게 유입되는 외부 테스트 이미지의 건축 스타일을 자동으로 신속·정확하게 추론 분류합니다.

---

## IV. 사용 기술 및 환경 (Technologies Used)
* **프로그램 및 라이브러리:** Orange3 Data Mining Tool, Python 3.x, OpenCV, scikit-learn, PIL, Matplotlib, Seaborn
* **형상 관리:** GitHub (코드 및 보고서 배포 관리)

---

## V. 실행 결과물 및 시각화 자료 (Evaluation Outputs)
스크립트 실행 시 교차 검증 데이터 유출(Data Leakage)이 없는 엄격한 검증 상태에서 다음 파일들이 자동으로 생성됩니다:

1. **`confusion_matrices.png`**: 각 모델이 어떤 건축 양식으로 오인했는지 보여주는 3x3 매트릭스
2. **`roc_curves.png`**: One-vs-Rest 기반 다중 클래스 분류 판별력 곡선 (AUC 점수 포함)
3. **`performance_comparison.png`**: 세 모델의 주요 지표(정확도, F1, AUC)를 직관적으로 비교한 바 차트
4. **`evaluation_report.txt`**: 각 양식별 정밀도(Precision), 재현율(Recall) 수치가 기록된 종합 평가 텍스트 파일

---

## 리포지토리 디렉터리 구조 (Directory Structure)
GitHub 저장소는 아래와 같이 깔끔하게 정돈되어 제출됩니다.

```text
인공지능개론/
├── arch/                     <-- 3.1 수집 데이터셋 폴더
│   ├── 한옥(Hanok)/          
│   ├── 현대(Modern)/          
│   └── Gothic/               
├── 기말 과제.ows             <-- 설계 완료된 Orange3 워크플로우 파일
├── test/                     <-- 3.5 모델 예측을 위한 독립형 테스트 이미지 폴더
├── architecture_classifier.py <-- 교차 검증 및 플로팅 자동화 파이썬 스크립트
├── requirements.txt           <-- 설치 필요 외부 패키지 명세서
├── confusion_matrices.png     <-- [출력 자료] 혼동 행렬 시각화 그래프
├── roc_curves.png             <-- [출력 자료] OvR ROC-AUC 성능 평가 곡선
├── evaluation_report.txt      <-- [출력 자료] 최종 텍스트 스코어 레포트
└── README.md                  <-- 현재 메인 안내 파일 (본 문서)
└── README.txt                 <-- GITHUB 없이 열기 가능한 txt 안내 파일   
