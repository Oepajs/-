===========================================================================
                  ARCHITECTURE CLASSIFICATION PROJECT REPORT
               Course: Introduction to Artificial Intelligence
===========================================================================

[1. Personal Information & Course Details]
---------------------------------------------------------------------------
* Name: Cimen Yakup Emre
* Student ID: 2555040
* Department: Department of Big Data (빅데이터과)
* Course Title: Introduction to Artificial Intelligence (인공지능개론)
* Project Submission Date: May 28, 2026
* Official GitHub Repository URL:
  https://github.com/Oepajs/Introduction-of-Artificial-Intelligence-Final-Report


[2. Project Topic & Objectives]
---------------------------------------------------------------------------
* Topic: 
  Automated Architectural Style Classification & Big Data Analysis Project
  (건축 양식 자동 분류 및 빅데이터 분석 프로젝트)

* Objectives:
  This project integrates Orange3-based visual data mining pipelines with 
  an independent Python automation framework (architecture_classifier.py). 
  By looking beyond superficial object boundaries, the model learns the 
  distinct geometric, morphological, and structural visual signatures 
  of three classic architectural styles: Traditional Korean Hanok, Minimalist 
  Modern, and European Gothic. Over 100 high-resolution images are processed 
  to build a highly standardized classification model and benchmark multiple 
  machine learning algorithms under zero-data-leakage constraints.


[3. Pipeline & Processing Steps]
---------------------------------------------------------------------------
3.1. Data Collection:
     High-resolution samples for Hanok, Modern, and Gothic architecture are 
     curated and organized within the target data structure ('arch/' folder).

3.2. Dynamic Scan & Preprocessing (Pure OpenCV Engine):
     The python framework automatically scans path branches using the 
     'find_actual_folders' mechanism to locate subdirectories regardless of 
     Korean or English labeling. Multi-extension images are processed via 
     OpenCV (cv2), resized to 128x128 pixels using linear interpolation, 
     normalized into [0, 1] floating scales, and flattened into feature arrays.

3.3. Multi-Classifier Architecture:
     The system trains three diverse mathematical modeling environments:
     - Neural Network: MLPClassifier with 500 hidden layer units (max_iter=500).
     - SVM: RBF Kernel Support Vector Machine with active probability estimation.
     - Random Forest: Multi-threaded tree ensemble (n_estimators=100, n_jobs=-1).

3.4. Rigorous 5-Fold Stratified Cross-Validation:
     To block out artificial data leakage or distribution bias, 5-Fold 
     Stratified CV is enforced. Model evaluation yields precise Accuracy, 
     Macro F1, and One-vs-Rest (OvR) ROC-AUC performance matrices.

3.5. Automated Report Generation:
     Upon validation, matplotlib/seaborn write high-resolution visual plots 
     and comprehensive metrics reports directly to the workspace directory.


[4. Core Benchmark Results Summary]
---------------------------------------------------------------------------
4.1. Cross-Classifier Performance (Test & Score Metrics):
     - Neural Network: Accuracy (CA): 0.9700 | Mean ROC-AUC: 0.9980 (Best)
     - Support Vector Machine: Accuracy (CA): 0.9600 | Mean ROC-AUC: 0.9990
     - Random Forest Ensemble: Accuracy (CA): 0.9100 | Mean ROC-AUC: 0.9900

4.2. Diagnostic Error & Latent Structure Analysis:
     - Confusion Matrices show Gothic styles are heavily isolated with 
       an incredibly low average error rate (0.025), whereas Modern 
       structures exhibit minor confluence artifacts when bounded near Hanok.
     - One-way ANOVA testing yields an F-statistic of 2.655 (p = 0.075), 
       confirming variations across classification error profiles.
     - Explainable AI (SHAP Framework) highlights feature n1201 as a major 
       positive driver (+600 impact) for Modern classification scoring.


[5. Project Directory Structure]
---------------------------------------------------------------------------
인공지능개론/
├── arch/                           <-- Core image dataset folder
│   ├── 한옥(Hanok)/         
│   ├── 현대(Modern)/         
│   └── Gothic/               
├── 기말 과제.ows                  <-- Orange3 Visual Workflow Design File
├── test/                           <-- Test folder containing prediction samples
├── architecture_classifier.py      <-- Automation, cross-validation & plot script
├── requirements.txt                <-- Required Python packages specification
├── 다중 분류기 벤치마킹 및 건축 양식의 특징 공간 분석.docx <-- Final Term Paper Doc
├── confusion_matrices.png          <-- [Auto-Output] Generated Confusion Matrix
├── roc_curves.png                  <-- [Auto-Output] Generated Multi-Class ROC Curves
├── evaluation_report.txt           <-- [Auto-Output] Detailed metrics profile text
├── README.md                       <-- Main Markdown documentation for GitHub
└── README.txt                      <-- Current Text orientation document (This File)

===========================================================================
For detailed analytics, code implementations, or execution steps, please 
refer to the official GitHub link or run 'python architecture_classifier.py'.
===========================================================================