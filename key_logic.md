## Core Execution Flow

### 1. Training (`train_xgboost.py`)
- Load `ProjectConfig` (optionally from custom YAML/JSON).
- Decide `run_id` (timestamp by default) and create `artifacts/<run_id>/`.
- Assemble the training use case:
  - `PandasDatasetRepository` reads LOT-level splits.
  - `TargetScaler` scales tiny defect rates for stable XGBoost training.
  - `XGBoostModelTrainer` fits the model, `SklearnMetricEvaluator` scores it.
  - Writers persist model, metrics, and feature importance under the run directory.
- Save `latest_run.txt` so 후속 단계가 동일한 Run을 참조.

### 2. SHAP Analysis (`analyze_shap_interaction.py`)
- Load the same `ProjectConfig` and resolve the run (`RUN_ID` or `latest_run.txt`).
- Build `AnalyzeShapInteractionsUseCase` with identical repositories/scaler.
- Load the trained model and generate:
  - SHAP summary plot/table
  - 주요 feature dependence plots, interaction heatmap
  - LOT별 SHAP bar plot 및 선택된 3D surface plot
- 모든 산출물은 `artifacts/<run_id>/plots|reports/`에 저장.

### 3. Temperature Optimization (`optimize_temp_combination.py`)
- 다시 동일한 Run을 불러와 학습된 모델을 로드.
- `OptunaTemperatureOptimizer`로 지정된 LOT/feature 범위를 탐색.
- 각 LOT에 대해 baseline 예측 대비 개선된 온도 조합을 CSV로 출력.
- 결과는 `artifacts/<run_id>/optimization/` 에 기록되며, 이후 분석 단계에서 활용.

> 이 구조 덕분에 **단 한 번의 학습 결과를 기준으로 SHAP 분석과 최적화가 서로 일관된 Run ID를 공유**하며, 실행 스크립트 상단 변수만 수정해 전체 파이프라인을 제어할 수 있습니다.

