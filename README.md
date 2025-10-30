# Temp Range Optimizer

A clean-architecture implementation for training an XGBoost regression model on LOT-level heat-treatment data, interpreting SHAP interactions, and searching optimal temperature feature ranges that minimise the predicted defect rate.

## Environment Setup

1. Create a virtual environment (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install Python dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Ensure the dataset CSV files exist under `data/` as described in `requirement.md`.

## Project Layout

- `src/temp_range_optimizer/` – core package organised into domain, application (use cases), infrastructure, and common utilities following Clean Architecture and SOLID principles.
- `train_xgboost.py` – trains the XGBoost model and persists metrics / feature importances.
- `analyze_shap_interaction.py` – produces SHAP summaries, dependence plots, interaction heatmaps, and surface plots.
- `optimize_temp_combination.py` – runs Optuna-based searches for temperature feature combinations per LOT.
- `evaluate_by_lot.py` – reports LOT-level prediction quality and optional SHAP attribution summaries.
- `artifacts/` – automatically created output directory containing models, plots, reports, and optimisation results.

## Usage

Activate your virtual environment (if applicable) before running the commands below. Scripts automatically add `src/` to `PYTHONPATH` and configure Matplotlib for headless execution.

### 1. Train the Model
```bash
python train_xgboost.py --model-name xgb_baseline
```
Outputs: `artifacts/models/<model>.joblib`, metrics JSON, and feature importance CSV.

### 2. SHAP Interaction Analysis
```bash
python analyze_shap_interaction.py --model-name xgb_baseline
```
Generates SHAP summary/interaction plots under `artifacts/plots/shap/` and summary CSV artefacts.

### 3. Temperature Optimisation
```bash
python optimize_temp_combination.py --model-name xgb_baseline --split val --max-lots 5
```
Writes recommended settings per LOT to `artifacts/optimization/`.

### 4. LOT-level Evaluation
```bash
python evaluate_by_lot.py --model-name xgb_baseline --split val
```
Produces LOT comparison plots, error tables, metrics JSON, and optional SHAP summaries.

### 5. Custom Configuration
Each CLI accepts `--config path/to/config.yaml` to override defaults such as data paths, model hyperparameters, SHAP behaviour, or optimisation settings (see `temp_range_optimizer/common/config.py` for schema details).

## Notes

- Matplotlib fonts are configured to support Hangul labels; adjust `common/environment.py` if a different font is preferred.
- Optuna logging is verbose by default; set `OPTUNA_LOGGING_LEVEL=WARNING` to reduce console output.
- Ensure a trained model exists before running SHAP, optimisation, or evaluation scripts.

