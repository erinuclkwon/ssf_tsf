"""
Main entry point for reproducing the experiments in:

    Quan W, Gorse D (2026) "Investigating the predictive power of seismic
    statistical features using ensemble learning." PLoS ONE 21(2): e0342765.
    https://doi.org/10.1371/journal.pone.0342765

This script runs the full pipeline for a given region:
1. Data preprocessing (seismic statistical features + tsfresh features)
2. Feature selection (BorutaShap)
3. Hyperparameter tuning (GridSearchCV with PredefinedSplit)
4. Test set evaluation (AUC, MCC, SHAP)

Usage
-----
    python main.py

Notes
-----
- Raw earthquake catalogue data and pre-computed MATLAB SSF files
  should be placed in the `data/` folder.
- Results (tuning logs, SHAP plots, ROC curves) are saved to `results/`.
- The data used in the paper is available at:
  https://doi.org/10.6084/m9.figshare.31048849
"""

import pandas as pd
import numpy as np
from data_processor import DataProcessor
from select_tune import SelectTune
from utils import custom_data_split, compute_class_weights, find_path
from xgboost import XGBClassifier


# ============================================================================
# Configuration
# ============================================================================

# Region and data file names (without .csv extension)
REGION = 'Chile'
RAW_CATALOGUE = 'Chile_E'
SSF_TRAINVAL = 'chileMdtp15_trainVal_24_E_mag50'
SSF_WHOLE = 'chileMdtp15_24_E_mag50'

# Pipeline parameters
EVENT_WINDOW = 50          # number of events per sliding window
NAN_PERCENT_DROP = 15      # max NaN % before dropping a tsfresh column
THRESHOLD_MAG = 5          # magnitude threshold for binary classification
DATA_FOLDER = 'data'       # folder containing input data files

# XGBoost fixed parameters
FIXED_PARAMS = {"verbosity": 0, "n_jobs": -1}

# Hyperparameter grids
PARAM_GRID_IND = {
    'max_depth': [2, 5, 8],
    'n_estimators': list(range(100,850,100)),
    'learning_rate': [0.1, 0.01]
}

PARAM_GRID_TS = {
    'max_depth': [2, 5, 8],
    'n_estimators': list(range(10, 550, 100)),
    'learning_rate': [0.1, 0.01]
}

PARAM_GRID_MIX = {
    'max_depth': [2, 5, 8],
    'n_estimators': list(range(10, 550, 100)),
    'learning_rate': [0.1, 0.01]
}

# Fine-tuning grids - change based on your need
TUNE_GRID_IND = {
    'max_depth': [2, 5],
    'learning_rate': [0.01],
    'subsample': [0.5, 1],
    'min_child_weight': [3, 6, 8],
    'gamma': [0, 5],
    'n_estimators': list(range(400, 450, 10))
}

TUNE_GRID_TS = {
    'max_depth': [2, 5],
    'learning_rate': [0.1],
    'subsample': [0.5, 1],
    'min_child_weight': [3, 6, 8],
    'gamma': [0, 2, 5],
    'n_estimators': list(range(10, 50, 10))
}

TUNE_GRID_MIX = {
    'max_depth': [2, 5],
    'learning_rate': [0.1],
    'subsample': [0.5, 1],
    'min_child_weight': [3, 6, 8],
    'gamma': [0, 5],
    'n_estimators': list(range(10, 50, 10))
}


# ============================================================================
# Step 1: Data Preprocessing
# ============================================================================

def preprocess():
    """Run the data preprocessing pipeline."""
    print("=" * 60)
    print("Step 1: Data Preprocessing")
    print("=" * 60)

    processor = DataProcessor(
        raw_whole=RAW_CATALOGUE,
        SSF_train_val=SSF_TRAINVAL,
        SSF_whole=SSF_WHOLE,
        event_n=EVENT_WINDOW,
        percent_drop=NAN_PERCENT_DROP,
        thresh=THRESHOLD_MAG,
        save_folder=DATA_FOLDER
    )
    trainVal, whole = processor.create_all_fs_data()
    print(f"Preprocessing complete. trainVal: {trainVal.shape}, whole: {whole.shape}")
    return trainVal, whole


# ============================================================================
# Step 2: Feature Selection, Tuning, and Evaluation
# ============================================================================

def run_experiment(trainVal, whole, fs_type, param_grid, tune_grid):
    """
    Run the full experiment pipeline for a single feature set type.

    Parameters
    ----------
    trainVal : pd.DataFrame
        Training+validation data.
    whole : pd.DataFrame
        Complete dataset for final evaluation.
    fs_type : str
        'SSF', 'ts', or 'mixed'.
    param_grid : dict
        Initial hyperparameter grid.
    tune_grid : dict
        Fine-tuning hyperparameter grid.

    Returns
    -------
    dict
        Test results including AUC and MCC.
    """
    print(f"\n{'=' * 60}")
    print(f"Running experiment: {fs_type} features")
    print(f"{'=' * 60}")

    # Compute class weights from the training portion
    X_t, y_train_r, X_tv, y_tv = custom_data_split(
        trainVal, THRESHOLD_MAG, 'SSF'
    )
    class_weight = compute_class_weights(y_train_r)
    model = XGBClassifier(class_weight=class_weight, eval_metric='auc')

    # Initialise SelectTune
    st = SelectTune(
        model=model,
        parameter_grid=param_grid,
        fixed_params=FIXED_PARAMS,
        train_data=trainVal,
        threshold_mag=THRESHOLD_MAG,
        region=REGION,
        fs_type=fs_type
    )

    # Feature selection
    features = st.feature_select()
    print(f"Selected features ({len(features)}): {features}")

    # Fine-tune hyperparameters
    tuned_model, X_train, y_train = st.hyperparameter_fine_tune(param_fine=tune_grid)
    print(f"Best model: {tuned_model}")

    # Evaluate on test set
    results = st._predict(whole)
    print(f"Results: AUC={results['AUC']:.4f}, MCC={results['MCC']:.4f}")

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # --- Preprocessing ---
    # Uncomment below to run preprocessing from raw data:
    # trainVal, whole = preprocess()

    # --- Or load pre-processed data ---
    trainVal_path = find_path(DATA_FOLDER, f'{RAW_CATALOGUE}_all_fs_trainVal_{THRESHOLD_MAG}')
    whole_path = find_path(DATA_FOLDER, f'{RAW_CATALOGUE}_all_fs_whole_{THRESHOLD_MAG}')
    trainVal = pd.read_csv(trainVal_path, index_col=False).astype(float)
    whole = pd.read_csv(whole_path, index_col=False)

    print(f"trainVal shape: {trainVal.shape}")
    print(f"whole shape: {whole.shape}")

    # --- Run experiments for all three feature set types ---
    all_results = []

    # SSF only
    results_ind = run_experiment(trainVal, whole, 'SSF', PARAM_GRID_IND, TUNE_GRID_IND)
    all_results.append(results_ind)

    # tsfresh features only
    results_ts = run_experiment(trainVal, whole, 'ts', PARAM_GRID_TS, TUNE_GRID_TS)
    all_results.append(results_ts)

    # Combined (mixed) features
    results_mix = run_experiment(trainVal, whole, 'mixed', PARAM_GRID_MIX, TUNE_GRID_MIX)
    all_results.append(results_mix)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Summary of Results")
    print(f"{'=' * 60}")
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    results_df.to_csv(f'results/{REGION}_{THRESHOLD_MAG}_summary.csv', index=False)
    print(f"\nResults saved to results/{REGION}_{THRESHOLD_MAG}_summary.csv")
