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

import os
os.environ['NUMBA_DISABLE_CUDA'] = '1'

import pandas as pd
import numpy as np
from data_processor import DataProcessor
from select_tune import SelectTune
from utils import custom_data_split, compute_class_weights, find_path
from xgboost import XGBClassifier


REGION = 'Chile'
RAW_CATALOGUE = 'Chile_E'
SSF_TRAINVAL = 'chileMdtp15_trainVal_24_E_mag50'
SSF_WHOLE = 'chileMdtp15_24_E_mag50'

EVENT_WINDOW = 50          
NAN_PERCENT_DROP = 15      # max NaN % before dropping a tsfresh column
THRESHOLD_MAG = 5          
DATA_FOLDER = 'data'       

FIXED_PARAMS = {"verbosity": 0, "n_jobs": -1}

PARAM_GRID_IND = {
    'max_depth': [2, 5, 8],
    'n_estimators': list(range(100,850,100)),
    'learning_rate': [0.1, 0.01]
}

PARAM_GRID_TS = {
    'max_depth': [2, 5, 8],
    'n_estimators': list(range(10, 550, 50)),
    'learning_rate': [0.1, 0.01]
}

PARAM_GRID_MIX = {
    'max_depth': [2, 5, 8],
    'n_estimators': list(range(10, 550, 50)),
    'learning_rate': [0.1, 0.01]
}

# Fine-tuning grids - change based on your need or prior coarser search
TUNE_GRID_IND = {
    'max_depth': [2, 5, 8],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.5, 1],
    'min_child_weight': [3, 6, 8],
    'gamma': [0, 2, 5],
    'n_estimators': list(range(100, 450, 10))
}

TUNE_GRID_TS = {
    'max_depth': [2, 5, 8],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.5, 1],
    'min_child_weight': [3, 6, 8],
    'gamma': [0, 2, 5],
    'n_estimators': list(range(10, 300, 20))
}

TUNE_GRID_MIX = {
    'max_depth': [2, 5, 8],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.5, 1],
    'min_child_weight': [3, 6, 8],
    'gamma': [0, 2, 5],
    'n_estimators': list(range(10, 300, 20))
}


def preprocess():
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
    return trainVal, whole

def run_experiment(trainVal, whole, fs_type, param_grid, tune_grid):
    """
    Run the full experiment pipeline for a single feature set type.
    """

    # Compute class weights from the training portion
    X_t, y_train_r, X_tv, y_tv = custom_data_split(
        trainVal, THRESHOLD_MAG, 'SSF'
    )
    class_weight = compute_class_weights(y_train_r)
    model = XGBClassifier(class_weight=class_weight, eval_metric='auc')

    st = SelectTune(
        model=model,
        parameter_grid=param_grid,
        fixed_params=FIXED_PARAMS,
        train_data=trainVal,
        threshold_mag=THRESHOLD_MAG,
        region=REGION,
        fs_type=fs_type
    )

    features = st.feature_select()

    tuned_model, X_train, y_train = st.hyperparameter_fine_tune(param_fine=tune_grid)
    print(f"Best model: {tuned_model}")

    results = st._predict(whole)
    print(f"Results: AUC={results['AUC']:.4f}, MCC={results['MCC']:.4f}")

    return results


if __name__ == '__main__':
    trainVal, whole = preprocess()
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

    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    results_df.to_csv(f'results/{REGION}_{THRESHOLD_MAG}_summary.csv', index=False)
