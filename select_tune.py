"""
Feature selection and hyperparameter tuning pipeline.

This module implements the three-stage model development process:
1. Initial hyperparameter tuning with GridSearchCV + PredefinedSplit
2. BorutaShap feature selection
3. Fine-tuned hyperparameter search on selected features
4. Final evaluation on the held-out test set
"""

from BorutaShap import BorutaShap
import numpy as np
import pandas as pd
from utils import custom_data_split, sep_x_y, compute_class_weights, shap_test_eval, roc_plot
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import PredefinedSplit, GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt
import copy
import logging

logging.basicConfig(filename="tune.log", level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


class SelectTune:
    """
    Feature selection and hyperparameter tuning for earthquake prediction.

    This class orchestrates the full model selection pipeline:
    - GridSearchCV with PredefinedSplit (train vs validation)
    - BorutaShap feature selection
    - Fine-tuned hyperparameter search
    - Test set evaluation with MCC and AUC metrics

    Parameters
    ----------
    model : estimator
        Base model (e.g., XGBClassifier).
    parameter_grid : dict
        Initial hyperparameter search grid.
    fixed_params : dict
        Parameters that remain fixed during tuning (e.g., verbosity, n_jobs).
    train_data : pd.DataFrame
        Training+validation dataset (features + 'Actual Mag' target as last col).
    threshold_mag : float
        Magnitude threshold for binary classification.
    region : str
        Region name (e.g., 'Chile', 'Japan').
    fs_type : str
        Feature set type: 'indicator', 'ts', or 'mixed'.
    """

    def __init__(self, model, parameter_grid, fixed_params, train_data,
                 threshold_mag, region, fs_type):
        self.param_grid = parameter_grid
        self.train_val = train_data
        self.thresh = threshold_mag
        self.region = region
        self.fixed_params = fixed_params
        self.model = copy.copy(model)
        self.model_origin = copy.copy(model)
        self.fs_type = fs_type
        self.plot = False
        self.result = None
        self.features = None
        self.best_param = None

    def gridsearch(self, X_train_r, X_train, y_train, param_fine):
        """
        The PredefinedSplit uses X_train_r indices as the training fold
        and the remaining indices in X_train as the validation fold.
        """
        if self.features is not None:
            X_train = X_train[self.features]
            X_train_r = X_train_r[self.features]

        split_index = [-1 if x in X_train_r.index else 0 for x in X_train.index]
        pds = PredefinedSplit(test_fold=split_index)

        ms = make_scorer(roc_auc_score, greater_is_better=True)

        self.model.set_params(**self.fixed_params)

        if param_fine is None:
            param_fine = self.param_grid

        gs = GridSearchCV(
            estimator=self.model,
            cv=pds,
            scoring=ms,
            param_grid=param_fine,
            return_train_score=True,
            verbose=4
        )
        gs_results = gs.fit(X_train, y_train)

        mean_score = gs_results.cv_results_['mean_test_score']
        self.best_param = gs.best_params_
        self.model.set_params(**self.best_param)
        self.model.set_params(**self.fixed_params)
        perf = gs_results.best_score_

        if not self.plot:
            self.result = {param: val for param, val in self.best_param.items()}
            self.result['mag'] = self.thresh
            self.result['performance'] = perf
            self.result['region'] = self.region
            self.result['type'] = self.fs_type
        return mean_score

    def tune(self, new_grid=None):
        X_train_r, y_train_r, X_train_Val, y_train_Val = custom_data_split(
            self.train_val, self.thresh, self.fs_type
        )
        logger.info(f'train_r and trainVal shape after custom split {X_train_r.shape} {X_train_Val.shape}')
        print((f'train_r and trainVal shape after custom split {X_train_r.shape} {X_train_Val.shape}'))
        mean_score = self.gridsearch(X_train_r, X_train_Val, y_train_Val, new_grid)

        return mean_score, X_train_Val, y_train_Val

    def param_investigate(self, new_grid):
        """
        Investigate a single parameter (e.g., n_estimators) by plotting
        mean CV scores against parameter values.

        """
        self.plot = True
        mean_score, X_train_Val, y_train_Val = self.tune(new_grid=new_grid)
        save_name = f"{self.region}_n_estimators_{self.fs_type}_{self.thresh}.png"
        save_path = Path(__file__).parent.joinpath('results', save_name)
        plt.errorbar(new_grid['n_estimators'], mean_score)
        plt.savefig(save_path)
        plt.show()
        self.plot = False

    def borutashap_fs(self, X, y):
        Feature_Selector = BorutaShap(
            model=self.model,
            importance_measure='shap',
            classification=True
        )

        Feature_Selector.fit(
            X=X, y=y, n_trials=100, sample=False,
            train_or_test='test', normalize=True,
            verbose=True
        )
        features = Feature_Selector.accepted
        Feature_Selector.plot(which_features='accepted')
        return features

    def feature_select(self):
        mean_score, X_train_Val, y_train_Val = self.tune()
        self.features = self.borutashap_fs(X_train_Val, y_train_Val)
        return self.features

    def hyperparameter_fine_tune(self, param_fine):
        self.model = self.model_origin
        mean_score, X_train_Val, y_train_Val = self.tune(new_grid=param_fine)

        final_result_para = pd.DataFrame([self.result])

        save_name = f'{self.region}_fine_tuned_{str(self.thresh)}.csv'
        save_path = Path(__file__).parent.joinpath('results', save_name)
        final_result_para.to_csv(save_path, mode='a', index=False)

        return self.model, X_train_Val, y_train_Val

    def _predict(self, whole):
        x, y = sep_x_y(whole, self.thresh)
        x = x[self.features]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

        x, y_train = sep_x_y(self.train_val, self.thresh)
        X_train = x[self.features]

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_pro = self.model.predict_proba(X_test)[:, 1]
        mcc = matthews_corrcoef(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pro)

        final_results = {
            'MCC': mcc,
            'AUC': auc,
            'Type': self.fs_type,
            'Mag': self.thresh,
            'Region': self.region
        }

        shap_test_eval(self.model, X_test, self.thresh, self.region, self.fs_type)
        roc_plot(y_test, y_pro, auc, self.fs_type)

        return final_results
