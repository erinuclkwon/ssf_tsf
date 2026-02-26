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
        Run GridSearchCV with PredefinedSplit for hyperparameter tuning.

        The PredefinedSplit uses X_train_r indices as the training fold
        and the remaining indices in X_train as the validation fold.

        Parameters
        ----------
        X_train_r : pd.DataFrame
            Reduced training features (inner fold).
        X_train : pd.DataFrame
            Full training+validation features.
        y_train : pd.Series
            Full training+validation labels.
        param_fine : dict or None
            Parameter grid to search. If None, uses self.param_grid.

        Returns
        -------
        np.ndarray
            Mean test scores from cross-validation.
        """
        print(self.features)
        logger.info(self.features)
        if self.features is not None:
            X_train = X_train[self.features]
            X_train_r = X_train_r[self.features]

        logger.info(f"fine tune: train shape: {X_train_r.shape}, trainVal shape: {X_train.shape, y_train.shape}")
        print(f"fine tune: train shape: {X_train_r.shape}, trainVal shape: {X_train.shape, y_train.shape}")
        split_index = [-1 if x in X_train_r.index else 0 for x in X_train.index]
        pds = PredefinedSplit(test_fold=split_index)

        ms = make_scorer(roc_auc_score, greater_is_better=True)

        self.model.set_params(**self.fixed_params)

        if param_fine is None:
            param_fine = self.param_grid

        logger.info(f'parameters and models passed before grid search{param_fine} {self.model}')
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
        logger.info(f"models obtained after grid search {self.model}")

        if not self.plot:
            self.result = {param: val for param, val in self.best_param.items()}
            self.result['mag'] = self.thresh
            self.result['performance'] = perf
            self.result['region'] = self.region
            self.result['type'] = self.fs_type
        return mean_score

    def tune(self, new_grid=None):
        """
        Run the tuning pipeline: split data then run grid search.

        Parameters
        ----------
        new_grid : dict, optional
            Override parameter grid. If None, uses self.param_grid.

        Returns
        -------
        mean_score : np.ndarray
            Mean test scores from cross-validation.
        X_train_Val : pd.DataFrame
            Training+validation features.
        y_train_Val : pd.Series
            Training+validation labels.
        """
        X_train_r, y_train_r, X_train_Val, y_train_Val = custom_data_split(
            self.train_val, self.thresh, self.fs_type
        )
        logger.info(f'train_r and trainVal shape after custom split {X_train_r.shape} {X_train_Val.shape}')
        mean_score = self.gridsearch(X_train_r, X_train_Val, y_train_Val, new_grid)

        return mean_score, X_train_Val, y_train_Val

    def param_investigate(self, new_grid):
        """
        Investigate a single parameter (e.g., n_estimators) by plotting
        mean CV scores against parameter values.

        Parameters
        ----------
        new_grid : dict
            Parameter grid (typically with a single key like 'n_estimators').
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
        """
        Run BorutaShap feature selection.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Binary target labels.

        Returns
        -------
        list of str
            Names of accepted (selected) features.
        """
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
        """
        Run initial tuning followed by BorutaShap feature selection.

        Returns
        -------
        list of str
            Selected feature names.
        """
        mean_score, X_train_Val, y_train_Val = self.tune()
        self.features = self.borutashap_fs(X_train_Val, y_train_Val)
        logger.info(f"features after fs select: {self.features}")
        logger.info(f"orginal model: {self.model_origin}")
        return self.features

    def hyperparameter_fine_tune(self, param_fine):
        """
        Fine-tune hyperparameters after feature selection.

        Resets the model to the original (pre-tuning) state before
        running the search on the selected feature subset.

        Parameters
        ----------
        param_fine : dict
            Fine-tuning parameter grid.

        Returns
        -------
        model : estimator
            Model with best hyperparameters set.
        X_train_Val : pd.DataFrame
            Training+validation features.
        y_train_Val : pd.Series
            Training+validation labels.
        """
        logger.info(f'train_val shape before fine tune{self.train_val.shape}{self.thresh}{self.fs_type}')
        logger.info(f"features used before running tune: {self.features}")
        logger.info(f"original model: {self.model_origin}")

        self.model = self.model_origin
        mean_score, X_train_Val, y_train_Val = self.tune(new_grid=param_fine)

        logger.info(f'model params after tune: {self.model.get_params()}')
        final_result_para = pd.DataFrame([self.result])

        save_name = f'{self.region}_fine_tuned_{str(self.thresh)}.csv'
        save_path = Path(__file__).parent.joinpath('results', save_name)
        final_result_para.to_csv(save_path, mode='a', index=False)

        return self.model, X_train_Val, y_train_Val

    def _predict(self, whole):
        """
        Evaluate the tuned model on the held-out test set.

        The model is trained on self.train_val (with selected features)
        and evaluated on the last 30% of the whole dataset.

        Parameters
        ----------
        whole : pd.DataFrame
            Complete dataset (features + 'Actual Mag' as last column).

        Returns
        -------
        dict
            Dictionary with MCC, AUC, feature type, magnitude threshold,
            and region.
        """
        x, y = sep_x_y(whole, self.thresh)
        logger.info(f"x: {x.shape}, y: {y.shape}")
        print(f"x: {x.shape}, y: {y.shape}")
        x = x[self.features]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

        x, y_train = sep_x_y(self.train_val, self.thresh)
        X_train = x[self.features]

        logger.info(f"shapes: {len(X_train)},{len(y_train)},{len(X_test)}, {len(y_test)}")
        print((len(X_train), len(y_train), len(X_test), len(y_test)))
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        y_pro = self.model.predict_proba(X_test)[:, 1]
        logger.info(f"ypred: {len(y_pred)}, y_pro: {len(y_pro)}")
        print(f"ypred: {y_pred}, y_pro: {y_pro}")
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
