import pandas as pd
import numpy as np
import copy
import os
import shutil
import optuna
from pathlib import Path
import joblib

import automl_alex
import sklearn
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from automl_alex.logger import *

predict_proba_metrics = ['roc_auc_score', 'log_loss', 'brier_score_loss']
TMP_FOLDER = '.automl-alex_tmp/'

class CrossValidation(object):
    """
    Cross-validation is a method for evaluating an analytical model and its behavior on independent data. 
    When evaluating the model, the available data is split into k parts. 
    Then the model is trained on k − 1 pieces of data, and the rest of the data is used for testing. 
    The procedure is repeated k times; in the end, each of the k pieces of data is used for testing. 
    The result is an assessment of the effectiveness of the selected model with the most even use of the available data.
    
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit model.
    folds=7 : int, default=7
        Number of folds. Must be at least 2.
    score_folds : int, default=5
        Number of score folds. Must be at least 1.
    n_repeats : int, default=1
        Number of times cross-validator needs to be repeated.
    metric : If None, the estimator's default scorer (if available) is used.
    print_metric=False :
            metric_round=4 (undefined):
    random_state : int, RandomState instance or None, default=42
        Controls the generation of the random states for each repetition.

    Notes
    -----
    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.
    """
    __name__ = 'CrossValidation'
    fit_models = False
    fited_models = {}
    estimator = None


    def __init__(
        self,  
        estimator, # model
        folds=7,
        score_folds=5,
        n_repeats=1,
        metric=None,
        print_metric=False, 
        metric_round=4, 
        random_state=42
        ):
        
        self.estimator = estimator
        self.folds = folds
        self.score_folds = score_folds
        self.n_repeats = n_repeats
        self.print_metric = print_metric
        self.metric_round = metric_round

        if metric is None:
            if estimator.type_of_estimator == 'classifier':
                self.metric = sklearn.metrics.roc_auc_score
            elif estimator.type_of_estimator == 'regression':
                self.metric = sklearn.metrics.mean_squared_error
        else:
            self.metric = metric

        if estimator.type_of_estimator == 'classifier':
            self.skf = RepeatedStratifiedKFold(
                n_splits=folds, 
                n_repeats=n_repeats,
                random_state=random_state,
                )
        else:
            self.skf = RepeatedKFold(
                n_splits=folds,
                n_repeats=n_repeats, 
                random_state=random_state,
                )


    def _clean_temp_folder(self):
        Path(TMP_FOLDER).mkdir(parents=True, exist_ok=True)
        if os.path.isdir(TMP_FOLDER+'cross-v_tmp'):
            shutil.rmtree(TMP_FOLDER+'cross-v_tmp')
        os.mkdir(TMP_FOLDER+'cross-v_tmp')


    @logger.catch
    def fit(self, X, y, cat_features=None):
        self._clean_temp_folder()
        self.cv_split_idx = [(train_idx, valid_idx) for (train_idx, valid_idx) in self.skf.split(X, y)]

        for i, (train_idx, valid_idx) in enumerate(self.cv_split_idx):
            train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
            # Fit
            self.estimator.fit(X_train=train_x, y_train=train_y, cat_features=cat_features)
            self.fited_models[f'model_{self.estimator.__name__}_fold_{i}'] = copy.deepcopy(self.estimator)
        self.fit_models = True


    @logger.catch
    def predict_test(self, X_test):
        if not self.fit_models:
            raise Exception("No fit models")

        stacking_y_pred_test = np.zeros(len(X_test))

        for i in range(self.folds*self.n_repeats):
            y_pred_test = self.fited_models[f'model_{self.estimator.__name__}_fold_{i}'].predict_or_predict_proba(X_test)
            stacking_y_pred_test += y_pred_test
        predict = stacking_y_pred_test / (self.folds*self.n_repeats)
        
        return(predict)


    @logger.catch
    def predict_train(self, X):
        if not self.fit_models:
            raise Exception("No fit models")

        stacking_y_pred_train = np.zeros(len(X))

        for i, (train_idx, valid_idx) in enumerate(self.cv_split_idx):
            val_x = X.iloc[valid_idx]
            y_pred = self.fited_models[f'model_{self.estimator.__name__}_fold_{i}'].predict_or_predict_proba(val_x)
            stacking_y_pred_train[valid_idx] += y_pred
        
        predict = stacking_y_pred_train / self.n_repeats
        
        return(predict)


    @logger.catch
    def get_feature_importance(self, X):
        if not self.fit_models:
            raise Exception("No fit models")
        
        if not self.estimator._is_possible_feature_importance():
            raise Exception("Can't get the feature importance for this estimator")

        feature_importance_df = pd.DataFrame(np.zeros(len(X.columns)), index=X.columns)

        for i in range(self.folds*self.n_repeats):
            if i == 0:
                feature_importance_df = \
                    self.fited_models[f'model_{self.estimator.__name__}_fold_{i}'].get_feature_importance(X)
            feature_importance_df['value'] += \
                self.fited_models[f'model_{self.estimator.__name__}_fold_{i}'].get_feature_importance(X)['value']
        
        return(feature_importance_df)


    @logger.catch
    def fit_score(self, X, y, print_metric=None, trial=None):
        self._pruned_cv = False
        if print_metric is None:
            print_metric = self.print_metric

        self.cv_split_idx = [(train_idx, valid_idx) for (train_idx, valid_idx) in self.skf.split(X, y)]

        folds_scores = []

        for i, (train_idx, valid_idx) in enumerate(self.cv_split_idx):
            train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
            val_x, val_y = X.iloc[valid_idx], y.iloc[valid_idx]
            # Fit

            score_model = self.estimator.fit_score( 
                X_train=train_x, 
                y_train=train_y, 
                X_test=val_x, 
                y_test=val_y,
                metric=self.metric,
                print_metric=False, 
                metric_round=self.metric_round, 
                )

            folds_scores.append(score_model)

            if (trial is not None) and i < 1:
                trial.report(score_model, i)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    self._pruned_cv=True
                    break

            # score_folds
            if i+1 >= self.score_folds:
                break
            
        if self._pruned_cv:
            score = score_model
            score_std = 0
        else:
            if self.score_folds > 1:
                score = round(np.mean(folds_scores), self.metric_round)
                score_std = round(np.std(folds_scores), self.metric_round+2)
            else:
                score = round(score_model, self.metric_round)
                score_std = 0

        if print_metric:
            print(f'\n Mean Score {self.metric.__name__} on {self.score_folds} Folds: {score} std: {score_std}')

        return(score, score_std)


    @logger.catch
    def save(self, name='cv_dump', folder='./', verbose=1):
        if not self.fit_models:
            raise Exception("No fit models")

        dir_tmp = TMP_FOLDER+'cross-v_tmp/'
        self._clean_temp_folder()

        for i in range(self.folds*self.n_repeats):
            self.fited_models[f'model_{self.estimator.__name__}_fold_{i}'].save(f'{dir_tmp}model_{self.estimator.__name__}_fold_{i}', verbose=0)

        joblib.dump(self, dir_tmp+'CV'+'.pkl')

        shutil.make_archive(folder+name, 'zip', dir_tmp)

        shutil.rmtree(dir_tmp)
        if verbose>0:
            print('Save CrossValidation')


    @logger.catch
    def load(self, name='cv_dump', folder='./', verbose=1):
        self._clean_temp_folder()
        dir_tmp = TMP_FOLDER+'cross-v_tmp/'

        shutil.unpack_archive(folder+name+'.zip', dir_tmp)

        cv = joblib.load(dir_tmp+'CV'+'.pkl')

        for i in range(cv.folds*cv.n_repeats):
            cv.fited_models[f'model_{self.estimator.__name__}_fold_{i}'] = \
                copy.deepcopy(cv.estimator.load(f'{dir_tmp}model_{self.estimator.__name__}_fold_{i}', verbose=0))

        shutil.rmtree(dir_tmp)
        if verbose>0:
            print('Load CrossValidation')
        return(cv)