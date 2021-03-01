import pandas as pd
import numpy as np
import sys
import time
import optuna
from tqdm import tqdm
import joblib

from .logger import *
from .automl_alex import BestSingleModel
import automl_alex
from .optimizer import *


import sklearn
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")

# disable chained assignments
pd.options.mode.chained_assignment = None 

predict_proba_metrics = ['roc_auc_score', 'log_loss', 'brier_score_loss']

class ModelBase(object):
    """
    Base class for a specific ML algorithm implementation factory,
    i.e. it defines algorithm-specific hyperparameter space and generic methods for model training & inference
    """
    pbar = 0
    model = None
    study = None
    history_trials = []
    history_trials_dataframe = pd.DataFrame()
    best_model_param = None

    def __init__(self,  
                    model_param=None, 
                    type_of_estimator=None, # classifier or regression
                    gpu=False, 
                    verbose=0,
                    random_state=42
                    ):
        self._gpu = gpu
        self._random_state = random_state
        logger_print_lvl(verbose)

        if type_of_estimator is not None:
            self.type_of_estimator = type_of_estimator

        self.model_param = self._init_default_model_param()
        if model_param is not None:
            self.model_param = self.model_param.update(model_param)
    

    def _init_default_model_param(self,):
        """
        Default model_param
        """
        model_param = {}
        return(model_param)


    def fit(self, X_train=None, y_train=None, X_test=None, y_test=None, verbose=False):
        """
        Args:
            X (pd.DataFrame, shape (n_samples, n_features)): the input data
            y (pd.DataFrame, shape (n_samples, ) or (n_samples, n_outputs)): the target data
        Return:
            model (Class)
        """
        raise NotImplementedError("Pure virtual class.")


    def predict(self, X=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        raise NotImplementedError("Pure virtual class.")


    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        raise NotImplementedError("Pure virtual class.")


    def predict_proba(self, X):
        """
        Args:
            dataset (np.array, shape (n_samples, n_features)): the input data

        Return:
            np.array, shape (n_samples, n_classes): predicted probabilities
        """
        raise NotImplementedError("Pure virtual class.")


    def predict_or_predict_proba(self, X):
        """
        Сheck and if it is possible get predict_proba
        """
        if (self.is_possible_predict_proba()) and \
                (self.type_of_estimator == 'classifier'):
            predicts = self.predict_proba(X)
        else:
            predicts = self.predict(X)
        return(predicts)


    def _is_possible_feature_importance(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return False


    def get_feature_importance(self, train_x, importance_type='gain',):
        """
        Return:
            list feature_importance
        """
        if not self._is_possible_feature_importance(): 
            raise Exception("Model cannot get feature_importance")
        raise NotImplementedError("Pure virtual class.")


    @logger.catch
    def score(self, 
            X_test, 
            y_test,
            metric=None,
            print_metric=False, 
            metric_round=4, 
            ):
        if self.model is None:
            raise Exception("No fit models")

        if metric is None:
            if self.type_of_estimator == 'classifier':
                metric = sklearn.metrics.roc_auc_score
            elif self.type_of_estimator == 'regression':
                metric = sklearn.metrics.mean_squared_error

        # Predict
        if (metric.__name__ in predict_proba_metrics):
            y_pred_test = self.predict_or_predict_proba(X_test)
        else:
            y_pred_test = self.predict(X_test)
        score = round(metric(y_test, y_pred_test),metric_round)

        if print_metric:
            logger_print_lvl(3)
            logger.info(f'{metric.__name__}: {score}')
        return(score)


    @logger.catch
    def fit_score(self, 
            X_train, 
            y_train, 
            X_test, 
            y_test,
            metric=None,
            print_metric=False, 
            metric_round=4, 
            ):
        start = time.time()
        # Fit
        self.fit(X_train, y_train,)

        total_time_fit = round((time.time() - start),2)
        if print_metric:
            logger_print_lvl(3)
            logger.info(f'fit time: {total_time_fit} sec')

        # Score
        score = self.score(X_test, y_test, 
            metric=metric,
            print_metric=print_metric, 
            metric_round=metric_round,
            )
        return(score)


    def y_format(self, y):
        if isinstance(y, pd.DataFrame):
            y = np.array(y[y.columns[0]].values)
        return y


    def get_model_opt_params(self, ):
        """
        Return:
            dict from parameter name to hyperopt distribution: default
            parameter space
        """
        raise NotImplementedError("Pure virtual class.")


    def __metric_direction_detected__(self, metric, y):
        zero_y = np.zeros(len(y))
        zero_score = metric(y, zero_y)
        best_score = metric(y, y)

        if best_score > zero_score:
            direction = 'maximize'
        else:
            direction = 'minimize'
        return(direction)



    def opt(self,X,y,
            timeout=200, # optimization time in seconds
            metric=None,
            metric_round=4,
            combined_score_opt=False,
            cold_start=30,
            auto_parameters=True,
            folds=7,
            score_folds=2,
            opt_lvl=2,
            early_stoping=100,
            verbose=1,):
        """
        Description of opt:
        in progress... 

        Args:
            timeout=100 (int):
            folds=None (None or int):
            cold_start=None (None or int):
            score_cv_folds=None (None or int):
            opt_lvl=None (None or int):
            direction=None (None or str):
            early_stoping=100 (int):
            feature_selection=True (bool):
            verbose=1 (int):
        
        Returns:
            history_trials (pd.DataFrame)
        """

        if metric is not None:
            self.metric = metric
            self.direction = self.__metric_direction_detected__(self.metric, y)
        else:
            if self.type_of_estimator == 'classifier':
                self.metric = sklearn.metrics.roc_auc_score
                self.direction = 'maximize'
            elif self.type_of_estimator == 'regression':
                self.metric = sklearn.metrics.mean_squared_error
                self.direction = 'minimize'

        logger.info(f'{self.type_of_estimator} optimize: {self.direction}')

        self.optimizer = BestSingleModel(
            type_of_estimator=self.type_of_estimator,
            models_names = [self.__name__,],
            feature_selection=False,
            auto_parameters=auto_parameters,
            folds=folds,
            score_folds=score_folds,
            metric=self.metric,
            metric_round=metric_round, 
            cold_start=cold_start,
            opt_lvl=opt_lvl,
            early_stoping=early_stoping,
            gpu=self._gpu,
            random_state=self._random_state)

        history = self.optimizer.opt(X, y, 
            timeout, 
            verbose=verbose, 
            )

        self.model_param = self.optimizer.cv_model.estimator.model_param
        self.fit(X,y)
        return(history)


    def plot_opt_param_importances(self,):
        '''
        Plot hyperparameter importances.
        '''
        if self.optimizer.study is None:
            raise Exception('No history to visualize!')
        return(optuna.visualization.plot_param_importances(self.optimizer.study))


    def plot_opt_history(self,):
        '''
        Plot optimization history of all trials in a study.
        '''
        if self.optimizer.study is None:
            raise Exception('No history to visualize!')
        return(optuna.visualization.plot_optimization_history(self.optimizer.study))


    def plot_parallel_coordinate(self,):
        """
        Plot the high-dimentional parameter relationships in a study.
        Note that, If a parameter contains missing values, a trial with missing values is not plotted.
        """
        if self.optimizer.study is None:
            raise Exception('No history to visualize!')
        return(optuna.visualization.plot_parallel_coordinate(self.optimizer.study))


    def plot_slice(self, params=None):
        """
        Plot the parameter relationship as slice plot in a study.
        Note that, If a parameter contains missing values, a trial with missing values is not plotted.
        """
        if self.optimizer.study is None:
            raise Exception('No history to visualize!')
        return(optuna.visualization.plot_slice(self.optimizer.study, params=params))

    
    def plot_contour(self, params=None):
        """
        Plot the parameter relationship as contour plot in a study.
        Note that, If a parameter contains missing values, a trial with missing values is not plotted.
        """
        if self.optimizer.study is None:
            raise Exception('No history to visualize!')
        return(optuna.visualization.plot_contour(self.optimizer.study, params=params))


    def _is_model_start_opt_params(self,):
        return(False)
    

    @logger.catch
    def save(self, name, verbose=1):
        joblib.dump(self, name+'.pkl')
        logger.info('Save Model')


    @logger.catch
    def load(self, name,verbose=1):
        model = joblib.load(name+'.pkl')
        logger.info('Load Model')
        return(model)
    


class ModelClassifier(ModelBase):
    type_of_estimator='classifier'


class ModelRegressor(ModelBase):
    type_of_estimator='regression'



class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    '''
    Custom EarlyStop for Optuna
    '''
    def __init__(self, early_stop=100, best_score = None):
        self.early_stop = early_stop
        self.early_stop_count = 0
        self.best_score = None

    def early_stopping_opt_maximize(self, study, trial):
        if self.best_score is None:
            self.best_score = study.best_value

        if study.best_value > self.best_score:
            self.best_score = study.best_value
            self.early_stop_count = 0
        else:
            if self.early_stop_count < self.early_stop:
                self.early_stop_count=self.early_stop_count+1
            else:
                self.early_stop_count = 0
                self.best_score = None
                raise EarlyStoppingExceeded()
    
    def early_stopping_opt_minimize(self, study, trial):
        if self.best_score is None:
            self.best_score = study.best_value

        if study.best_value < self.best_score:
            self.best_score = study.best_value
            self.early_stop_count = 0
        else:
            if self.early_stop_count < self.early_stop:
                self.early_stop_count=self.early_stop_count+1
            else:
                self.early_stop_count = 0
                self.best_score = None
                raise EarlyStoppingExceeded()