import pandas as pd
import numpy as np
import sys
import time
import optuna
from tqdm import tqdm
import joblib

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
            print(f'{metric.__name__}: {score}')
        return(score)


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
            print(f'fit time: {total_time_fit} sec')

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


    def cross_validation(
            self, 
            X=None,
            y=None,
            X_test=None,
            folds=10,
            score_folds=5,
            n_repeats=2,
            metric=None,
            print_metric=False, 
            metric_round=4, 
            predict=False,
            get_feature_importance=False,
            ):
        """
        Cross-validation is a method for evaluating an analytical model and its behavior on independent data. 
        When evaluating the model, the available data is split into k parts. 
        Then the model is trained on k − 1 pieces of data, and the rest of the data is used for testing. 
        The procedure is repeated k times; in the end, each of the k pieces of data is used for testing. 
        The result is an assessment of the effectiveness of the selected model with the most even use of the available data.
        
        Args:
                X : array-like of shape (n_samples, n_features)
                    The data to fit. Can be for example a list, or an array.
                y : array-like of shape (n_samples,) or (n_samples, n_outputs),
                    The target variable to try to predict in the case of
                    supervised learning.
                model : estimator object implementing 'fit'
                    The object to use to fit the data.
                folds=10 :
                score_folds=5 :
                n_repeats=2 :
                metric : If None, the estimator's default scorer (if available) is used.
                print_metric=False :
                metric_round=4 (undefined):
                predict=False (undefined):
                get_feature_importance=False (undefined):
            
            Returns:
                result (dict)
        """
        if get_feature_importance:
            if not self._is_possible_feature_importance():
                get_feature_importance = False

        if metric is None:
            if self.type_of_estimator == 'classifier':
                metric = sklearn.metrics.roc_auc_score
            elif self.type_of_estimator == 'regression':
                metric = sklearn.metrics.mean_squared_error

        if self.type_of_estimator == 'classifier':
            skf = RepeatedStratifiedKFold(
                n_splits=folds, 
                n_repeats=n_repeats,
                random_state=self._random_state,
                )
        else:
            skf = RepeatedKFold(
                n_splits=folds,
                n_repeats=n_repeats, 
                random_state=self._random_state,
                )

        folds_scores = []
        result = {}
        stacking_y_pred_train = np.zeros(len(X))
        if predict:
            stacking_y_pred_test = np.zeros(len(X_test))
        feature_importance_df = pd.DataFrame(np.zeros(len(X.columns)), index=X.columns)

        for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

            train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
            val_x, val_y = X.iloc[valid_idx], y.iloc[valid_idx]

            # Fit
            self.fit(X_train=train_x, y_train=train_y,)

            # Predict
            if (metric.__name__ in predict_proba_metrics):
                y_pred = self.predict_or_predict_proba(val_x)
                if predict:
                    y_pred_test = self.predict_or_predict_proba(X_test)
            else:
                y_pred = self.predict(val_x)
                if predict:
                    y_pred_test = self.predict(X_test)

            score_model = metric(val_y, y_pred)
            folds_scores.append(score_model)

            if get_feature_importance:
                if i == 0:
                    feature_importance_df = self.get_feature_importance(train_x)
                feature_importance_df['value'] += self.get_feature_importance(train_x)['value']
            
            if predict:
                stacking_y_pred_train[valid_idx] += y_pred
                stacking_y_pred_test += y_pred_test
            else:
                # score_folds
                if i+1 >= score_folds:
                    break

        if predict:
            stacking_y_pred_train = stacking_y_pred_train / n_repeats
            stacking_y_pred_test = stacking_y_pred_test / (folds*n_repeats)
            result['test_predict'] = stacking_y_pred_test
            result['train_predict'] = stacking_y_pred_train

        if get_feature_importance:
            result['feature_importance'] = dict(feature_importance_df['value'])
        
        if score_folds > 1 or predict:
            score = round(np.mean(folds_scores), metric_round)
            score_std = round(np.std(folds_scores), metric_round+2)
        else:
            score = round(score_model, metric_round)
            score_std = 0
        
        result['score'] = score
        result['score_std'] = score_std

        if print_metric:
            print(f'\n Mean Score {metric.__name__} on {i+1} Folds: {score} std: {score_std}')
        return(result)


    def get_model_opt_params(self, ):
        """
        Return:
            dict from parameter name to hyperopt distribution: default
            parameter space
        """
        raise NotImplementedError("Pure virtual class.")


    def __calc_combined_score_opt__(self, direction, score, score_std):
        """
        Args:
            direction (str): 'minimize' or 'maximize'
            score (float): the input score
            score_std (float): the input score_std

        Return:
            score_opt (float): combined score
        """
        if direction == 'maximize':
            score_opt = score - score_std
        else:
            score_opt = score + score_std
        return(score_opt)


    def __auto_parameters_calc(self, possible_iters, verbose=1):
        """
        Automatic determination of optimization parameters depending on the number of possible iterations

        Args:
            possible_iters (int): possible_iters
            verbose (int): print status

        Return:
            early_stoping (int)
            cv (int)
            score_cv_folds (int)
            opt_lvl (int)
            cold_start (int)
        """
        if verbose > 0: 
                print('> Start Auto calibration parameters')
        
        early_stoping = 50
        folds = 5
        score_folds = 1
        opt_lvl = 1
        cold_start = 10
            
        if possible_iters > 100:
            folds = 5
            score_folds = 2
            opt_lvl = 1
            cold_start = possible_iters // 2
            early_stoping = 100

        if possible_iters > 200:
            score_folds = 2
            opt_lvl = 2
            cold_start = (possible_iters / score_folds) // 3

        if possible_iters > 300:
            folds = 10
            score_folds = 3
            cold_start = (possible_iters / score_folds) // 5

        if possible_iters > 900:
            score_folds = 5
            opt_lvl = 3
            early_stoping = cold_start * 2
        
        if possible_iters > 10000:
            opt_lvl = 4
            score_folds = 10
            cold_start = (possible_iters / score_folds) // 10
            early_stoping = cold_start * 2

        if possible_iters > 25000:
            opt_lvl = 5
            score_folds = 20
            cold_start = (possible_iters / score_folds) // 30
            early_stoping = cold_start * 2
        return(early_stoping, folds, score_folds, opt_lvl, cold_start,)


    def _tqdm_opt_print(self, pbar):
        """
        Printing information in tqdm. Use pbar. 
        See the documentation for tqdm: https://github.com/tqdm/tqdm
        """
        if pbar is not None:
            if self.direction == 'maximize':
                self.history_trials_dataframe = pd.DataFrame(self.history_trials).sort_values('score_opt', ascending=False)
            else:
                self.history_trials_dataframe = pd.DataFrame(self.history_trials).sort_values('score_opt', ascending=True)
            
            best_trail = self.history_trials_dataframe.head(1)
            best_model_name = best_trail['model_name'].iloc[0]
            self.best_score = best_trail['score_opt'].iloc[0]
            self.best_model_param = best_trail['model_param'].iloc[0]
            self.best_score_std = best_trail['score_std'].iloc[0]
            best_metric_score = best_trail['model_score'].iloc[0]

            message = f' | Model: {best_model_name} | OptScore: {self.best_score} | Best {self.metric.__name__}: {best_metric_score} '
            if self._score_folds > 1:
                message+=f'+- {self.best_score_std}'
            pbar.set_postfix_str(message)
            pbar.update(1)


    def _print_opt_parameters(self, early_stoping, feature_selection):
        print('CV_Folds = ', self._folds)
        print('Score_CV_Folds = ', self._score_folds)
        print('Feature_Selection = ', feature_selection)
        print('Opt_lvl = ', self._opt_lvl)
        print('Cold_start = ', self._cold_start)
        print('Early_stoping = ', early_stoping)
        print('Metric = ', self.metric.__name__)
        print('Direction = ', self.direction)


    def _opt_model(self, trial, len_data, model=None):
        """
        Description of _opt_model:
            Model extraction for optimization with new parameters
            Created for a more flexible change of the model in optimization during class inheritance

        Args:
            trial (undefined):
            model=None (None or class):

        """
        self.model_param = self.get_model_opt_params(
            trial=trial, 
            opt_lvl=self._opt_lvl, 
            len_data=len_data,
            )
        return(self)


    def metric_direction_detected(self, metric, y):
        zero_y = np.zeros(len(y))
        zero_score = metric(y, zero_y)
        best_score = metric(y, y)

        if best_score > zero_score:
            direction = 'maximize'
        else:
            direction = 'minimize'
        return(direction)


    def _opt_feature_selector(self, columns, trial):
        """
        Description of _opt_feature_selector

        Args:
            columns (list):
            trial (undefined):

        Returns:
            selected columns (list)
        """
        select_columns = {}
        for colum in columns:
            select_columns[colum] = trial.suggest_categorical(colum, [True, False])
        select_columns_ = {k: v for k, v in select_columns.items() if v is True}
        
        if select_columns_:
            result = select_columns_.keys()
        else:
            result = columns
        return(result)


    def _opt_core(self, X, y, timeout, early_stoping, feature_selection, iterations, iteration_check, verbose=1):
        """
        Description of _opt_core:
            in progress...

        Args:
            timeout (int):
            early_stoping (int):
            feature_selection (bool):
            verbose=1 (int):

        Returns:
            history_trials_dataframe (pd.DataFrame)
        """

        # time model
        time.sleep(0.1)
        start_time = time.time()
        score, score_std = self.cross_validation(X=X, y=y, folds=self._folds, score_folds=1, print_metric=False,)
        iter_time = (time.time() - start_time)

        if verbose > 0: 
            print(f'One iteration takes ~ {round(iter_time,1)} sec')
        
        if iterations is None:
            possible_iters = timeout // (iter_time)

            if (possible_iters < 100) and iteration_check:
                print("Not enough time to find the optimal parameters. \n \
                    Possible iters < 100. \n \
                    Please, Increase the 'timeout' parameter for normal optimization.")
                raise Exception('Not enough time to find the optimal parameters')

            # Auto_parameters
            if self._auto_parameters:
                early_stoping, self._folds, self._score_folds, self._opt_lvl, self._cold_start = \
                    self.__auto_parameters_calc(possible_iters, verbose)
        
        cv_result = self.cross_validation(X=X, y=y, folds=self._folds, score_folds=self._score_folds, print_metric=False,)
        # _combined_score_opt
        if self._combined_score_opt:
            score_opt = self.__calc_combined_score_opt__(self.direction, cv_result['score'], cv_result['score_std'])
        else: 
            score_opt = cv_result['score']
        self.best_score = round(score_opt, self._metric_round)

        if verbose > 0: 
            print('> Start optimization with the parameters:')
            self._print_opt_parameters(early_stoping, feature_selection)
            print('#'*50)
            print(f'Default model OptScore = {self.best_score}')
        
        # OPTUNA objective
        def objective(trial, fast_check=True):
            # generate model
            opt_model = self._opt_model(trial=trial, len_data=len(X),)
            # feature selector
            data_kwargs = {}
            data_kwargs['X'] = X
            data_kwargs['y'] = y
            select_columns = X.columns.values
            if feature_selection:
                select_columns = self._opt_feature_selector(X.columns, trial=trial)
                data_kwargs['X'] = X[select_columns]
            # score
            cv_result = opt_model.cross_validation(
                folds=opt_model._folds, 
                score_folds=opt_model._score_folds, 
                metric=self.metric,
                print_metric=False,
                **data_kwargs,
                )

            # _combined_score_opt
            if self._combined_score_opt:
                score_opt = self.__calc_combined_score_opt__(self.direction, cv_result['score'], cv_result['score_std'])
            else: 
                score_opt = cv_result['score']
            score_opt = round(score_opt, self._metric_round)

            # History trials
            self.history_trials.append({
                'score_opt': score_opt,
                'model_score': cv_result['score'],
                'score_std': cv_result['score_std'],
                'model_name': opt_model.__name__,
                'model_param': opt_model.model_param,
                'columns': select_columns,
                'cv_folds': opt_model._folds,
                                })
            
            # verbose
            if verbose >= 1:
                self._tqdm_opt_print(pbar)
            return score_opt
        
        sampler=optuna.samplers.TPESampler(#consider_prior=True, 
                                            #prior_weight=1.0, 
                                            #consider_magic_clip=True, 
                                            #consider_endpoints=False, 
                                            n_startup_trials=self._cold_start, 
                                            #n_ei_candidates=50, 
                                            seed=self._random_state,
                                            #multivariate=True,
                                            )
        print('optimize:',self.direction)
        if self.study is None:
            self.study = optuna.create_study(direction=self.direction, sampler=sampler,)

        if verbose < 2:
            optuna.logging.disable_default_handler()

        es = EarlyStoppingExceeded()
        es.early_stop = early_stoping
        es.early_stop_count = 0
        es.best_score = None

        es_callback = es.early_stopping_opt_minimize
        if self.direction == 'maximize':
            es_callback = es.early_stopping_opt_maximize

        if verbose > 0:
            disable_tqdm = False
        else:
            disable_tqdm = True
            
        study_params = {}
        if iterations is not None:
            study_params['n_trials'] = iterations
        else:
            study_params['timeout'] = timeout

        with tqdm(
            file=sys.stdout,
            desc="Optimize: ", 
            disable=disable_tqdm,
            ) as pbar:
            try:
                self.study.optimize(
                    objective,  
                    callbacks=[es_callback], 
                    show_progress_bar=False,
                    **study_params,
                    )
            except EarlyStoppingExceeded:
                if verbose == 1: 
                    print(f'\n EarlyStopping Exceeded: Best Score: {self.study.best_value}', 
                        self.metric.__name__)
        
        self.history_trials_dataframe = pd.DataFrame(self.history_trials).sort_values('score_opt', ascending=True)
        if self.direction == 'maximize':
            self.history_trials_dataframe = pd.DataFrame(self.history_trials).sort_values('score_opt', ascending=False)
        self.model_param = self.history_trials_dataframe['model_param'].iloc[0]
        return(self.history_trials_dataframe)

    def opt(self,X,y,
            timeout=200, # optimization time in seconds
            metric=None,
            metric_round=4,
            combined_score_opt=False,
            iterations=None,
            cold_start=30,
            auto_parameters=True,
            folds=10,
            score_folds=2,
            opt_lvl=2,
            early_stoping=100,
            feature_selection=False,
            iteration_check=True,
            verbose=1,):
        """
        Description of opt:
        in progress... 

        Args:
            timeout=100 (int):
            cv_folds=None (None or int):
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
            self.direction = self.metric_direction_detected(self.metric, y)
        else:
            if self.type_of_estimator == 'classifier':
                self.metric = sklearn.metrics.roc_auc_score
                self.direction = 'maximize'
            elif self.type_of_estimator == 'regression':
                self.metric = sklearn.metrics.mean_squared_error
                self.direction = 'minimize'

        print(self.type_of_estimator,
            'optimize:',self.direction)

        self._combined_score_opt = combined_score_opt
        self._metric_round = metric_round

        self._folds = folds
        self._score_folds = score_folds
        self._cold_start = cold_start
        self._opt_lvl = opt_lvl
        self._auto_parameters = auto_parameters

        history = self._opt_core(X,y,
            timeout, 
            early_stoping, 
            feature_selection,
            iterations=iterations,
            iteration_check=iteration_check,
            verbose=verbose,)

        return(history)

    def plot_opt_history(self, figsize=(15,5)):
        """
        Plot optimization history of all trials in a study.
        """
        best_score_ls = []
        opt_df = pd.DataFrame(self.history_trials)
        for i, score in enumerate(opt_df.score_opt):
            if i == 0:
                best_score = score
                best_score_ls.append(score)
            else:
                if self.direction == 'maximize':
                    if best_score < score:
                        best_score = score
                        best_score_ls.append(best_score)
                    else:
                        best_score_ls.append(best_score)
                else:
                    if best_score > score:
                        best_score = score
                        best_score_ls.append(best_score)
                    else:
                        best_score_ls.append(best_score)

        opt_df['best_score'] = best_score_ls
        opt_df['Id'] = list(opt_df.index)

        plt.figure(figsize=figsize) 
        points = plt.scatter(x=opt_df.Id, y=opt_df.score_opt, label='Iter Score',
                            c=opt_df.score_opt, s=25, cmap="coolwarm")
        plt.colorbar(points)
        plt.plot(opt_df.best_score, color='red', label='Best Score',)
        plt.xlabel("Iter")
        plt.ylabel("Score")
        plt.title('Plot optimization history')
        plt.legend()
        return(plt.show())
    
    def plot_opt_params_parallel_coordinate(self,):
        """
        Plot the high-dimentional parameter relationships in a study.
        Note that, If a parameter contains missing values, a trial with missing values is not plotted.
        """
        if self.study is None:
            raise Exception('No history to visualize!')
        return(optuna.visualization.plot_parallel_coordinate(self.study))

    def plot_opt_params_slice(self, params=None):
        """
        Plot the parameter relationship as slice plot in a study.
        Note that, If a parameter contains missing values, a trial with missing values is not plotted.
        """
        if self.study is None:
            raise Exception('No history to visualize!')
        return(optuna.visualization.plot_slice(self.study, params=params))
    
    def save(self, name, verbose=1):
        joblib.dump(self, name+'.pkl')
        if verbose>0:
            print('Save Model')

    def load(self, name,verbose=1):
        model = joblib.load(name+'.pkl')
        if verbose>0:
            print('Load Model')
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