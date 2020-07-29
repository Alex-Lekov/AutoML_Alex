from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import *
from tqdm import tqdm
import sklearn
import optuna
import pandas as pd
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style="darkgrid")

# disable chained assignments
pd.options.mode.chained_assignment = None 

from automl_alex.databunch import DataBunch
from automl_alex.encoders import *


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

    def __init__(self,  
                X_train=None, 
                y_train=None,
                X_test=None,
                y_test=None,
                cat_features=None,
                clean_and_encod_data=True,
                cat_encoder_names=['OneHotEncoder', 'HelmertEncoder', 'HashingEncoder', 'FrequencyEncoder'],
                clean_nan=True,
                num_generator_features=True,
                group_generator_features=False,
                frequency_enc_num_features=True,
                normalization=True,
                databunch=None,
                model_param=None, 
                wrapper_params=None,
                auto_parameters=True,
                cv=10,
                score_cv_folds=5, # how many folds are actually used
                opt_lvl=3,
                metric=None,
                direction=None,
                combined_score_opt=True,
                metric_round=4, 
                cold_start=100,
                gpu=False, 
                type_of_estimator=None, # classifier or regression
                verbose=0,
                random_state=42):
        if type_of_estimator is not None:
            self.type_of_estimator = type_of_estimator

        if metric is not None:
            self.metric = metric
        else:
            if self.type_of_estimator == 'classifier':
                self.metric = sklearn.metrics.roc_auc_score
                self.direction = 'maximize'
            elif self.type_of_estimator == 'regression':
                self.metric = sklearn.metrics.mean_squared_error
                self.direction = 'minimize'

        if direction is not None:
            self.direction = direction
        self._auto_parameters = auto_parameters
        self._metric_round = metric_round
        self._cv = cv
        self._cold_start = cold_start
        self._gpu = gpu
        self._random_state = random_state
        self._score_cv_folds = score_cv_folds
        self._opt_lvl = opt_lvl
        self._combined_score_opt = combined_score_opt

        self.wrapper_params = wrapper_params
        if wrapper_params is None:
            self.wrapper_params = self._init_default_wrapper_params()

        self.model_param = model_param
        if model_param is None:
            self.model_param = self._init_default_model_param()

        self.history_trials = []
        self.history_trials_dataframe = pd.DataFrame()

        # dataset
        if databunch: 
            self._data = databunch
        else:
            if X_train is not None:
                self._data = DataBunch(X_train=X_train, 
                                    y_train=y_train,
                                    X_test=X_test,
                                    y_test=y_test,
                                    cat_features=cat_features,
                                    clean_and_encod_data=clean_and_encod_data,
                                    cat_encoder_names=cat_encoder_names,
                                    clean_nan=clean_nan,
                                    num_generator_features=num_generator_features,
                                    group_generator_features=group_generator_features,
                                    frequency_enc_num_features=frequency_enc_num_features,
                                    verbose=verbose,
                                    random_state=random_state,)
            else: 
                raise Exception("no Data?")
        #self._init_dataset()

    def _init_default_wrapper_params(self,):
        """
        Default wrapper_params
        """
        wrapper_params = {}
        return(wrapper_params)
    
    def _init_default_model_param(self,):
        """
        Default model_param
        """
        model_param = {}
        return(model_param)

    def _fit(self, dataset, weights=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
        Return:
            self
        """
        raise NotImplementedError("Pure virtual class.")


    def save_snapshot(self, filename):
        """
        Return:
            serializable internal model state snapshot.
        """
        raise NotImplementedError("Pure virtual class.")

    @staticmethod
    def load_from_snapshot(self, filename):
        """
        :snapshot serializable internal model state
        loads from serializable internal model state snapshot.
        """
        raise NotImplementedError("Pure virtual class.")


    def _predict(self, dataset):
        """
        Args:
            dataset : the input data,
                dataset.y may be None

        Return:
            np.array, shape (n_samples, ): predictions
        """
        raise NotImplementedError("Pure virtual class.")


    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        raise NotImplementedError("Pure virtual class.")


    def _predict_proba(self, X):
        """
        Args:
            dataset (np.array, shape (n_samples, n_features)): the input data

        Return:
            np.array, shape (n_samples, n_classes): predicted probabilities
        """
        raise NotImplementedError("Pure virtual class.")

    #@staticmethod
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
        cv = 5
        score_cv_folds = 1
        opt_lvl = 1
        cold_start = 10
            
        if possible_iters > 100:
            cv = 5
            score_cv_folds = 2
            opt_lvl = 1
            cold_start = possible_iters // 2
            early_stoping = 100

        if possible_iters > 200:
            score_cv_folds = 2
            opt_lvl = 2
            cold_start = (possible_iters / score_cv_folds) // 3

        if possible_iters > 300:
            cv = 10
            score_cv_folds = 3
            cold_start = (possible_iters / score_cv_folds) // 5

        if possible_iters > 900:
            score_cv_folds = 5
            opt_lvl = 3
            early_stoping = cold_start * 2
        
        if possible_iters > 10000:
            opt_lvl = 4
            score_cv_folds = 10
            cold_start = (possible_iters / score_cv_folds) // 10
            early_stoping = cold_start * 2

        if possible_iters > 25000:
            opt_lvl = 5
            score_cv_folds = 20
            cold_start = (possible_iters / score_cv_folds) // 30
            early_stoping = cold_start * 2
        return(early_stoping, cv, score_cv_folds, opt_lvl, cold_start,)

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
            self.best_score_std = best_trail['score_std'].iloc[0]
            best_metric_score = best_trail['model_score'].iloc[0]

            message = f' | Model: {best_model_name} | OptScore: {self.best_score} | Best {self.metric.__name__}: {best_metric_score} '
            if self._score_cv_folds > 1:
                message+=f'+- {self.best_score_std}'
            pbar.set_postfix_str(message)
            pbar.update(1)

    def _print_opt_parameters(self, early_stoping, feature_selection):
        print('CV_Folds = ', self._cv)
        print('Score_CV_Folds = ', self._score_cv_folds)
        print('Feature_Selection = ', feature_selection)
        print('Opt_lvl = ', self._opt_lvl)
        print('Cold_start = ', self._cold_start)
        print('Early_stoping = ', early_stoping)
        print('Metric = ', self.metric.__name__)
        print('Direction = ', self.direction)

    def _opt_model(self, trial, model=None):
        """
        Description of _opt_model:
            Model extraction for optimization with new parameters
            Created for a more flexible change of the model in optimization during class inheritance

        Args:
            trial (undefined):
            model=None (None or class):

        """
        if model is None:
            model = self

        model.model_param = model.get_model_opt_params(
            trial=trial, 
            model=model, 
            opt_lvl=model._opt_lvl, 
            metric_name=model.metric.__name__,
            )
        return(model)
    
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

    def _opt_core(self, timeout, early_stoping, feature_selection, iterations, iteration_check, verbose=1):
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
        # X
        #X=self._data.X_train
        # time model
        start_time = time.time()
        score, score_std = self.cross_val_score(X=self._data.X_train, folds=self._cv, score_folds=2, print_metric=False,)
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
                early_stoping, self._cv, self._score_cv_folds, self._opt_lvl, self._cold_start = \
                    self.__auto_parameters_calc(possible_iters, verbose)
        
        config = self.fit(print_metric=False,)
        self.best_score = config['score_opt'].iloc[0]

        if verbose > 0: 
            print('> Start optimization with the parameters:')
            self._print_opt_parameters(early_stoping, feature_selection)
            print('#'*50)
            print(f'Default model OptScore = {round(self.best_score,4)}')
        
        # OPTUNA objective
        def objective(trial, fast_check=True):
            # generate model
            opt_model = self._opt_model(trial=trial)
            # feature selector
            data_kwargs = {}
            select_columns = opt_model._data.X_train.columns.values
            if feature_selection:
                select_columns = self._opt_feature_selector(
                    opt_model._data.X_train.columns, 
                    trial=trial)
                data_kwargs['X'] = opt_model._data.X_train[select_columns]
                data_kwargs['y'] = opt_model._data.y_train
            # score
            score, score_std = opt_model.cross_val_score(
                folds=opt_model._cv, 
                score_folds=opt_model._score_cv_folds, 
                print_metric=False,
                **data_kwargs,
                )

            # _combined_score_opt
            if self._combined_score_opt:
                score_opt = self.__calc_combined_score_opt__(self.direction, score, score_std)
            else: 
                score_opt = score
            score_opt = round(score_opt, self._metric_round)

            # History trials
            self.history_trials.append({
                'score_opt': score_opt,
                'model_score': score,
                'score_std': score_std,
                'model_name': opt_model.__name__,
                'model_param': opt_model.model_param,
                'wrapper_params': opt_model.wrapper_params,
                'cat_encoders': opt_model._data.cat_encoder_names,
                'columns': select_columns,
                'cv_folds': opt_model._cv,
                                })
            
            # verbose
            if verbose >= 1:
                self._tqdm_opt_print(pbar)
            return score_opt
        
        sampler=optuna.samplers.TPESampler(consider_prior=True, 
                                            prior_weight=1.0, 
                                            consider_magic_clip=True, 
                                            consider_endpoints=False, 
                                            n_startup_trials=self._cold_start, 
                                            n_ei_candidates=50, 
                                            seed=self._random_state)
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
        return(self.history_trials_dataframe)

    def opt(self,
            timeout=100, # optimization time in seconds
            iterations=None,
            auto_parameters=None,
            cv_folds=None,
            cold_start=None,
            score_cv_folds=None,
            opt_lvl=None,
            direction=None,
            early_stoping=100,
            feature_selection=True,
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
        if cv_folds is not None:
            self._cv = cv_folds
        if score_cv_folds is not None:
            self._score_cv_folds = score_cv_folds
        if cold_start is not None:
            self._cold_start = cold_start
        if opt_lvl is not None:
            self._opt_lvl = opt_lvl
        if auto_parameters is not None:
            self._auto_parameters = auto_parameters
        if direction is not None:
            self.direction = direction
        if self.direction is None:
            raise Exception('Need direction for optimize!')

        history = self._opt_core(
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

    def preproc_data_in_cv(self, train_x, train_y, val_x, X_test):
        """
        Preproc data in CV for TargetEncoders
        in progress...
        """
        return(train_x, val_x, X_test)

    def cross_val(self, 
        X=None,
        y=None,
        X_test=None,
        model=None, 
        folds=10,
        score_folds=5,
        n_repeats=2,
        print_metric=False, 
        metric_round=4, 
        predict=False,
        get_feature_importance=False,
        ):
        """
        Description of cross_val:
            Cross-validation function

        Args:
            X=None (undefined):
            y=None (undefined):
            X_test=None (undefined):
            model=None (undefined):
            folds=10 (undefined):
            score_folds=5 (undefined):
            n_repeats=2 (undefined):
            print_metric=False (undefined):
            metric_round=4 (undefined):
            predict=False (undefined):
            get_feature_importance=False (undefined):
        
        Returns:
            result (dict)
        """
        if model is None:
            model = self

        if X is None:
            X = model._data.X_train
        if y is None:
            y = model._data.y_train
            
        if X_test is None:
            X_test = model._data.X_test

        if predict and (X_test is None):
            raise Exception("No X_test for predict")

        if model.type_of_estimator == 'classifier':
            skf = RepeatedStratifiedKFold(
                n_splits=folds, 
                n_repeats=n_repeats,
                random_state=model._random_state,
                )
        else:
            skf = RepeatedKFold(
                n_splits=folds,
                n_repeats=n_repeats, 
                random_state=model._random_state,
                )

        folds_scores = []
        stacking_y_pred_train = np.zeros(X.shape[0])
        stacking_y_pred_test = np.zeros(X_test.shape[0])
        feature_importance_df = pd.DataFrame(np.zeros(len(X.columns)), index=X.columns)

        for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

            train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
            val_x, val_y = X.iloc[valid_idx], y.iloc[valid_idx]

            # TargetEncoders
            train_x, val_x, X_test = model.preproc_data_in_cv(train_x, train_y, val_x, X_test) 
            
            # Fit
            model._fit(model=model,
                X_train=train_x.reset_index(drop=True), 
                y_train=train_y.reset_index(drop=True), 
                X_test=val_x.reset_index(drop=True), 
                y_test=val_y.reset_index(drop=True),
                )

            # Predict
            if (model.metric.__name__ in predict_proba_metrics) and (model.is_possible_predict_proba()):
                y_pred = model._predict_proba(val_x)
                if predict:
                    y_pred_test = model._predict_proba(X_test)
            else:
                y_pred = model._predict(val_x)
                if predict:
                    y_pred_test = model._predict(X_test)

            score_model = model.metric(val_y, y_pred)
            folds_scores.append(score_model)

            if get_feature_importance:
                feature_importance_df += model._get_feature_importance(train_x)

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
        
        if score_folds > 1 or predict:
            score = round(np.mean(folds_scores), metric_round)
            score_std = round(np.std(folds_scores), metric_round+2)
        else:
            score = round(score_model, metric_round)
            score_std = 0

        if print_metric:
            print(f'\n Mean Score {model.metric.__name__} on {i+1} Folds: {score} std: {score_std}')

        # Total
        result = {
            'Score':score,
            'Score_Std':score_std,
            'Test_predict':stacking_y_pred_test,
            'Train_predict':stacking_y_pred_train,
            'Feature_importance': dict(feature_importance_df[0]),
            }
        return(result)

    def cross_val_score(self, **kwargs):
        """
        cross_val_score

        Args:
            **kwargs (kwargs): 

        Returns:
            score, score_std (float)
            
        """
        res = self.cross_val(predict=False,**kwargs)
        score = res['Score']
        score_std = res['Score_Std']
        return(score, score_std)
    
    def cross_val_predict(self, **kwargs):
        """
        Description of cross_val_predict

        Args:
            **kwargs (kwargs):

        Returns:
            predict_test, predict_train (array)
            
        """
        res = self.cross_val(predict=True, **kwargs)
        predict_test = res['Test_predict']
        predict_train = res['Train_predict']
        return(predict_test, predict_train)

    def fit(self, model=None, print_metric=True):
        """
        Description of fit

        Args:
            model=None (Class or None):
            print_metric=True (bool):

        Returns:
            config (pd.DataFrame)
            
        """
        if model is None:
            model = self
        score, score_std = model.cross_val_score(
            model=model, 
            folds=model._cv,
            score_folds=model._score_cv_folds,
            metric_round=model._metric_round,
            print_metric=print_metric,
            )

        if model._combined_score_opt:
            score_opt = model.__calc_combined_score_opt__(model.direction, score, score_std)
        else: 
            score_opt = score

        config = {
            'score_opt': score_opt,
            'model_score': score,
            'score_std': score_std,
            'model_name': model.__name__,
            'model_param': model.model_param,
            'wrapper_params': model.wrapper_params,
            'cat_encoder': model._data.cat_encoder_names,
            'columns': model._data.X_train.columns.values,
            'cv_folds': model._cv,
            }
        return(pd.DataFrame([config,]))
    
    def _predict_on_full_dataset(self, X=None, y=None, X_test=None, model=None,):
        if model is None:
            model = self
        if X is None:
            X = model._data.X_train
        if y is None:
            y = model._data.y_train
        if X_test is None:
            X_test = model._data.X_test
        if X_test is None:
            raise Exception("No X_test for predict")
        
        # TargetEncoders
        X, X_test, _ = model.preproc_data_in_cv(X, y, X_test, None) 
            
        # Fit
        model._fit(model=model,
            X_train=X.reset_index(drop=True), 
            y_train=y.reset_index(drop=True), 
            )
        
        # Predict
        if (model.metric.__name__ in predict_proba_metrics) and (model.is_possible_predict_proba()):
            y_pred = model._predict_proba(X)
            y_pred_test = model._predict_proba(X_test)
        else:
            y_pred = model._predict(X)
            y_pred_test = model._predict(X_test)
        
        result = {
            'Test_predict': y_pred_test,
            'Train_predict': y_pred,
            }
        return(result)

    def _predict_preproc_model(self, model_cfg, model,):
        model.model_param = model_cfg['model_param']
        model.wrapper_params = model_cfg['wrapper_params']
        return(model)

    def _predict_from_cfg(self, index, model, model_cfg, on_cv, cv_folds, databunch, n_repeats=2, print_metric=True,):
        """
        Description of _predict_from_cfg

        Args:
            index (int):
            model (Class):
            model_cfg (dict):
            cv_folds (int):
            databunch (Class):
            n_repeats=3 (int):
            print_metric=True (bool):

        Returns:
            predict (dict)
            
        """
        model = model._predict_preproc_model(model_cfg=model_cfg, model=model,)
        #print(model_cfg)
        if on_cv:
            res = model.cross_val(
                                X=databunch.X_train[model_cfg['columns']],
                                X_test=databunch.X_test[model_cfg['columns']],
                                model=model, 
                                folds=cv_folds,
                                metric_round=model._metric_round,
                                print_metric=print_metric,
                                n_repeats=n_repeats,
                                predict=True,
                                )
        else:
            res = model._predict_on_full_dataset(
                X=databunch.X_train[model_cfg['columns']],
                X_test=databunch.X_test[model_cfg['columns']],
                model=model,
                )
        predict = {
                    'model_name': f'{index}_' + model_cfg['model_name'], 
                    'predict_test': res['Test_predict'],
                    'predict_train': res['Train_predict'],
                    }
        return(predict)

    def _predict_get_default_model_cfg(self, model):
        if len(model.history_trials_dataframe) < 1:
            config = {
            'score_opt': 0,
            'model_score': 0,
            'score_std': 0,
            'model_name': model.__name__,
            'model_param': model.model_param,
            'wrapper_params': model.wrapper_params,
            'cat_encoder': model._data.cat_encoder_names,
            'columns': model._data.X_train.columns.values,
            'cv_folds': model._cv,
            }
            model_cfgs = pd.DataFrame([config,])
        else: 
            model_cfgs = model.history_trials_dataframe.head(1)
        return(model_cfgs)

    def predict(self, 
                model=None, 
                databunch=None,
                on_cv=True, 
                cv_folds=None, 
                n_repeats=2, 
                models_cfgs=None, 
                print_metric=True, 
                verbose=1,) -> pd.DataFrame:
        """
        Description of predict

        Args:
            model=None (undefined):
            databunch=None (undefined):
            cv_folds=None (undefined):
            n_repeats=3 (undefined):
            models_cfgs=None (undefined):
            print_metric=True (undefined):
            verbose=1 (int):

        Returns:
            predicts (pd.DataFrame)

        """
        if model is None:
            model = self
        if databunch is None:
            databunch=model._data
        if cv_folds is None:
            cv_folds = model._cv
        if models_cfgs is None:
            models_cfgs = model._predict_get_default_model_cfg(model)

        if verbose > 0:
            disable_tqdm = False
        else: 
            disable_tqdm = True
        predicts = []
        total_tqdm = len(models_cfgs)
        for index, model_cfg in tqdm(models_cfgs.iterrows(), total=total_tqdm, disable=disable_tqdm):
            predict = self._predict_from_cfg(
                index,
                model=model,
                model_cfg=model_cfg,
                on_cv=on_cv,
                cv_folds=cv_folds,
                databunch=databunch, 
                n_repeats=n_repeats, 
                print_metric=print_metric,
                )
            predicts.append(predict)
        self.predicts = predicts
        return(pd.DataFrame(predicts))



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