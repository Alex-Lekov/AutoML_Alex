from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import *
from tqdm import tqdm
from time import time
import sklearn
import optuna
import pandas as pd
import numpy as np
import sys
import warnings
from automl_alex.databunch import DataBunch

predict_proba_metrics = ['roc_auc_score', 'log_loss', 'brier_score_loss']


class ModelBase(object):
    """
    Base class for a specific ML algorithm implementation factory,
    i.e. it defines algorithm-specific hyperparameter space and generic methods for model training & inference
    """
    pbar = 0

    def __init__(self,  
                X_train=None, 
                y_train=None,
                X_test=None,
                y_test=None,
                cat_features=None,
                clean_and_encod_data=True,
                cat_encoder_name='HelmertEncoder',
                target_encoder_name='JamesSteinEncoder',
                clean_nan=True,
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
        
        self._init_wrapper_params(wrapper_params)
        self._init_model_param(model_param)
        # variables
        self._init_variables()
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
                                    cat_encoder_name=cat_encoder_name,
                                    target_encoder_name=target_encoder_name,
                                    clean_nan=clean_nan,
                                    random_state=random_state,)
            else: 
                raise Exception("no Data?")
        self._init_dataset()

    def _init_wrapper_params(self, wrapper_params=None):
        """
        Args:
            params (dict or None): parameters for model.
        """
        raise NotImplementedError("Pure virtual class.")
    
    def _init_model_param(self, model_param=None):
        """
        Args:
            params (dict or None): parameters for model.
        """
        raise NotImplementedError("Pure virtual class.")
    
    def _init_variables(self,):
        self.best_model_wrapper_params = None
        self.best_model_param = None
        self.best_score = 0
        self.best_score_std = 0
        self.model = None
        self.study = None
        self.history_trials = []
    
    def _init_dataset(self,):
        # Y
        self.y_train = self._data.y_train
        self.y_test = self._data.y_test
        # X
        self.X_train = self._data.X_train
        self.X_test = self._data.X_test


    def _fit(self, dataset, weights=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
            y (np.array, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            weights (np.array, shape (n_samples, ) or (n_samples, n_outputs) or None): weights of the data
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
            dataset (modelgym.utils.XYCDataset): the input data,
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

    @staticmethod
    def get_model_opt_params(self, ):
        """
        Return:
            dict from parameter name to hyperopt distribution: default
            parameter space
        """
        raise NotImplementedError("Pure virtual class.")

    def __calc_combined_score_opt__(self, direction, score, score_std):
        if direction == 'maximize':
            score_opt = score - score_std
        else:
            score_opt = score + score_std
        return(score_opt)

    def __auto_parameters_calc(self, possible_iters, early_stoping, verbose=1):
        if verbose > 0: 
                print('Start Auto calibration parameters')

        if possible_iters > 100:
            self._cv = 5
            self._score_cv_folds = 1
            self._opt_lvl = 1
            self._cold_start = possible_iters // 3

        if possible_iters > 300:
            self._score_cv_folds = 2
            self._opt_lvl = 2
            self._cold_start = (possible_iters / self._score_cv_folds) // 5

        if possible_iters > 600:
            self._cv = 10
            self._score_cv_folds = 3
            self._cold_start = (possible_iters / self._score_cv_folds) // 10

        if possible_iters > 900:
            self._opt_lvl = 3
            early_stoping = self._cold_start * 3
        
        if possible_iters > 10000:
            self._opt_lvl = 4
            self._score_cv_folds = 4
            self._cold_start = (possible_iters / self._score_cv_folds) // 20
            early_stoping = self._cold_start * 2

        if possible_iters > 25000:
            self._opt_lvl = 5
            self._score_cv_folds = 5
            self._cold_start = (possible_iters / self._score_cv_folds) // 30
            early_stoping = self._cold_start * 2
        return(early_stoping)

    def _opt_model(self, trial):
        opt_model = self
        opt_model.get_model_opt_params(opt_model, trial=trial)
        return(opt_model)

    def _opt_core(self, timeout, early_stoping, save_to_sqlite, verbose=1):
        # time model
        start_time = time()
        score, score_std = self.cv(score_cv_folds=1)
        iter_time = (time() - start_time)

        if verbose > 0: 
            print(f'One iteration takes ~ {round(iter_time,1)} sec')
        
        possible_iters = timeout // iter_time

        if possible_iters < 100:
            print("Not enough time to find the optimal parameters. \n \
                Possible iters < 100. \n \
                Please, Increase the 'timeout' parameter for normal optimization.")
            raise Exception('Not enough time to find the optimal parameters')

        # Auto_parameters
        if self._auto_parameters:
            early_stoping = self.__auto_parameters_calc(possible_iters, early_stoping, verbose)

        # default model
        score, score_std = self.cv()

        # _combined_score_opt
        if self._combined_score_opt:
            score_opt = self.__calc_combined_score_opt__(self.direction, score, score_std)
        else: 
            score_opt = score

        self.best_score = round(score_opt, self._metric_round)
        self.best_score_std = score_std
        self.best_model_name = self.__name__
        self.best_model_wrapper_params = self.wrapper_params
        self.best_model_param = self.model_param
        self.best_cat_encoder = self._data.cat_encoder_name
        self.best_target_encoder = self._data.target_encoder_name

        if verbose > 0: 
            print('Start optimization with the parameters:')
            print('Score_folds = ', self._score_cv_folds)
            print('Opt_lvl = ', self._opt_lvl)
            print('Cold_start = ', self._cold_start)
            print('Early_stoping = ', early_stoping)
            print('Metric = ', self.metric.__name__)
            print('Direction = ', self.direction)
            print(f'Default model {self.metric.__name__} = {round(self.best_score,4)}')
            print('#'*40)
        
        # OPTUNA objective
        def objective(trial, fast_check=True):
            opt_model = self._opt_model(trial=trial)
            # score
            score, score_std = opt_model.cv()
            # _combined_score_opt
            if self._combined_score_opt:
                score_opt = self.__calc_combined_score_opt__(self.direction, score, score_std)
            else: 
                score_opt = score
            
            score_opt = round(score_opt, self._metric_round)
            
            # best_score
            flag_update_best = False
            if self.direction == 'maximize':
                if score_opt > self.best_score:
                    flag_update_best = True
            else:
                if score_opt < self.best_score:
                    flag_update_best = True

            if flag_update_best:        
                self.best_score = score_opt
                self.best_score_std = score_std
                self.best_model_name = opt_model.__name__
                self.best_model_wrapper_params = opt_model.wrapper_params
                self.best_model_param = opt_model.model_param
                self.best_cat_encoder = self._data.cat_encoder_name
                self.best_target_encoder = self._data.target_encoder_name

            # History trials
            self.history_trials.append({
                'score_opt': score_opt,
                'model_score': score,
                'score_std': score_std,
                'model_name': opt_model.__name__,
                'model_param': opt_model.model_param,
                'wrapper_params': opt_model.wrapper_params,
                'cat_encoder': self._data.cat_encoder_name,
                'target_encoder': self._data.target_encoder_name,
                                })
            
            # verbose
            if verbose == 1:
                if pbar is not None:
                    message = f'{self.best_model_name} Best Score {self.metric.__name__} = {self.best_score} '
                    if self._score_cv_folds > 1:
                        message+=f'+- {self.best_score_std}'
                    pbar.set_postfix_str(message)
                    pbar.update()
            return score_opt
        
        sampler=optuna.samplers.TPESampler(consider_prior=True, 
                                            prior_weight=1.0, 
                                            consider_magic_clip=True, 
                                            consider_endpoints=False, 
                                            n_startup_trials=self._cold_start, 
                                            n_ei_candidates=50, 
                                            seed=self._random_state)
        
        if self.study is None:
            if save_to_sqlite:
                self.study = optuna.create_study(
                    direction=self.direction, 
                    sampler=sampler,
                    study_name = 'automl', 
                    storage = 'sqlite:///study_history.db',
                    load_if_exists = True,
                    )
            else:
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

        with tqdm(
            file=sys.stdout,
            desc="Optimize: ", 
            disable=disable_tqdm,
            ) as pbar:
            try:
                self.study.optimize(
                    objective, 
                    timeout=timeout, 
                    callbacks=[es_callback], 
                    show_progress_bar=False,
                    )
            except EarlyStoppingExceeded:
                if verbose == 1: 
                    print(f'\n EarlyStopping Exceeded: Best Score: {self.study.best_value}', 
                        self.metric.__name__)
        
        self.history_trials_dataframe = pd.DataFrame(self.history_trials).sort_values('model_score', ascending=True)
        if self.direction == 'maximize':
            self.history_trials_dataframe = pd.DataFrame(self.history_trials).sort_values('model_score', ascending=False)
        return(self.history_trials_dataframe)

    def opt(self,
            timeout=100,
            auto_parameters=None,
            cv=None,
            cold_start=None,
            score_cv_folds=None,
            opt_lvl=None,
            direction=None,
            early_stoping=100,
            save_to_sqlite=False,
            verbose=1,):
        if cv is not None:
            self._cv = cv
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
            raise Exception('Need direction for optimaze!')

        history = self._opt_core(
            timeout, 
            early_stoping, 
            save_to_sqlite,
            verbose)

        self.wrapper_params = self.best_model_wrapper_params
        self.model_param = self.best_model_param
        return(history)

    def plot_opt_history(self, figsize=(15,5)):
        """
        Plot optimization history of all trials in a study.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style(style="darkgrid")
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

    def _norm_data(self, train_x, val_x, test_x=None):
        # Scaler
        if self.wrapper_params['need_norm_data']: 
            train_x, val_x, test_x = self._data.use_scaler(train_x,
                                                    val_x=val_x,
                                                    test_x=test_x,
                                                    name=self.wrapper_params['scaler_name'],)
        return(train_x, val_x, test_x)

    def cv(self, print_metric=False, metric_round=4, predict=False, score_cv_folds=None):
        if score_cv_folds is None:
            score_cv_folds = self._score_cv_folds
        if self.type_of_estimator == 'classifier':
            skf = StratifiedKFold(n_splits=self._cv, shuffle=True, random_state=self._random_state,)
        else:
            skf = KFold(n_splits=self._cv, shuffle=True, random_state=self._random_state,)

        folds_scores = []
        if predict:
            stacking_y_pred_train = np.zeros(self.X_train.shape[0])
            stacking_y_pred_test = np.zeros(self.X_test.shape[0])

        count=0
        for train_idx, valid_idx in skf.split(self.X_train, self.y_train):
            count+=1
            if not predict:
                if count > score_cv_folds:
                    break

            train_x, train_y = self.X_train.iloc[train_idx], self.y_train.iloc[train_idx]
            val_x, val_y = self.X_train.iloc[valid_idx], self.y_train.iloc[valid_idx]
            
            # TargetEncoder and Scaling
            train_x, val_x, test_x = self._data.target_encodet(train_x=train_x, train_y=train_y, 
                                                        val_x=val_x, test_x=self.X_test)
            train_x, val_x, test_x = self._norm_data(train_x, val_x, test_x=test_x)
            #print(train_x.shape, val_x.shape, test_x.shape)
            # Fit
            self._fit(
                train_x.reset_index(drop=True), 
                train_y.reset_index(drop=True), 
                val_x.reset_index(drop=True), 
                val_y.reset_index(drop=True))

            # Predict
            if (self.metric.__name__ in predict_proba_metrics) and (self.is_possible_predict_proba()):
                y_pred = self._predict_proba(val_x)
                if predict:
                    y_pred_test = self._predict_proba(test_x)
            else:
                y_pred = self._predict(val_x)
                if predict:
                    y_pred_test = self._predict(test_x)

            if predict:
                stacking_y_pred_train[valid_idx] = y_pred
                stacking_y_pred_test += y_pred_test/self._cv

            score_model = self.metric(val_y, y_pred)
            folds_scores.append(score_model)
        
        if score_cv_folds > 1 or predict:
            score = round(np.mean(folds_scores),self._metric_round)
            score_std = round(np.std(folds_scores),self._metric_round+2)
        else:
            score = round(score_model, self._metric_round)
            score_std = 0

        if print_metric:
            print(f'Mean Score {self.metric.__name__} on {self._cv} Folds: {score} std: {score_std}')

        if predict:
            return(stacking_y_pred_test, stacking_y_pred_train,)
        else:
            return(score, score_std)

    def fit_predict(self, print_metric=True,):
        return(self.cv(print_metric=print_metric, predict=True))

    def _need_scaler_params(self, trial):
        """
        Return:
            dict of DistributionWrappers
        """
        self.wrapper_params['scaler_name'] = trial.suggest_categorical('scaler_name', [
                                                                            'RobustScaler', 
                                                                            'StandardScaler', 
                                                                            'MinMaxScaler', 
                                                                            ])


class ModelClassifier(ModelBase):
    type_of_estimator='classifier'
    __name__ = 'ModelClassifier'


class ModelRegressor(ModelBase):
    type_of_estimator='regression'
    __name__ = 'ModelRegressor'


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
            if self.early_stop_count > self.early_stop:
                self.early_stop_count = 0
                self.best_score = None
                raise EarlyStoppingExceeded()
            else:
                self.early_stop_count=self.early_stop_count+1
        return
    
    def early_stopping_opt_minimize(self, study, trial):
        if self.best_score is None:
            self.best_score = study.best_value

        if study.best_value < self.best_score:
            self.best_score = study.best_value
            self.early_stop_count = 0
        else:
            if self.early_stop_count > self.early_stop:
                self.early_stop_count = 0
                self.best_score = None
                raise EarlyStoppingExceeded()
            else:
                self.early_stop_count=self.early_stop_count+1
        return