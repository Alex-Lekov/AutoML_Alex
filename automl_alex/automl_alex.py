from sklearn.metrics import *
import sklearn
from tqdm import trange
from tqdm import tqdm
import optuna
import pandas as pd
import numpy as np
import time
import sys
from .models import *
from .databunch import DataBunch
from .encoders import *


class AutoMLBase(object):
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
                auto_parameters=True,
                cv=10,
                score_cv_folds=3, # how many folds are actually used
                opt_lvl=3,
                metric=None,
                direction=None,
                combined_score_opt=True,
                metric_round=4, 
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
        self._gpu = gpu
        self._random_state = random_state
        self._score_cv_folds = score_cv_folds
        self._opt_lvl = opt_lvl
        self._combined_score_opt = combined_score_opt
        
        # variables
        self._init_variables()
        # dataset
        if databunch: self._data = databunch
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
            else: raise Exception("no Data?")
        #self._init_dataset(self._data)
    
    def _init_variables(self,):
        self.best_model_wrapper_params = None
        self.best_model_param = None
        self.best_score = 0
        self.best_score_std = 0
        self.model = None
        self.study = None
        self.history_trials = []
    
    def __calc_combined_score_opt__(self, direction, score, score_std):
        if direction == 'maximize':
            score_opt = score - score_std
        else:
            score_opt = score + score_std
        return(score_opt)

    def predict(self, dataset):
        """
        Args:
            dataset (modelgym.utils.XYCDataset): the input data,
                dataset.y may be None
        Return:
            np.array, shape (n_samples, ): predictions
        """
        raise NotImplementedError("Pure virtual class.")


##################################### BestSingleModel ################################################


class BestSingleModel(AutoMLBase):
    """
    Trying to find which model work best on our data
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """
    def __auto_parameters_calc(self, possible_iters, early_stoping, verbose=1):
        if verbose > 0: 
                print('Start Auto calibration parameters')

        if possible_iters > 100:
            self._cv = 5
            self._score_cv_folds = 1
            self._opt_lvl = 1
            self._cold_start = possible_iters // 3

        if possible_iters > 300:
            self._cv = 10
            self._score_cv_folds = 2
            self._cold_start = (possible_iters / self._score_cv_folds) // 5
            self._opt_lvl = 2

        if possible_iters > 1000:
            self._opt_lvl = 3
            self._score_cv_folds = 3
            self._cold_start = (possible_iters / self._score_cv_folds) // 10

        if possible_iters > 5000:
            early_stoping = self._cold_start + 25
        
        if possible_iters > 10000:
            self._opt_lvl = 4
            self._score_cv_folds = 4
            self._cold_start = (possible_iters / self._score_cv_folds) // 20
            early_stoping = self._cold_start + 50

        if possible_iters > 25000:
            self._opt_lvl = 5
            self._score_cv_folds = 5
            self._cold_start = (possible_iters / self._score_cv_folds) // 30
            early_stoping = self._cold_start + 100
        return(early_stoping)

    def opt(self, 
        timeout=1000, 
        early_stoping=100, 
        cold_start=100,
        direction='maximize',
        verbose=1,
        opt_lvl=3,
        cv=None,
        score_cv_folds=None,
        auto_parameters=True,
        models_names=None,
        cat_encoder_names=None,
        target_cat_encoder_names=None,
        ):
        if cv is not None:
            self._cv = cv
        if score_cv_folds is not None:
            self._score_cv_folds = score_cv_folds

        if models_names is None:
            self.models_names = all_models.keys()
        else:
            self.models_names = models_names

        if cold_start is not None:
            self._cold_start = cold_start
        if opt_lvl is not None:
            self._opt_lvl = opt_lvl
        if self.direction is None:
            self.direction = direction

        if cat_encoder_names is None:
            self.encoders_names = encoders_names.keys()
        else:
            self.encoders_names = cat_encoder_names
            
        if target_cat_encoder_names is None:
            self.target_encoder_names = target_encoders_names.keys()
        else:
            self.target_encoder_names = target_cat_encoder_names

        # Setup best_score
        start_model_name = list(self.models_names)[0]
        model = all_models[start_model_name](databunch=self._data,
                                            cv=self._cv,
                                            score_cv_folds=3,
                                            metric=self.metric,
                                            metric_round=self._metric_round,
                                            gpu=self._gpu, 
                                            random_state=self._random_state,
                                            type_of_estimator= self.type_of_estimator)
        # score
        # time model
        start_time = time.time()
        score, score_std = model.cv()
        iter_time = (time.time() - start_time)/3
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
        
        # _combined_score_opt
        if self._combined_score_opt:
            score_opt = self.__calc_combined_score_opt__(self.direction, score, score_std)
        else: 
            score_opt = score

        score_opt = round(score_opt, self._metric_round)
        self.best_score = score_opt
        self.best_score_std = score_std
        self.best_model_name = start_model_name
        self.best_model_wrapper_params = model.wrapper_params
        self.best_model_param = model.model_param
        self.best_cat_encoder = self._data.cat_encoder_name
        self.best_target_encoder = self._data.target_encoder_name

        if verbose > 0: 
            print('Start optimization with the parameters:')
            print('CV = ', self._cv)
            print('Score_folds = ', self._score_cv_folds)
            print('Opt_lvl = ', self._opt_lvl)
            print('Cold_start = ', self._cold_start)
            print('Early_stoping = ', early_stoping)
            print('Metric = ', self.metric.__name__)
            print('Direction = ', self.direction)
            print(f'Start Model {start_model_name} Score {self.metric.__name__} = {self.best_score}')


        def objective(trial, fast_check=True):
            # Encoders
            encoder_name = trial.suggest_categorical('cat_encoder_name', self.encoders_names)
            target_encoder_name = trial.suggest_categorical('target_cat_encoder_name', self.target_encoder_names)
            
            self._data = DataBunch(X_train=self._data.X_train_source, 
                                    y_train=self._data.y_train,
                                    X_test=self._data.X_test_source,
                                    y_test=self._data.y_test,
                                    cat_features=self._data.cat_features,
                                    clean_and_encod_data=True,
                                    cat_encoder_name=encoder_name,
                                    target_encoder_name=target_encoder_name,
                                    clean_nan=True,
                                    random_state=self._random_state,)

            # Model
            model_name = trial.suggest_categorical('model_name', self.models_names)
            
            model = all_models[model_name](databunch=self._data, 
                                            opt_lvl=self._opt_lvl,
                                            cv=self._cv,
                                            score_cv_folds = self._score_cv_folds,
                                            auto_parameters = False,
                                            metric=self.metric,
                                            metric_round=self._metric_round,
                                            gpu=self._gpu, 
                                            random_state=self._random_state,
                                            type_of_estimator= self.type_of_estimator)
            model.get_model_opt_params(model, trial=trial)

            # score
            score, score_std = model.cv()

            # _combined_score_opt
            if self._combined_score_opt:
                score_opt = self.__calc_combined_score_opt__(self.direction, score, score_std)
            else: 
                score_opt = score
            
            score_opt = round(score_opt, self._metric_round)
            
            # best_score
            flag_update_best = False
            if direction == 'maximize':
                if score_opt > self.best_score:
                    flag_update_best = True
            else:
                if score_opt < self.best_score:
                    flag_update_best = True

            if flag_update_best:        
                self.best_score = score_opt
                self.best_score_std = score_std
                self.best_model_name = model_name
                self.best_model_wrapper_params = model.wrapper_params
                self.best_model_param = model.model_param
                self.best_cat_encoder = self._data.cat_encoder_name
                self.best_target_encoder = self._data.target_encoder_name

            # History trials
            self.history_trials.append({
                'score_opt': score_opt,
                'model_score': score,
                'score_std': score_std,
                'model_name': model_name,
                'model_param': model.model_param,
                'wrapper_params': model.wrapper_params,
                'cat_encoder': self._data.cat_encoder_name,
                'target_encoder': self._data.target_encoder_name,
                                })

            if verbose == 1:
                if pbar is not None:
                    message = f'Best Score {self.metric.__name__} = {self.best_score} '
                    if self._score_cv_folds > 1:
                        message+=f'+- {self.best_score_std} '
                    message+=f'Best Model: {self.best_model_name}'
                    pbar.set_postfix_str(message)
                    pbar.update(1)
            return round(score_opt, self._metric_round)
        
        sampler=optuna.samplers.TPESampler(consider_prior=True, 
                                            prior_weight=1.0, 
                                            consider_magic_clip=True, 
                                            consider_endpoints=False, 
                                            n_startup_trials=self._cold_start, 
                                            n_ei_candidates=50, 
                                            seed=self._random_state)
        self.study = optuna.create_study(direction=self.direction, sampler=sampler)

        if verbose < 2:
            optuna.logging.disable_default_handler()
        
        # EarlyStopping
        es = EarlyStoppingExceeded()
        es.early_stop = early_stoping
        es.early_stop_count = 0
        es.best_score = None

        es_callback = es.early_stopping_opt_minimize
        if self.direction == 'maximize':
            es_callback = es.early_stopping_opt_maximize

        # Opt
        if verbose > 0:
            disable_tqdm = False
        else: disable_tqdm = True

        with tqdm(file=sys.stdout, leave=False, disable=disable_tqdm) as pbar:
            try:
                self.study.optimize(objective, timeout=timeout, callbacks=[es_callback])
            except EarlyStoppingExceeded:
                if verbose == 1:
                    print(f'\n EarlyStopping Exceeded: Best Score: {self.study.best_value}', self.metric.__name__)

        self.history_trials_dataframe = pd.DataFrame(self.history_trials).sort_values('score_opt', ascending=True)
        if self.direction == 'maximize':
            self.history_trials_dataframe = pd.DataFrame(self.history_trials).sort_values('score_opt', ascending=False)
        
        #self.wrapper_params = self.best_model_wrapper_params
        #self.model_param = self.best_model_param
        self._data.cat_encoder_name = self.best_cat_encoder
        self._data.target_encoder_name = self.best_target_encoder
        return(self.history_trials_dataframe)

    def predict(self, 
        model_name=None,
        cat_encoder_name=None,
        target_encoder_name=None,
        model_param=None,
        wrap_params=None,
        print_metric=False,
        ):
        if model_name is None:
            if self.best_model_name is None:
                raise Exception("No best model")
            model_name = self.best_model_name
        if cat_encoder_name is None:
            cat_encoder_name = self.best_cat_encoder
        if target_encoder_name is None:
            target_encoder_name = self.best_target_encoder

        self._model_param = model_param
        if model_param is None:
            self._model_param = self.best_model_param

        self._wrapper_params = wrap_params
        if wrap_params is None:
            self._wrapper_params = self.best_model_wrapper_params 

        self._data = DataBunch(X_train=self._data.X_train_source, 
                                    y_train=self._data.y_train,
                                    X_test=self._data.X_test_source,
                                    y_test=self._data.y_test,
                                    cat_features=self._data.cat_features,
                                    clean_and_encod_data=True,
                                    cat_encoder_name=cat_encoder_name,
                                    target_encoder_name=target_encoder_name,
                                    clean_nan=True,
                                    random_state=self._random_state,)

        model = all_models[model_name](databunch=self._data, 
                                        cv = self._cv,
                                        score_cv_folds = self._score_cv_folds,
                                        metric=self.metric,
                                        combined_score_opt=self._combined_score_opt,
                                        metric_round=self._metric_round,
                                        model_param=self._model_param, 
                                        wrapper_params=self._wrapper_params, 
                                        gpu=self._gpu, 
                                        random_state=self._random_state,
                                        type_of_estimator=self.type_of_estimator)
        
        predicts_test, predict_train = model.cv(metric_round=self._metric_round, 
                                                print_metric=print_metric,
                                                predict=True,)
        return(predicts_test, predict_train)
    
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


class BestSingleModelClassifier(BestSingleModel):
    type_of_estimator='classifier'
    __name__ = 'BestSingleModelClassifier'


class BestSingleModelRegressor(BestSingleModel):
    type_of_estimator='regression'
    __name__ = 'BestSingleModelRegressor'


##################################### ModelsReview ################################################


class ModelsReview(AutoMLBase):
    '''
    ModelsReview...
    '''
    __name__ = 'ModelsReview'

    def fit(self, 
        models_names=None,
        verbose=1,
        ):
        if models_names is None:
            self.models_names = all_models.keys()
        else:
            self.models_names = models_names
        
        self.review_models_dataframe = pd.DataFrame()
        review_models = []

        if verbose > 0:
            disable_tqdm = False
        else: disable_tqdm = True
        
        pbar = tqdm(self.models_names, disable=disable_tqdm)
        for model_name in pbar:
            start_unixtime = time.time()
            # Model
            model = all_models[model_name](databunch=self._data, 
                                            cv=self._cv,
                                            score_cv_folds = self._cv,
                                            metric=self.metric,
                                            metric_round=self._metric_round,
                                            combined_score_opt=self._combined_score_opt,
                                            gpu=self._gpu, 
                                            random_state=self._random_state,
                                            type_of_estimator=self.type_of_estimator)
            # score
            score, score_std = model.cv()

            # _combined_score_opt
            if self._combined_score_opt:
                if self.direction == 'maximize':
                    score_opt = score - score_std
                else:
                    score_opt = score + score_std
            else: 
                score_opt = score
            
            score_opt = round(score_opt, self._metric_round)

            review_models.append({
                'score_opt': score_opt,
                'model_score': score,
                'score_std': score_std,
                'model_name': model_name,
                'model_param': model.model_param,
                'wrapper_params': model.wrapper_params,
                'cat_encoder': self._data.cat_encoder_name,
                'target_encoder': self._data.target_encoder_name,
                                })
            
            if verbose >= 1:
                if pbar is not None:
                    message = f'{model_name} Score {self.metric.__name__} = {score} '
                    if self._score_cv_folds > 1:
                        message+=f'+- {score_std}'
                    pbar.set_postfix_str(message)
                    pbar.update(1)
            
        self.review_models_dataframe = pd.DataFrame(review_models)
        return(self.review_models_dataframe)

    def opt(self, 
        timeout=1000, 
        early_stoping=100, 
        auto_parameters=True,
        direction=None,
        verbose=1,
        models_names=None,
        ):
        if auto_parameters is not None:
            self._auto_parameters = auto_parameters
        if direction is not None:
            self.direction = direction
        if self.direction is None:
            raise Exception('Need direction for optimaze!')

        if models_names is None:
            self.models_names = all_models.keys()
        else:
            self.models_names = models_names

        self.history_trials_dataframe = pd.DataFrame()
        self.top1_models_cfgs = pd.DataFrame()
        self.top10_models_cfgs = pd.DataFrame()

        timeout_per_model = timeout//len(self.models_names)
        
        if verbose > 0:
            disable_tqdm = False
        else: disable_tqdm = True
        
        pbar = tqdm(self.models_names, disable=disable_tqdm)
        for model_name in pbar:
            start_unixtime = time.time()
            # Model
            model = all_models[model_name](databunch=self._data, 
                                            opt_lvl=self._opt_lvl,
                                            cv=self._cv,
                                            score_cv_folds = self._score_cv_folds,
                                            auto_parameters = self._auto_parameters,
                                            metric=self.metric,
                                            direction=self.direction,
                                            metric_round=self._metric_round,
                                            combined_score_opt=self._combined_score_opt,
                                            gpu=self._gpu, 
                                            random_state=self._random_state,
                                            type_of_estimator=self.type_of_estimator)
            # Opt
            _ = model.opt(timeout=timeout_per_model,
                                early_stoping=early_stoping, 
                                verbose= (lambda x: 0 if x <= 1 else 1)(verbose),
                                )
            # Trials
            self.history_trials_dataframe = self.history_trials_dataframe.append(other=model.history_trials_dataframe, 
                                                                    ignore_index=True)

            # Top1:
            self.top1_models_cfgs = self.top1_models_cfgs.append(other=model.history_trials_dataframe.head(1), 
                                                                    ignore_index=True)

            # Top10:
            self.top10_models_cfgs = self.top10_models_cfgs.append(other=model.history_trials_dataframe.head(10), 
                                                                    ignore_index=True)

            if verbose > 0:
                message = f'Model: {model_name} | Best Score {self.metric.__name__} = {model.best_score} '
                if self._score_cv_folds > 1:
                    message+=f'+- {model.best_score_std} '
                pbar.set_postfix_str(message)
                pbar.update(1)

            # time dinamic
            sec_iter = start_unixtime - time.time()
            sec_dev = timeout_per_model - sec_iter
            if sec_dev > 10:
                timeout_per_model = timeout_per_model + (sec_dev // (len(self.models_names)))
            
        return(self.top1_models_cfgs)
    
    
    def predict(self, best_models_cfgs=None, get_pred_train=True, print_metric=False, verbose=1,):
        if best_models_cfgs is None:
            if self.top1_models_cfgs is None:
                raise Exception("No best models")
            else: best_models_cfgs = self.top1_models_cfgs
        
        if verbose > 0:
            disable_tqdm = False
        else: disable_tqdm = True
        
        predicts = []
        total_tqdm = len(best_models_cfgs)
        for index, model_cfg in tqdm(best_models_cfgs.iterrows(), total=total_tqdm, disable=disable_tqdm):

            # Model
            model = all_models[model_cfg['model_name']](databunch=self._data, 
                                                        model_param=model_cfg['model_param'],
                                                        wrapper_params=model_cfg['wrapper_params'],
                                                        cv=self._cv,
                                                        metric=self.metric,
                                                        metric_round=self._metric_round,
                                                        gpu=self._gpu, 
                                                        random_state=self._random_state,
                                                        type_of_estimator=self.type_of_estimator)

            # Predict
            predict_test, predict_train = model.cv(
                                                metric_round=self._metric_round,
                                                print_metric=print_metric,
                                                predict=True,
                                                )
            
            predict = {
                        'model_name': f'{index}_' + model_cfg['model_name'], 
                        'predict_test': predict_test,
                        'predict_train': predict_train,
                        }
            predicts.append(predict)
            #counter+=1
            
        self.predicts = predicts
        return(predicts)


class ModelsReviewClassifier(ModelsReview):
    type_of_estimator='classifier'
    __name__ = 'ModelsReviewClassifier'


class ModelsReviewRegressor(ModelsReview):
    type_of_estimator='regression'
    __name__ = 'ModelsReviewRegressor'


##################################### Stacking #########################################


class Stacking(ModelsReview):
    '''
    Stack top models
    '''
    __name__ = 'Stacking'

    def opt(self, 
            timeout=1000,
            early_stoping=100,
            cold_start=None,
            cv=None,
            score_cv_folds=None,
            stack_models_names=None,
            meta_models_names=['MLP', 'XGBoost'],
            cat_encoder_names=None,
            target_cat_encoder_names=None,
            meta_model_cv=12,
            verbose=1,):

        if cv is not None:
            self._cv = cv
        if score_cv_folds is not None:
            self._score_cv_folds = score_cv_folds

        if self._cv < 2:
            raise Exception("Stacking no CV? O_o")
        
        if verbose > 0:
            print("\n Step1: Opt StackingModels")
            time.sleep(0.2) # clean print 

        if timeout < 600:
            raise Exception(f"opt Stacking in {timeout}sec? it does not work so fast)")
        select_models_timeout = timeout-300

        if stack_models_names is None:
            self.stack_models_names = all_models.keys()
        else:
            self.stack_models_names = stack_models_names
        self.meta_models_names = meta_models_names

        if cat_encoder_names is None:
            self.encoders_names = encoders_names.keys()
        else:
            self.encoders_names = cat_encoder_names
            
        if target_cat_encoder_names is None:
            self.target_encoder_names = target_encoders_names.keys()
        else:
            self.target_encoder_names = target_cat_encoder_names
    
        # Model
        model = BestSingleModel(databunch=self._data,
                                opt_lvl=self._opt_lvl,
                                cv=self._cv,
                                score_cv_folds = self._score_cv_folds,
                                auto_parameters = self._auto_parameters,
                                metric=self.metric,
                                direction=self.direction,
                                metric_round=self._metric_round,
                                combined_score_opt=self._combined_score_opt,
                                gpu=self._gpu, 
                                random_state=self._random_state,
                                type_of_estimator=self.type_of_estimator)

        # Opt
        history = model.opt(timeout=select_models_timeout, 
            models_names=self.stack_models_names,
            cat_encoder_names=self.encoders_names,
            target_cat_encoder_names=self.target_encoder_names,
            verbose= (lambda x: 0 if x <= 1 else 1)(verbose), )

        history = history.drop_duplicates(subset=['model_score', 'score_std'], keep='last')
        self.stack_models_trails = history.head(25)

        if verbose > 0:
            print("\n Step2: Get new X_train from StackingModels")
            time.sleep(0.2) # clean print 
        # Predict
        stack_predicts_tests = []
        stack_predict_trains = []
        for i in trange(len(self.stack_models_trails)):
            predicts_test, predict_train = model.predict(
                model_name=self.stack_models_trails.iloc[i]['model_name'],
                cat_encoder_name=self.stack_models_trails.iloc[i]['cat_encoder'],
                target_encoder_name=self.stack_models_trails.iloc[i]['target_encoder'],
                model_param=self.stack_models_trails.iloc[i]['model_param'],
                wrap_params=self.stack_models_trails.iloc[i]['wrapper_params'],
                )
            stack_predicts_tests.append(predicts_test)
            stack_predict_trains.append(predict_train)
        
        # get new X_train
        self._data.X_train_predicts = pd.DataFrame(stack_predict_trains).T
        self._data.X_test_predicts = pd.DataFrame(stack_predicts_tests).T
        
        # Score:
        score_mean_models = self.metric(self._data.y_train, pd.DataFrame(stack_predict_trains).mean())
        if verbose > 0:
            print(f'\n StackModels Mean {self.metric.__name__} Score Train: ', 
                round(score_mean_models, self._metric_round))
            print("\n Step3: Opt MetaModels")
            time.sleep(0.1) # clean print 

        meta_data = DataBunch(X_train=self._data.X_train_predicts, 
                            y_train=self._data.y_train,
                            X_test=self._data.X_test_predicts,
                            y_test=None,
                            cat_features=None,
                            clean_and_encod_data=False,
                            cat_encoder_name=None,
                            target_encoder_name=None,
                            clean_nan=False,
                            random_state=self._random_state,)
        
        # Meta model
        meta_model = ModelsReview(databunch=meta_data,
                                opt_lvl=self._opt_lvl,
                                cv=meta_model_cv,
                                score_cv_folds = meta_model_cv,
                                auto_parameters = False,
                                metric=self.metric,
                                direction=self.direction,
                                metric_round=self._metric_round,
                                combined_score_opt=self._combined_score_opt,
                                gpu=self._gpu, 
                                random_state=self._random_state,
                                type_of_estimator=self.type_of_estimator)
        
        _ = meta_model.opt(timeout=300,
                            models_names=self.meta_models_names,
                            auto_parameters=False,
                            verbose= (lambda x: 0 if x <= 1 else 1)(verbose),)

        self.top10_meta_models = meta_model.top10_models_cfgs.sort_values('model_score', ascending=True).head(10)
        if self.direction == 'maximize':
            self.top10_meta_models = meta_model.top10_models_cfgs.sort_values('model_score', ascending=False).head(10)
        
        # Predict
        if verbose > 0:
            print("\n Step4: Predict from MetaModels")
            time.sleep(0.1) # clean print 

        predict_meta_models = meta_model.predict(best_models_cfgs=self.top10_meta_models, verbose=verbose,)

        stacking_y_pred_test = pd.DataFrame(predict_meta_models)['predict_test'].mean()
        stacking_y_pred_train = pd.DataFrame(predict_meta_models)['predict_train'].mean()
        
        if verbose > 0:
            # Score:
            score_mean_meta_models = self.metric(self._data.y_train, stacking_y_pred_train)
            print(f'MetaModels Mean {self.metric.__name__} Score Train : {score_mean_meta_models}')
            print("\n Finish!")

        return (stacking_y_pred_test, stacking_y_pred_train)

    fit_predict = opt
    

class StackingClassifier(Stacking):
    type_of_estimator='classifier'
    __name__ = 'StackingClassifier'


class StackingRegressor(Stacking):
    type_of_estimator='regression'
    __name__ = 'StackingRegressor'



##################################### AutoML #########################################

class AutoML(Stacking):
    '''
    in progress AutoML
    '''
    pass


class AutoMLClassifier(AutoML):
    type_of_estimator='classifier'
    __name__ = 'AutoMLClassifier'


class AutoMLRegressor(AutoML):
    type_of_estimator='regression'
    __name__ = 'AutoMLRegressor'
