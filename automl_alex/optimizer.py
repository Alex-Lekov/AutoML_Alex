
import pandas as pd
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import optuna
import sklearn

from .logger import *
from .models import *
from .cross_validation import CrossValidation

optuna.logging.disable_default_handler()

############################################
# IN progress....
class Optimizer(object):
    '''
    In this library, I tried to simplify the process of optimizing and iterating parameters as much as possible. 
    You just specify how much time you are willing to spend on optimization, 
    and the library will select the rules for cross-validation, optimization depth, 
    and other optimization parameters based on this time. 
    The more time you give it, the better it will be able to select parameters.

    Automatic hyperparameter optimization based on Optuna (https://optuna.org/) but with some modifications.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit model.

    timeout : int, default=200
        Optimization time in seconds

    metric : Class, default=None 
        you can use standard metrics from sklearn.metrics or add custom metrics.
        If None, the metric is selected from the type of estimator:
        'classifier': sklearn.metrics.roc_auc_score
        'regression': sklearn.metrics.mean_squared_error

    metric_round : int, default=4

    combined_score_opt : bool, default=True
        score_opt = score - std

    iterations : int, default=None
        The number of optimization iterations. If this argument is not given, as many trials run as possible in timeout

    cold_start : int, default=30
        In the cold_start parameter, we specify how many iterations we give for this warm-up. 
        before the algorithm starts searching for optimal parameters, it must collect statistics on the represented space.
        this is why it starts in the randomsample solution at the beginning. 
        The longer it works in this mode , the less likely it is to get stuck in the local minimum. 
        But if you take too long to warm up, you may not have time to search with a more "smart" algorithm. 
        Therefore, it is important to maintain a balance.

    opt_lvl : int, default=2 
        by limiting the optimization time, we will have to choose how deep we should optimize the parameters. 
        Perhaps some parameters are not so important and can only give a fraction of a percent. 
        By setting the opt_lvl parameter, you control the depth of optimization.
        in the code automl_alex.models.model_lightgbm.LightGBM you can find how parameters are substituted for iteration
    
    early_stoping : int, default=100
        stop optimization if no better parameters are found through iterations
    
    auto_parameters : bool, default=True
        If we don't want to select anything, we just set auto_parameters=True. Then the algorithm itself will select, depending on the time allotted to it, the optimal values for:
            *folds
            *score_folds
            *cold_start
            *opt_lvl
    
    feature_selection : bool, default=True
        add feature_selection in optimization
    
    verbose : int, default=1
    
    random_state : int, default=42
        RandomState instance
        Controls the generation of the random states for each repetition.
    '''
    __name__ = 'Optimizer'

    def __init__(
        self,
        estimator,
        folds=10,
        score_folds=2,
        metric=None,
        metric_round=4, 
        cold_start=25,
        opt_lvl=2,
        early_stoping=50,
        auto_parameters=True,
        feature_selection=True,
        random_state=42,
        ):
        self._random_state = random_state
        self.estimator = estimator
        self.folds = folds
        self.score_folds = score_folds
        self.metric_round = metric_round
        self._auto_parameters = auto_parameters
        self.cold_start = cold_start
        self.opt_lvl = opt_lvl
        self.early_stoping = early_stoping
        self.feature_selection = feature_selection


        if metric is None:
            logger.info('metric is None! Default metric will be used. classifier: AUC, regression: MSE')
            if estimator.type_of_estimator == 'classifier':
                self.metric = sklearn.metrics.roc_auc_score
            elif estimator.type_of_estimator == 'regression':
                self.metric = sklearn.metrics.mean_squared_error
        else:
            self.metric = metric


    def __metric_direction_detected__(self, metric, y):
        zero_y = np.zeros(len(y))
        zero_score = metric(y, zero_y)
        best_score = metric(y, y)

        if best_score > zero_score:
            direction = 'maximize'
        else:
            direction = 'minimize'
        return(direction)

    
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


    def __auto_parameters_calc__(self, possible_iters,):
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
        early_stoping = 25
        folds = 5
        score_folds = 1
        opt_lvl = 1
        cold_start = 10
            
        if possible_iters > 100:
            opt_lvl = 2
            folds = 10
            score_folds = 2
            cold_start = 20
            early_stoping = 30

        if possible_iters > 500:
            opt_lvl = 3
            score_folds = 3
            cold_start = 25
            early_stoping = cold_start * 2

        if possible_iters > 800:
            opt_lvl = 4
            score_folds = 4
            cold_start = 40
            early_stoping = cold_start * 2
        
        if possible_iters > 1500:
            opt_lvl = 5
            score_folds = 5
            cold_start = 50
            early_stoping = cold_start * 2

        return(early_stoping, folds, score_folds, opt_lvl, cold_start,)


    def _print_opt_parameters(self,):
        logger.info('> Start optimization with the parameters:')
        logger.info(f'CV_Folds = {self.folds}')
        logger.info(f'Score_CV_Folds = {self.score_folds}')
        logger.info(f'Feature_Selection = {self.feature_selection}')
        logger.info(f'Opt_lvl = {self.opt_lvl}')
        logger.info(f'Cold_start = {self.cold_start}')
        logger.info(f'Early_stoping = {self.early_stoping}')
        logger.info(f'Metric = {self.metric.__name__}')
        logger.info(f'Direction = {self.direction}')


    def _tqdm_opt_print(self, pbar, score_opt, pruned=False):
        """
        Printing information in tqdm. Use pbar. 
        See the documentation for tqdm: https://github.com/tqdm/tqdm
        """
        if pbar is not None:
            self.best_score = self.study.best_value

            message = f' | OptScore: {score_opt} | Best {self.metric.__name__}: {self.best_score} '
            if pruned:
                message+=f'| Trail Pruned! '
            pbar.set_postfix_str(message)
            pbar.update(1)


    def _set_opt_info(self, model, timeout):
        self.study.set_user_attr("Model", model.__name__)
        self.study.set_user_attr("Type_estimator", model.type_of_estimator)
        self.study.set_user_attr("Metric", self.metric.__name__,)
        self.study.set_user_attr("direction", self.direction,)
        self.study.set_user_attr("Timeout", timeout)
        self.study.set_user_attr("auto_parameters", self._auto_parameters)
        self.study.set_user_attr("early_stoping", self.early_stoping)
        self.study.set_user_attr("cold_start", self.cold_start)
        self.study.set_user_attr("opt_lvl", self.opt_lvl,)
        self.study.set_user_attr("Folds", self.folds)
        self.study.set_user_attr("Score_folds", self.score_folds,)
        self.study.set_user_attr("Opt_lvl", self.opt_lvl,)
        self.study.set_user_attr("random_state", self._random_state,)

    
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


    @logger.catch
    def opt(
        self,
        X,
        y,
        timeout=600, # optimization time in seconds
        verbose=0,
        ):
        """
        Description of opt:
        in progress... 

        Args:
        X (pd.DataFrame, shape (n_samples, n_features)): the input X_train data
        y (pd.DataFrame or np.array, shape (n_samples)): Targets
        timeout=100 (int): optimization time in seconds
        verbose=0 (int):
        
        Returns:
            model (Class)
        """
        start_opt_time = time.time()
        self.study = None

        self.verbose = verbose
        logger_print_lvl(self.verbose)

        if verbose > 0:
            disable_tqdm = False
        else:
            disable_tqdm = True

        if self.metric is not None:
            self.direction = self.__metric_direction_detected__(self.metric, y)

        model = self.estimator

        ###############################################################################
        # Optuna EarlyStoppingExceeded 
        es = EarlyStoppingExceeded()
        es.early_stop = self.early_stoping
        es.early_stop_count = 0
        es.best_score = None

        es_callback = es.early_stopping_opt_minimize
        if self.direction == 'maximize':
            es_callback = es.early_stopping_opt_maximize
        
        ###############################################################################
        if self._auto_parameters:
            # time 1 iter
            start_time = time.time()
            model = model.fit(X, y)
            iter_time = (time.time() - start_time)
            logger.info(f'One iteration takes ~ {round(iter_time,1)} sec')

            possible_iters = timeout // (iter_time)
            if possible_iters < 100:
                logger.warning("Not enough time to find the optimal parameters. \n \
                    Possible iters < 100. \n \
                    Please, Increase the 'timeout' parameter for normal optimization.")

            self.early_stoping, self.folds, self.score_folds, self.opt_lvl, self.cold_start = \
                    self.__auto_parameters_calc__(possible_iters)
        
        self._print_opt_parameters()

        ###############################################################################
        # Opt objective
        def objective(
            trial,
            opt,
            model,
            X,
            y,
            step=1,
            return_model=False,
            verbose=1,
            ):

            model.model_param = model.get_model_opt_params(
                    trial=trial, 
                    opt_lvl=opt.opt_lvl,
                    )

            model.select_columns = X.columns
            if opt.feature_selection:
                model.select_columns = opt._opt_feature_selector(X.columns, trial=trial)

            cv = CrossValidation(
                estimator=model,
                folds=opt.folds,
                score_folds=opt.score_folds,
                n_repeats=1,
                metric=opt.metric,
                print_metric=False, 
                metric_round=opt.metric_round, 
                random_state=opt._random_state,
                )

            score, score_std = cv.fit_score(X, y, print_metric=False, trial=trial)
            score_opt = round(opt.__calc_combined_score_opt__(opt.direction, score, score_std), opt.metric_round)
                    
            if cv._pruned_cv:
                #logger.info(f'- {trial.number} Trial Pruned, Score: {score_opt}')
                raise optuna.TrialPruned()

            if return_model:
                return(model)
            else:
                if verbose >= 1 and step > 1:
                    opt._tqdm_opt_print(pbar, score_opt, cv._pruned_cv)

                return(score_opt)

        ###############################################################################
        # Optuna
        logger.info('#'*50)
        sampler=optuna.samplers.TPESampler(#consider_prior=True, 
                                            n_startup_trials=self.cold_start, 
                                            #n_ei_candidates=50, 
                                            seed=self._random_state,
                                            multivariate=False,
                                            )

        datetime_now = datetime.now().strftime("%Y_%m_%d__%H:%M:%S")
        self.study = optuna.create_study(
            study_name=f"{datetime_now}_{model.__name__}", 
            storage="sqlite:///db.sqlite3",
            load_if_exists=True,
            direction=self.direction, 
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
            )

        self._set_opt_info(model, timeout)
        
        if model._is_model_start_opt_params():
            dafault_params = model.get_model_start_opt_params()
            self.study.enqueue_trial(dafault_params)

        obj_config = {
            'model':model,
            'X':X,
            'y':y,
            'verbose':verbose,
            }

        ###############################################################################
        # Step 1 
        # calc pruned score => get 10 n_trials and get score.median()
        logger.info(f'> Step 1: calc pruned score => get 10 trials')

        self.study.optimize(
            lambda trial: objective(trial, self, **obj_config,),
            n_trials=10, 
            show_progress_bar=False
            )
        
        df_tmp = self.study.trials_dataframe()
        pruned_scor = round((df_tmp.value.median()), self.metric_round)

        if self.direction == 'maximize':
            prun_params = {'lower':pruned_scor}
        else:
            prun_params = {'upper':pruned_scor}
        
        self.study.pruner = optuna.pruners.ThresholdPruner(**prun_params)
        self.study.set_user_attr("Pruned Threshold Score", pruned_scor,)

        logger.info(f'  Pruned Threshold Score: {pruned_scor}')
        logger.info('#'*50)

        ###############################################################################
        # Step 2
        # Full opt with ThresholdPruner
        logger.info(f'> Step 2: Full opt with Threshold Score Pruner')
        logger.info('#'*50)

        with tqdm(
            file=sys.stderr,
            desc="Optimize: ", 
            disable=disable_tqdm,
            ) as pbar:
            try:
                self.study.optimize(
                    lambda trial: objective(trial, self, step=2, **obj_config,),
                    timeout=(timeout - (start_opt_time - time.time())),  
                    callbacks=[es_callback], 
                    show_progress_bar=False,
                    )
            except EarlyStoppingExceeded:
                pbar.close()
                logger.info(f'\n EarlyStopping Exceeded: Best Score: {self.study.best_value} {self.metric.__name__}')
        
        pbar.close()

        ###############################################################################
        # get result
        logger.info(f'> Finish Opt!')
        model = objective(
            optuna.trial.FixedTrial(self.study.best_params), 
            self,
            return_model=True,
            **obj_config
            )
        logger.info(f'Best Score: {self.study.best_value} {self.metric.__name__}')

        return(model)


    def plot_param_importances(self,):
        '''
        Plot hyperparameter importances.
        '''
        if self.study is None:
            raise Exception('No history to visualize!')
        return(optuna.visualization.plot_param_importances(self.study))


    def plot_optimization_history(self,):
        '''
        Plot optimization history of all trials in a study.
        '''
        if self.study is None:
            raise Exception('No history to visualize!')
        return(optuna.visualization.plot_optimization_history(self.study))


    def plot_parallel_coordinate(self,):
        """
        Plot the high-dimentional parameter relationships in a study.
        Note that, If a parameter contains missing values, a trial with missing values is not plotted.
        """
        if self.study is None:
            raise Exception('No history to visualize!')
        return(optuna.visualization.plot_parallel_coordinate(self.study))


    def plot_slice(self, params=None):
        """
        Plot the parameter relationship as slice plot in a study.
        Note that, If a parameter contains missing values, a trial with missing values is not plotted.
        """
        if self.study is None:
            raise Exception('No history to visualize!')
        return(optuna.visualization.plot_slice(self.study, params=params))

    
    def plot_contour(self, params=None):
        """
        Plot the parameter relationship as contour plot in a study.
        Note that, If a parameter contains missing values, a trial with missing values is not plotted.
        """
        if self.study is None:
            raise Exception('No history to visualize!')
        return(optuna.visualization.plot_contour(self.study, params=params))



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