"""
In this library, I tried to simplify the process of optimizing and iterating parameters as much as possible. 
You just specify how much time you are willing to spend on optimization, 
and the library will select the rules for cross-validation, optimization depth, 
and other optimization parameters based on this time. 
The more time you give it, the better it will be able to select parameters.

Automatic hyperparameter optimization based on Optuna (https://optuna.org/) but with some modifications.
"""

from typing import Dict
import pandas as pd
import numpy as np
import time
import os
import shutil
import platform
import psutil
from datetime import datetime
from tqdm import tqdm
import optuna
import sklearn

from automl_alex._logger import *
from .cross_validation import *
from .data_prepare import *
import automl_alex

optuna.logging.disable_default_handler()

############################################
# IN progress....
class BestSingleModel(object):
    """
    Trying to find which model work best on our data

    Parameters
    ----------
    models_names : list
        models names estimator to opt. from automl_alex.models

    timeout : int
        Optimization time in seconds

    metric : Class
        you can use standard metrics from sklearn.metrics or add custom metrics.
        If None, the metric is selected from the type of estimator:
        classifier: sklearn.metrics.roc_auc_score
        regression: sklearn.metrics.mean_squared_error

    metric_round : int

    cold_start : int
        In the cold_start parameter, we specify how many iterations we give for this warm-up.
        before the algorithm starts searching for optimal parameters, it must collect statistics on the represented space.
        this is why it starts in the randomsample solution at the beginning.
        The longer it works in this mode , the less likely it is to get stuck in the local minimum.
        But if you take too long to warm up, you may not have time to search with a more "smart" algorithm.
        Therefore, it is important to maintain a balance.

    opt_lvl : int
        by limiting the optimization time, we will have to choose how deep we should optimize the parameters.
        Perhaps some parameters are not so important and can only give a fraction of a percent.
        By setting the opt_lvl parameter, you control the depth of optimization.
        in the code automl_alex.models.model_lightgbm.LightGBM you can find how parameters are substituted for iteration

    early_stoping : int
        stop optimization if no better parameters are found through iterations

    auto_parameters : bool
        If we don't want to select anything, we just set auto_parameters=True. Then the algorithm itself will select, depending on the time allotted to it, the optimal values for:
            *folds
            *score_folds
            *cold_start
            *opt_lvl

    feature_selection : bool
        add feature_selection in optimization

    random_state : int
        RandomState instance
        Controls the generation of the random states for each repetition.
    """

    __name__ = "BestSingleModel"
    cv_model = None
    """ Best Model in CV wraper (automl_alex.cross_validation)"""
    best_model_name: str = None
    best_model_param: Dict = None

    def __init__(
        self,
        clean_and_encod_data: bool = False,
        models_names: List[str] = ["LinearModel", "LightGBM", "ExtraTrees"],
        opt_data_prepare: bool = False,
        cat_encoder_names: List[str] = [
            "HelmertEncoder",
            "OneHotEncoder",
            "CountEncoder",
            "HashingEncoder",
            "BackwardDifferenceEncoder",
        ],
        target_encoders_names: List[str] = [
            "TargetEncoder",
            "JamesSteinEncoder",
            "CatBoostEncoder",
            "WOEEncoder",
        ],
        clean_outliers: List[bool] = [True, False],
        num_generator_select_operations: bool = True,
        num_generator_operations: List[str] = ["/", "*", "-"],
        normalization=False,
        folds: int = 7,
        score_folds: int = 2,
        metric: Optional[Callable] = None,
        metric_round: int = 4,
        cold_start: int = 15,
        opt_lvl: int = 1,
        early_stoping: int = 50,
        auto_parameters: bool = True,
        feature_selection: bool = False,
        type_of_estimator=None,  # classifier or regression
        gpu: bool = False,
        random_state: int = 42,
        verbose: Optional[bool] = None,
    ):
        if verbose is not None:
            logger_print_lvl(verbose)
            self._verbose = verbose
        self._random_state = random_state
        self._gpu = gpu
        self._clean_and_encod_data = clean_and_encod_data
        self._opt_data_prepare = opt_data_prepare
        self._cat_encoder_names = cat_encoder_names
        self._target_encoders_names = target_encoders_names
        self._clean_outliers = clean_outliers
        self._num_generator_select_operations = num_generator_select_operations
        self._num_generator_operations = num_generator_operations
        self._normalization = normalization

        if models_names is None:
            self.models_names = list(automl_alex.models.all_models.keys())
        else:
            self.models_names = models_names

        if type_of_estimator is not None:
            self._type_of_estimator = type_of_estimator

        self.folds = folds
        self.score_folds = score_folds
        self.metric_round = metric_round
        self._auto_parameters = auto_parameters
        self.cold_start = cold_start
        self.opt_lvl = opt_lvl
        self.early_stoping = early_stoping
        self.feature_selection = feature_selection

        if metric is None:
            logger.info(
                "metric is None! Default metric will be used. classifier: AUC, regression: MSE"
            )
            if self._type_of_estimator == "classifier":
                self.metric = sklearn.metrics.roc_auc_score
            elif self._type_of_estimator == "regression":
                self.metric = sklearn.metrics.mean_squared_error
            else:
                logger.warning("Need to set type_of_estimator!")
        else:
            self.metric = metric

    def __metric_direction_detected__(self, metric, y):
        zero_y = np.zeros(len(y))
        zero_score = metric(y, zero_y)
        best_score = metric(y, y)

        if best_score > zero_score:
            direction = "maximize"
        else:
            direction = "minimize"
        return direction

    def __calc_combined_score_opt__(self, direction, score, score_std):
        """
        Args:
            direction (str): 'minimize' or 'maximize'
            score (float): the input score
            score_std (float): the input score_std

        Return:
            score_opt (float): combined score
        """
        if direction == "maximize":
            score_opt = score - score_std
        else:
            score_opt = score + score_std
        return score_opt

    def __auto_parameters_calc__(
        self,
        possible_iters,
    ):
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
        folds = 7
        score_folds = 3
        opt_lvl = 1
        cold_start = 15

        if possible_iters > 100:
            opt_lvl = 2
            cold_start = 20
            early_stoping = 30

        if possible_iters > 500:
            opt_lvl = 3
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

        return (
            early_stoping,
            folds,
            score_folds,
            opt_lvl,
            cold_start,
        )

    def _print_opt_parameters(
        self,
    ):
        logger.info("> Start optimization with the parameters:")
        logger.info(f"CV_Folds = {self.folds}")
        logger.info(f"Score_CV_Folds = {self.score_folds}")
        logger.info(f"Feature_Selection = {self.feature_selection}")
        logger.info(f"Opt_lvl = {self.opt_lvl}")
        logger.info(f"Cold_start = {self.cold_start}")
        logger.info(f"Early_stoping = {self.early_stoping}")
        logger.info(f"Metric = {self.metric.__name__}")
        logger.info(f"Direction = {self.direction}")

    def _tqdm_opt_print(self, pbar, score_opt):
        """
        Printing information in tqdm. Use pbar.
        See the documentation for tqdm: https://github.com/tqdm/tqdm
        """
        if pbar is not None:
            self.best_score = self.study.best_value

            message = f"| Model: {self.model_name} | OptScore: {score_opt} | Best {self.metric.__name__}: {self.best_score} "
            pbar.set_postfix_str(message)
            pbar.update(1)

    def _set_opt_info(self, model, timeout):
        self.study.set_user_attr("Type_estimator", self._type_of_estimator)
        self.study.set_user_attr("Metric", self.metric.__name__)
        self.study.set_user_attr("direction", self.direction)
        self.study.set_user_attr("Timeout", timeout)
        self.study.set_user_attr("auto_parameters", self._auto_parameters)
        self.study.set_user_attr("early_stoping", self.early_stoping)
        self.study.set_user_attr("cold_start", self.cold_start)
        self.study.set_user_attr("opt_lvl", self.opt_lvl)
        self.study.set_user_attr("Folds", self.folds)
        self.study.set_user_attr("Score_folds", self.score_folds)
        self.study.set_user_attr("Opt_lvl", self.opt_lvl)
        self.study.set_user_attr("random_state", self._random_state)
        self.study.set_system_attr("System", platform.system())
        self.study.set_system_attr("CPU", platform.processor())
        self.study.set_system_attr("CPU cores", psutil.cpu_count())
        ram = str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"
        self.study.set_system_attr("RAM", ram)

    def _set_opt_sys_info(self):
        self.study.set_system_attr("CPU %", psutil.cpu_percent())
        free_mem = round(
            psutil.virtual_memory().available * 100 / psutil.virtual_memory().total, 1
        )
        self.study.set_system_attr("Free RAM %", free_mem)

    def get_model_from_iter(self, X, y, params):
        # Opt objective
        def objective(trial, self, X, y, verbose=1):
            # self._set_opt_sys_info()
            return self._opt_objective(trial, X, y, return_model=True, verbose=verbose)

        ###################################
        self.cv_model = objective(
            optuna.trial.FixedTrial(params),
            self,
            X=X,
            y=y,
            verbose=self._verbose,
        )
        return self

    def _get_opt_model_(self, trial):
        """
        now we can choose models in optimization
        """
        if len(self.models_names) > 1:
            self.model_name = trial.suggest_categorical("model_name", self.models_names)
        else:
            self.model_name = self.models_names[0]

        opt_model = automl_alex.models.all_models[self.model_name](
            type_of_estimator=self._type_of_estimator,
            random_state=self._random_state,
            gpu=self._gpu,
            verbose=None,
        )
        return opt_model

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
            result = list(select_columns_.keys())
        else:
            result = list(columns)
        return result

    def _opt_encoder_selector(self, columns, trial):
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
            result = list(select_columns_.keys())
        else:
            result = []
        return result

    def _opt_data_prepare_func(self, X: pd.DataFrame, trial):
        self.de_cfg = {
            "num_generator_features": True,
            "random_state": self._random_state,
            "verbose": 1,
        }

        self.de_cfg["cat_encoder_names"] = self._opt_encoder_selector(
            self._cat_encoder_names,
            trial,
        )

        self._select_target_encoders_names = self._opt_encoder_selector(
            self._target_encoders_names,
            trial,
        )

        if len(self._clean_outliers) > 1:
            self.de_cfg["clean_outliers"] = trial.suggest_categorical(
                "de_clean_outliers", [True, False]
            )
        else:
            self.de_cfg["clean_outliers"] = self._clean_outliers[0]

        if self._num_generator_select_operations:
            self.de_cfg["num_generator_operations"] = self._opt_encoder_selector(
                self._num_generator_operations, trial
            )

        if self.model_name in automl_alex.models.need_norm_data_models:
            self.de_cfg["normalization"] = True
        else:
            self.de_cfg["normalization"] = False

        self.de = DataPrepare(**self.de_cfg)
        X = self.de.fit_transform(X)
        logger_print_lvl(self._verbose)
        return X

    def _opt_objective(self, trial, X, y, return_model=False, verbose=1):
        if len(self.models_names) > 1:
            self.opt_model = self._get_opt_model_(trial)
        self.opt_model.model_param = self.opt_model.get_model_opt_params(
            trial=trial,
            opt_lvl=self.opt_lvl,
        )

        if self._opt_data_prepare:
            X = self._opt_data_prepare_func(self._X_source, trial)
        else:
            self._select_target_encoders_names = self._target_encoders_names

        cv = CrossValidation(
            estimator=self.opt_model,
            target_encoders_names=self._select_target_encoders_names,
            folds=self.folds,
            score_folds=self.score_folds,
            n_repeats=1,
            metric=self.metric,
            print_metric=False,
            metric_round=self.metric_round,
            random_state=self._random_state,
        )

        if return_model:
            if self.feature_selection:
                self.select_columns = self._opt_feature_selector(X.columns, trial=trial)
                cv.fit(X[self.select_columns], y)
            else:
                cv.fit(X, y)
            return cv
        else:
            if self.feature_selection:
                self.select_columns = self._opt_feature_selector(X.columns, trial=trial)
                score, score_std = cv.fit_score(
                    X[self.select_columns], y, print_metric=False, trial=trial
                )
            else:
                score, score_std = cv.fit_score(X, y, print_metric=False, trial=trial)
            return (score, score_std)

    def opt(
        self,
        X,
        y,
        timeout=600,  # optimization time in seconds
        verbose=None,
        fit_end=True,
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

        disable_tqdm = True
        if verbose is not None:
            logger_print_lvl(verbose)
            self._verbose = verbose
            if verbose > 0:
                disable_tqdm = False

        if self.metric is not None:
            self.direction = self.__metric_direction_detected__(self.metric, y)

        # model = self.estimator
        ###############################################################################
        # Data
        if self._clean_and_encod_data and not self._opt_data_prepare:
            self.de_cfg = {
                "cat_encoder_names": self._cat_encoder_names,
                "clean_outliers": self._clean_outliers,
                "normalization": self._normalization,
                "num_generator_operations": self._num_generator_operations,
                "num_generator_features": True,
                "random_state": self._random_state,
                "verbose": 1,
            }
            self.de = DataPrepare(**self.de_cfg)
            X = self.de.fit_transform(X)

        if self._clean_and_encod_data and self._opt_data_prepare:
            self._X_source = X.copy()
        ###############################################################################
        # Optuna _EarlyStoppingExceeded
        es = _EarlyStoppingExceeded()
        es.early_stop = self.early_stoping
        es.early_stop_count = 0
        es.best_score = None

        es_callback = es.early_stopping_opt_minimize
        if self.direction == "maximize":
            es_callback = es.early_stopping_opt_maximize

        ###############################################################################
        # Opt objective
        def objective(
            trial,
            self,
            X,
            y,
            step=1,
            return_model=False,
            verbose=1,
        ):
            # self._set_opt_sys_info()
            score, score_std = self._opt_objective(
                trial,
                X,
                y,
                return_model=return_model,
                verbose=verbose,
            )
            score_opt = round(
                self.__calc_combined_score_opt__(self.direction, score, score_std),
                self.metric_round,
            )
            if trial.should_prune():
                # logger.info(f'- {trial.number} Trial Pruned, Score: {score_opt}')
                raise optuna.TrialPruned()

            if verbose >= 1 and step > 1:
                self._tqdm_opt_print(pbar, score_opt)
            return score_opt

        ###############################################################################
        # Optuna
        logger.info("#" * 50)
        sampler = optuna.samplers.TPESampler(  # consider_prior=True,
            n_startup_trials=self.cold_start,
            seed=self._random_state,
            multivariate=False,
        )

        datetime_now = datetime.now().strftime("%Y_%m_%d__%H:%M:%S")
        self.study = optuna.create_study(
            study_name=f"{datetime_now}_{self.__name__}",
            storage="sqlite:///db.sqlite3",
            load_if_exists=True,
            direction=self.direction,
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
        )

        self._set_opt_info(self, timeout)
        self._set_opt_sys_info()

        # if self.estimator._is_model_start_opt_params():
        #     dafault_params = self.estimator.get_model_start_opt_params()
        #     self.study.enqueue_trial(dafault_params)

        obj_config = {
            "X": X,
            "y": y,
            "verbose": self._verbose,
        }

        # init opt model
        self.model_name = self.models_names[0]

        self.opt_model = automl_alex.models.all_models[self.model_name](
            type_of_estimator=self._type_of_estimator,
            random_state=self._random_state,
            gpu=self._gpu,
            verbose=self._verbose,
        )

        ###############################################################################
        # Step 1
        # calc pruned score => get 10 n_trials and get score.median()
        logger.info(f"> Step 1: calc parameters and pruned score: get test 10 trials")

        start_time = time.time()

        self.study.optimize(
            lambda trial: objective(
                trial,
                self,
                **obj_config,
            ),
            n_trials=10,
            show_progress_bar=False,
        )

        iter_time = (time.time() - start_time) / 10
        logger.info(f" One iteration ~ {round(iter_time,1)} sec")

        possible_iters = timeout // (iter_time)
        logger.info(f" Possible iters ~ {possible_iters}")
        if possible_iters < 100:
            logger.warning(
                "! Not enough time to find the optimal parameters. \n \
                    Possible iters < 100. \n \
                    Please, Increase the 'timeout' parameter for normal optimization."
            )
        logger.info("-" * 50)

        if self._auto_parameters:
            (
                self.early_stoping,
                self.folds,
                self.score_folds,
                self.opt_lvl,
                self.cold_start,
            ) = self.__auto_parameters_calc__(possible_iters)

        # pruners
        df_tmp = self.study.trials_dataframe()
        pruned_scor = round((df_tmp.value.median()), self.metric_round)

        if self.direction == "maximize":
            prun_params = {"lower": pruned_scor}
        else:
            prun_params = {"upper": pruned_scor}

        self.study.pruner = optuna.pruners.ThresholdPruner(**prun_params)
        self.study.set_user_attr(
            "Pruned Threshold Score",
            pruned_scor,
        )

        logger.info(f"  Pruned Threshold Score: {pruned_scor}")
        logger.info("#" * 50)

        ###############################################################################
        # Step 2
        # Full opt with ThresholdPruner
        logger.info(f"> Step 2: Full opt with Threshold Score Pruner")
        logger.info("#" * 50)
        self._print_opt_parameters()
        logger.info("#" * 50)

        timeout_step2 = timeout - (time.time() - start_opt_time)
        timeout_step2 = timeout_step2 - (iter_time * self.folds)  # prediction time
        with tqdm(file=sys.stderr, desc="Optimize: ", disable=disable_tqdm) as pbar:
            try:
                self.study.optimize(
                    lambda trial: objective(trial, self, step=2, **obj_config),
                    timeout=timeout_step2,
                    callbacks=[es_callback],
                    show_progress_bar=False,
                )
            except _EarlyStoppingExceeded:
                pbar.close()
                logger.info(
                    f"\n EarlyStopping Exceeded: Best Score: {self.study.best_value} {self.metric.__name__}"
                )

        pbar.close()

        ###############################################################################
        # fit CV model
        logger.info(f"> Finish Opt!")

        if fit_end:
            self.get_model_from_iter(X, y, self.study.best_params)
            self.best_model_name = self.cv_model.estimator.__name__
            self.best_model_param = self.cv_model.estimator.model_param

        logger.info(f"Best Score: {self.study.best_value} {self.metric.__name__}")
        return self.study.trials_dataframe()

    def predict_test(self, X):
        if not self.cv_model:
            raise Exception("No opt and fit models")

        if self._clean_and_encod_data:
            X = self.de.transform(X)

        if self.feature_selection:
            X = X[self.select_columns]

        predict = self.cv_model.predict_test(X)
        return predict

    def predict(self, X):
        return self.predict_test(X)

    def predict_train(self, X):
        if not self.cv_model:
            raise Exception("No opt and fit models")

        if self._clean_and_encod_data:
            X = self.de.transform(X)

        if self.feature_selection:
            X = X[self.select_columns]

        predict = self.cv_model.predict_train(X)
        return predict

    def _clean_temp_folder(self) -> None:
        Path(TMP_FOLDER).mkdir(parents=True, exist_ok=True)
        if os.path.isdir(TMP_FOLDER + "/cross-v_tmp"):
            shutil.rmtree(TMP_FOLDER + "/cross-v_tmp")
        os.mkdir(TMP_FOLDER + "/cross-v_tmp")

    def save(
        self, name: str = "opt_model_dump", folder: str = "./", verbose: int = 3
    ) -> None:
        if not self.cv_model:
            raise Exception("No opt and fit models")

        dir_tmp = TMP_FOLDER + "/opt_model_tmp/"
        self._clean_temp_folder()

        if self._clean_and_encod_data:
            self.de.save(name="DataPrepare_dump", folder=dir_tmp)

        self.cv_model.save(name="opt_model_cv", folder=dir_tmp, verbose=0)
        joblib.dump(self, dir_tmp + "opt_model" + ".pkl")

        shutil.make_archive(folder + name, "zip", dir_tmp)

        shutil.rmtree(dir_tmp)
        if verbose > 0:
            print("Save model")

    def load(self, name: str = "opt_model_dump", folder: str = "./", verbose: int = 1):
        """ Load Model """
        self._clean_temp_folder()
        dir_tmp = TMP_FOLDER + "/opt_model_tmp/"

        shutil.unpack_archive(folder + name + ".zip", dir_tmp)

        model = joblib.load(dir_tmp + "opt_model" + ".pkl")

        if model._clean_and_encod_data:
            model.de = DataPrepare()
            model.de = model.de.load("DataPrepare_dump", folder=dir_tmp)

        model.cv_model = model.cv_model.load(
            name="opt_model_cv",
            folder=dir_tmp,
        )

        shutil.rmtree(dir_tmp)
        if verbose > 0:
            print("Load model")
        return model

    def plot_opt_param_importances(
        self,
    ):
        """
        Plot hyperparameter importances.
        """
        if self.study is None:
            raise Exception("No history to visualize!")
        return optuna.visualization.plot_param_importances(self.study)

    def plot_opt_history(
        self,
    ):
        """
        Plot optimization history of all trials in a study.
        """
        if self.study is None:
            raise Exception("No history to visualize!")
        return optuna.visualization.plot_optimization_history(self.study)

    def plot_parallel_coordinate(
        self,
    ):
        """
        Plot the high-dimentional parameter relationships in a study.
        Note that, If a parameter contains missing values, a trial with missing values is not plotted.
        """
        if self.study is None:
            raise Exception("No history to visualize!")
        return optuna.visualization.plot_parallel_coordinate(self.study)

    def plot_slice(self, params=None):
        """
        Plot the parameter relationship as slice plot in a study.
        Note that, If a parameter contains missing values, a trial with missing values is not plotted.
        """
        if self.study is None:
            raise Exception("No history to visualize!")
        return optuna.visualization.plot_slice(self.study, params=params)

    def plot_contour(self, params=None):
        """
        Plot the parameter relationship as contour plot in a study.
        Note that, If a parameter contains missing values, a trial with missing values is not plotted.
        """
        if self.study is None:
            raise Exception("No history to visualize!")
        return optuna.visualization.plot_contour(self.study, params=params)


class BestSingleModelClassifier(BestSingleModel):
    _type_of_estimator = "classifier"


class BestSingleModelRegressor(BestSingleModel):
    _type_of_estimator = "regression"


class _EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    """
    Custom EarlyStop for Optuna
    """

    def __init__(self, early_stop=100, best_score=None):
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
                self.early_stop_count = self.early_stop_count + 1
            else:
                self.early_stop_count = 0
                self.best_score = None
                raise _EarlyStoppingExceeded()

    def early_stopping_opt_minimize(self, study, trial):
        if self.best_score is None:
            self.best_score = study.best_value

        if study.best_value < self.best_score:
            self.best_score = study.best_value
            self.early_stop_count = 0
        else:
            if self.early_stop_count < self.early_stop:
                self.early_stop_count = self.early_stop_count + 1
            else:
                self.early_stop_count = 0
                self.best_score = None
                raise _EarlyStoppingExceeded()
