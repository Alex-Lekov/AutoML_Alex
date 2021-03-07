"""AutoML and other Toolbox"""

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from sklearn.metrics import *
from tqdm import tqdm
import pandas as pd
import time
import joblib
import os
import contextlib
import copy
import automl_alex
from .models import *
from .cross_validation import *
from .data_prepare import *
from ._encoders import *
from .optimizer import *
from pathlib import Path
from ._logger import *

TMP_FOLDER = ".automl-alex_tmp/"

##################################### ModelsReview ################################################


class ModelsReview(object):
    """
    ModelsReview - allows you to see which models show good results on this data

    Examples
    --------
    >>> from automl_alex import ModelsReview, DataPrepare
    >>> import sklearn
    >>> # Get Dataset
    >>> dataset = sklearn.datasets.fetch_openml(name='adult', version=1, as_frame=True)
    >>> dataset.target = dataset.target.astype('category').cat.codes
    >>> X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    >>>                                             dataset.data,
    >>>                                             dataset.target,
    >>>                                             test_size=0.2,)
    >>> # clean up data before use ModelsReview
    >>> de = DataPrepare()
    >>> clean_X_train = de.fit_transform(X_train)
    >>> clean_X_test = de.transform(X_test)
    >>>
    >>> model = ModelsReview(type_of_estimator='classifier',
    >>>                     metric = sklearn.metrics.roc_auc_score,)
    >>> review = model.fit(X_train=X_train,
    >>>                     y_train=y_train,
    >>>                     X_test=X_test,
    >>>                     y_test=y_test,)
    """

    __name__ = "ModelsReview"

    def __init__(
        self,
        type_of_estimator: Optional[str] = None,  # classifier or regression
        metric: Optional[Callable] = None,
        metric_round: int = 4,
        gpu: bool = False,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        type_of_estimator : Optional[str], optional
            ['classifier', 'regression'], by default None
        metric : Callable, optional
            you can use standard metrics from sklearn.metrics or add custom metrics.
            If None, the metric is selected from the type of estimator:
            classifier: sklearn.metrics.roc_auc_score
            regression: sklearn.metrics.mean_squared_error.
        metric_round : int, optional
            round metric score., by default 4
        gpu : bool, optional
            Use GPU?, by default False
        random_state : int, optional
            Controls the generation of the random states for each repetition, by default 42
        """
        self._gpu = gpu
        self._random_state = random_state
        if type_of_estimator is not None:
            self._type_of_estimator = type_of_estimator

        if metric is None:
            if self._type_of_estimator == "classifier":
                self._metric = sklearn.metrics.roc_auc_score
            elif self._type_of_estimator == "regression":
                self._metric = sklearn.metrics.mean_squared_error
        else:
            self._metric = metric
        self._metric_round = metric_round

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[list, np.array, pd.DataFrame],
        X_test: pd.DataFrame,
        y_test: Union[list, np.array, pd.DataFrame],
        models_names: Optional[List[str]] = None,
        verbose: int = 3,
    ) -> pd.DataFrame:
        """
        Fit models from model_list and return scores

        Parameters
        ----------
        X_train : pd.DataFrame
            train data (pd.DataFrame, shape (n_samples, n_features))
        y_train : Union[list, np.array, pd.DataFrame]
            target
        X_test : pd.DataFrame
            test data (pd.DataFrame, shape (n_samples, n_features))
        y_test : Union[list, np.array, pd.DataFrame]
            test target
        models_names : Optional[List[str]], optional
            list of models from automl_alex.models.all_models, by default None
        verbose : int, optional
            print state, by default 3

        Returns
        -------
        pd.DataFrame
            results
        """
        logger_print_lvl(verbose)
        result = pd.DataFrame(columns=["Model_Name", "Score", "Time_Fit_Sec"])
        score_ls = []
        time_ls = []
        if models_names is None:
            self.models_names = automl_alex.models.all_models.keys()
        else:
            self.models_names = models_names

        result["Model_Name"] = self.models_names

        if verbose > 0:
            disable_tqdm = False
        else:
            disable_tqdm = True
        for model_name in tqdm(self.models_names, disable=disable_tqdm):
            # Model
            start_time = time.time()
            model_tmp = automl_alex.models.all_models[model_name](
                gpu=self._gpu,
                random_state=self._random_state,
                type_of_estimator=self._type_of_estimator,
            )
            # fit
            model_tmp.fit(X_train, y_train)
            # Predict
            if (self._metric.__name__ in predict_proba_metrics) and (
                model_tmp.is_possible_predict_proba()
            ):
                y_pred = model_tmp.predict_proba(X_test)
            else:
                y_pred = model_tmp.predict(X_test)

            score_model = round(self._metric(y_test, y_pred), self._metric_round)
            score_ls.append(score_model)
            iter_time = round((time.time() - start_time), 2)
            time_ls.append(iter_time)
            model_tmp = None

        result["Score"] = score_ls
        result["Time_Fit_Sec"] = time_ls
        self.result = result
        return result


class ModelsReviewClassifier(ModelsReview):
    """
    ModelsReview - allows you to see which models show good results on this data

    Examples
    --------
    >>> from automl_alex import ModelsReviewClassifier, DataPrepare
    >>> import sklearn
    >>> # Get Dataset
    >>> dataset = sklearn.datasets.fetch_openml(name='adult', version=1, as_frame=True)
    >>> dataset.target = dataset.target.astype('category').cat.codes
    >>> X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    >>>                                             dataset.data,
    >>>                                             dataset.target,
    >>>                                             test_size=0.2,)
    >>> # clean up data before use ModelsReview
    >>> de = DataPrepare()
    >>> clean_X_train = de.fit_transform(X_train)
    >>> clean_X_test = de.transform(X_test)
    >>>
    >>> model = ModelsReviewClassifier(metric = sklearn.metrics.roc_auc_score,)
    >>> review = model.fit(X_train=X_train,
    >>>                     y_train=y_train,
    >>>                     X_test=X_test,
    >>>                     y_test=y_test,)
    """

    _type_of_estimator = "classifier"


class ModelsReviewRegressor(ModelsReview):
    """
    ModelsReview - allows you to see which models show good results on this data

    Examples
    --------
    >>> from automl_alex import ModelsReviewRegressor, DataPrepare
    >>> import sklearn
    >>> # Get Dataset
    >>> dataset = sklearn.datasets.fetch_openml(data_id=543, as_frame=True)
    >>> X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    >>>                                             pd.DataFrame(dataset.data),
    >>>                                             pd.DataFrame(dataset.target),
    >>>                                             test_size=0.2,)
    >>> # clean up data before use ModelsReview
    >>> de = DataPrepare()
    >>> clean_X_train = de.fit_transform(X_train)
    >>> clean_X_test = de.transform(X_test)
    >>>
    >>> model = ModelsReviewRegressor(metric = sklearn.metrics.mean_squared_error,)
    >>> review = model.fit(X_train=X_train,
    >>>                     y_train=y_train,
    >>>                     X_test=X_test,
    >>>                     y_test=y_test,)
    """

    _type_of_estimator = "regression"


##################################### Stacking #########################################
# in progress...

##################################### AutoML #########################################


class AutoML(object):
    """
    AutoML in the process of developing

    Examples
    --------
    >>> from automl_alex import AutoML
    >>> import sklearn
    >>> # Get Dataset
    >>> dataset = sklearn.datasets.fetch_openml(name='adult', version=1, as_frame=True)
    >>> dataset.target = dataset.target.astype('category').cat.codes
    >>> X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    >>>                                             dataset.data,
    >>>                                             dataset.target,
    >>>                                             test_size=0.2,)
    >>>
    >>> model = AutoML(type_of_estimator='classifier')
    >>> model.fit(X_train, y_train, timeout=600)
    >>> predicts = model.predict(X_test)
    >>> print('Test AUC: ', round(sklearn.metrics.roc_auc_score(y_test, predicts),4))

    """

    __name__ = "AutoML"

    def __init__(
        self,
        type_of_estimator: Optional[str] = None,  # classifier or regression
        metric: Optional[Callable] = None,
        metric_round: int = 4,
        gpu: bool = False,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        type_of_estimator : Optional[str], optional
            ['classifier', 'regression'], by default None
        metric : Callable, optional
            you can use standard metrics from sklearn.metrics or add custom metrics.
            If None, the metric is selected from the type of estimator:
            classifier: sklearn.metrics.roc_auc_score
            regression: sklearn.metrics.mean_squared_error.
        metric_round : int, optional
            round metric score., by default 4
        gpu : bool, optional
            Use GPU?, by default False
        random_state : int, optional
            Controls the generation of the random states for each repetition, by default 42
        """
        self._gpu = gpu
        self._random_state = random_state

        if type_of_estimator is not None:
            self._type_of_estimator = type_of_estimator

        if metric is not None:
            self.metric = metric
        else:
            if self._type_of_estimator == "classifier":
                self.metric = sklearn.metrics.roc_auc_score
                self.direction = "maximize"
            elif self._type_of_estimator == "regression":
                self.metric = sklearn.metrics.mean_squared_error
                self.direction = "minimize"

        self._metric_round = metric_round

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[list, np.array, pd.DataFrame],
        timeout: int = 600,  # optimization time in seconds
        auto_parameters: bool = True,
        folds: int = 7,
        score_folds: int = 3,
        opt_lvl: int = 2,
        early_stoping: int = 100,
        feature_selection: bool = False,
        verbose: int = 3,
    ) -> None:
        """
        Fit the model.

        Parameters
        ----------
        X : pd.DataFrame
            data (pd.DataFrame, shape (n_samples, n_features))
        y : Union[list, np.array, pd.DataFrame]
            target
        timeout : int, optional
            Optimization time in seconds.
        auto_parameters: bool, optional
            If we don't want to select anything, we just set auto_parameters=True.
            Then the algorithm itself will select, depending on the time allotted to it, the optimal values for:
                -folds,
                -score_folds,
                -cold_start,
                -opt_lvl,
        folds : int, optional
            Number of folds for CrossValidation. Must be at least 2, by default 7
        score_folds : int, optional
            Number of score folds. Must be at least 1., by default 3
        opt_lvl : int, optional
            by limiting the optimization time, we will have to choose how deep we should optimize the parameters.
            Perhaps some parameters are not so important and can only give a fraction of a percent.
            By setting the opt_lvl parameter, you control the depth of optimization.
            in the code automl_alex.models.model_lightgbm.LightGBM you can find how parameters are substituted for iteration
            by default 2
        early_stoping : int, optional
            stop optimization if no better parameters are found through iterations
        feature_selection : bool, optional
            add feature_selection in optimization, by default True
        verbose : int, optional
            print state, by default 3

        Returns
        -------
        None
        """
        logger_print_lvl(verbose)
        ####################################################
        # STEP 0
        logger.info("> Start Fit Base Model")
        if timeout < 600:
            logger.warning(
                "! Not enough time to find the optimal parameters. \n \
                    Please, Increase the 'timeout' parameter for normal optimization. (min 600 sec)"
            )
        ####################################################
        # STEP 0
        start_step_0 = time.time()

        X_tmp = X.copy()
        self._cat_cat_features = X_tmp.columns[
            (X_tmp.dtypes == "object") | (X_tmp.dtypes == "category")
        ]
        X_tmp[self._cat_cat_features] = X_tmp[self._cat_cat_features].astype("str")
        X_tmp.fillna(0, inplace=True)
        X_tmp[self._cat_cat_features] = X_tmp[self._cat_cat_features].astype("category")

        self.model_1 = automl_alex.CatBoost(
            type_of_estimator=self._type_of_estimator,
            random_state=self._random_state,
            gpu=self._gpu,
            # verbose=verbose,
        )
        self.model_1 = self.model_1.fit(
            X_tmp, y, cat_features=self._cat_cat_features.tolist()
        )
        X_tmp = None

        start_step_1 = time.time()

        params = {
            "clean_and_encod_data": True,
            "opt_data_prepare": True,
            "metric": self.metric,
            "metric_round": self._metric_round,
            "auto_parameters": auto_parameters,
            "folds": folds,
            "score_folds": score_folds,
            "opt_lvl": opt_lvl,
            "early_stoping": early_stoping,
            "feature_selection": feature_selection,
            "type_of_estimator": self._type_of_estimator,
            "random_state": self._random_state,
            "gpu": self._gpu,
            "cat_encoder_names": [
                "HelmertEncoder",
                "OneHotEncoder",
                "CountEncoder",
                "HashingEncoder",
                "BackwardDifferenceEncoder",
            ],
            "target_encoders_names": [
                "TargetEncoder",
                "JamesSteinEncoder",
                "CatBoostEncoder",
            ],
            "clean_outliers": [True, False],
            "num_generator_select_operations": True,
            "num_generator_operations": ["/", "*", "-", "+"],
            #'iteration_check': False,
        }
        ####################################################

        logger.info(50 * "#")
        logger.info("> Start Fit Models 2")
        logger.info(50 * "#")
        # Model 2
        self.model_2 = automl_alex.BestSingleModel(
            models_names=[
                # "LinearModel",
                "LightGBM",
                # "ExtraTrees",
                # "RandomForest",
                # "MLP",
            ],
            **params,
        )

        timeout_model_2 = (timeout) - (time.time() - start_step_0) - 120
        history = self.model_2.opt(
            X=X,
            y=y,
            timeout=timeout_model_2,
            verbose=verbose,
            fit_end=False,
        )
        # self.model_2.save(name="model_2", folder=TMP_FOLDER)
        ####################################################
        # Blend top5 models
        logger.info(50 * "#")
        logger.info("> Fit Best Models")
        logger.info(50 * "#")

        if self.model_2.direction == "maximize":
            top_10_cfg = history.sort_values("value", ascending=False).head(5)
        else:
            top_10_cfg = history.sort_values("value", ascending=True).head(5)

        self._tmp_models_folder = TMP_FOLDER + "automl_models/"
        if os.path.isdir(self._tmp_models_folder):
            shutil.rmtree(self._tmp_models_folder)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            for i in range(5):
                n_model = top_10_cfg.number.values[i]
                model = self.model_2.get_model_from_iter(
                    X, y, self.model_2.study.trials[n_model].params
                )
                model.save(f"model_{i+1}", folder=self._tmp_models_folder)

        ####################################################
        logger.info(50 * "#")
        logger.info("> Finish!")
        return self

    def predict(self, X: pd.DataFrame, verbose: int = 0) -> list:
        """
        Predict the target for the input data

        Parameters
        ----------
        X : pd.DataFrame
            data (pd.DataFrame, shape (n_samples, n_features))
        verbose : int, optional
            print state, by default 0

        Returns
        -------
        list
            prediction

        Raises
        ------
        Exception
            If No fit models
        """
        if self.model_1 is None:
            raise Exception("No fit models")

        logger_print_lvl(verbose)

        X_source = X.copy()
        ####################################################
        # STEP 1
        X[self._cat_cat_features] = X[self._cat_cat_features].astype("str")
        X.fillna(0, inplace=True)
        X[self._cat_cat_features] = X[self._cat_cat_features].astype("category")

        # MODEL 1
        self.predict_model_1 = self.model_1.predict_or_predict_proba(X)

        ####################################################
        # STEP 2
        # MODEL 2
        # self.model_2 = self.model_2.load(name="model_2", folder=TMP_FOLDER)

        models_predicts = []
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            for i in range(5):
                model = self.model_2.load(
                    f"model_{i+1}", folder=self._tmp_models_folder, verbose=0
                )
                predicts = model.predict(X_source)
                models_predicts.append(predicts)

        self.predict_model_2 = pd.DataFrame(models_predicts).mean()

        ####################################################
        # STEP 3
        # Blend
        predicts = (self.predict_model_1 * 0.4) + (self.predict_model_2 * 0.6)
        return predicts

    @logger.catch
    def save(self, name: str = "AutoML_dump", folder: str = "./") -> None:
        """
        Save the model to disk

        Parameters
        ----------
        name : str, optional
            file name, by default 'AutoML_dump'
        folder : str, optional
            target folder, by default './'
        """
        dir_tmp = folder + "AutoML_tmp/"
        Path(dir_tmp).mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dir_tmp + "AutoML" + ".pkl")
        self.model_2.save(name="model_2", folder=dir_tmp)
        if os.path.isdir(dir_tmp + "automl_models/"):
            shutil.rmtree(dir_tmp + "automl_models/")
        shutil.copytree(self._tmp_models_folder, dir_tmp + "automl_models/")
        shutil.make_archive(folder + name, "zip", dir_tmp)
        shutil.rmtree(dir_tmp)
        logger.info("Save AutoML")

    @logger.catch
    def load(self, name: str = "AutoML_dump", folder: str = "./") -> Callable:
        """
        Loads the model and creates a function that will load the model

        Parameters
        ----------
        name : str, optional
            file name, by default 'AutoML_dump'
        folder : str, optional
            target folder, by default './'

        Returns
        -------
        Callable
            AutoML
        """
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            dir_tmp = folder + "AutoML_tmp/"
            Path(dir_tmp).mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(folder + name + ".zip", dir_tmp)
            model = joblib.load(dir_tmp + "AutoML" + ".pkl")
            model.model_2 = model.model_2.load(
                name="model_2", folder=dir_tmp, verbose=0
            )
            if os.path.isdir(model._tmp_models_folder):
                shutil.rmtree(model._tmp_models_folder)
            shutil.copytree(dir_tmp + "automl_models/", model._tmp_models_folder)
            shutil.rmtree(dir_tmp)
        logger.info("Load AutoML")
        return model


class AutoMLClassifier(AutoML):
    """
    AutoML in the process of developing

    Examples
    --------
    >>> from automl_alex import AutoMLClassifier
    >>> import sklearn
    >>> # Get Dataset
    >>> dataset = sklearn.datasets.fetch_openml(name='adult', version=1, as_frame=True)
    >>> dataset.target = dataset.target.astype('category').cat.codes
    >>> X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    >>>                                             dataset.data,
    >>>                                             dataset.target,
    >>>                                             test_size=0.2,)
    >>>
    >>> model = AutoMLClassifier()
    >>> model.fit(X_train, y_train, timeout=600)
    >>> predicts = model.predict(X_test)
    >>> print('Test AUC: ', round(sklearn.metrics.roc_auc_score(y_test, predicts),4))

    """

    _type_of_estimator = "classifier"
    __name__ = "AutoMLClassifier"


class AutoMLRegressor(AutoML):
    """
    AutoML in the process of developing

    Examples
    --------
    >>> from automl_alex import AutoMLRegressor
    >>> import sklearn
    >>> # Get Dataset
    >>> dataset = sklearn.datasets.fetch_openml(data_id=543, as_frame=True)
    >>> X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    >>>                                             pd.DataFrame(dataset.data),
    >>>                                             pd.DataFrame(dataset.target),
    >>>                                             test_size=0.2,)
    >>>
    >>> model = AutoMLRegressor()
    >>> model.fit(X_train, y_train, timeout=600)
    >>> predicts = model.predict(X_test)
    >>> print('Test MSE: ', round(sklearn.metrics.mean_squared_error(y_test, predicts),4))

    """

    _type_of_estimator = "regression"
    __name__ = "AutoMLRegressor"
