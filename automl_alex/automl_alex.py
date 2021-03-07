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


class Stacking(BestSingleModel):
    __name__ = "Stacking"
    history_trials = []
    fited_models = {}
    _fit_target_enc = {}

    def stack_cv(
        self,
        X,
        y,
        trial,
        estimator,
        metric,
        cat_features: Optional[List[str]] = None,
        target_encoders=[],
        folds=7,
        score_folds=3,
        n_repeats=1,
        metric_round=4,
        random_state=42,
    ):

        if cat_features is None:
            cat_features = X.columns[(X.nunique() < (len(X) // 100))]
        else:
            cat_features = cat_features

        if estimator._type_of_estimator == "classifier":
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

        self.cv_split_idx = [
            (train_idx, valid_idx) for (train_idx, valid_idx) in self.skf.split(X, y)
        ]

        folds_predicts = {}
        folds_y = {}
        folds_scores = []

        self._pruned_cv = False

        for i, (train_idx, valid_idx) in enumerate(self.cv_split_idx):
            train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
            val_x, val_y = X.iloc[valid_idx], y.iloc[valid_idx]
            # Target Encoder
            if len(target_encoders) > 0:
                val_x_copy = val_x[cat_features].copy()
                train_x_copy = train_x[cat_features].copy()
                for target_enc_name in target_encoders:
                    encoder_prefix = f"{trial.number}_model_{target_enc_name} _fold_{i}"
                    self._fit_target_enc[encoder_prefix] = copy.deepcopy(
                        target_encoders_names[target_enc_name](drop_invariant=True)
                    )

                    self._fit_target_enc[encoder_prefix] = self._fit_target_enc[
                        encoder_prefix
                    ].fit(train_x_copy, train_y)

                    data_encodet = self._fit_target_enc[encoder_prefix].transform(
                        train_x_copy
                    )
                    data_encodet = data_encodet.add_prefix(target_enc_name + "_")
                    train_x = train_x.join(data_encodet.reset_index(drop=True))
                    data_encodet = None

                    val_x_data_encodet = self._fit_target_enc[encoder_prefix].transform(
                        val_x_copy
                    )
                    val_x_data_encodet = val_x_data_encodet.add_prefix(
                        target_enc_name + "_"
                    )
                    val_x = val_x.join(val_x_data_encodet.reset_index(drop=True))
                    val_x_data_encodet = None
                train_x_copy = None
                train_x.fillna(0, inplace=True)
                val_x.fillna(0, inplace=True)

            # Fit
            estimator.fit(X_train=train_x, y_train=train_y)
            # moder_prefix = f"{trial.number}_model_fold_{i}"
            # self.fited_models[moder_prefix] = copy.deepcopy(estimator)
            if metric.__name__ in predict_proba_metrics:
                y_pred = estimator.predict_or_predict_proba(val_x)
            else:
                y_pred = estimator.predict(val_x)

            folds_predicts[f"y_pred_fold_{i}"] = y_pred
            folds_y[f"y_fold_{i}"] = val_y

            score_model = round(metric(val_y, y_pred), metric_round)
            folds_scores.append(score_model)

            if (trial is not None) and i < 1:
                trial.report(score_model, i)
                # Handle pruning based on the intermediate value.
                if trial.should_prune():
                    self._pruned_cv = True
                    break

            # score_folds
            if i + 1 >= score_folds:
                break

        if self._pruned_cv:
            score = score_model
            score_std = 0

        else:
            if score_folds > 1:
                score = round(np.mean(folds_scores), metric_round)
                score_std = round(np.std(folds_scores), metric_round + 2)
            else:
                score = round(score_model, metric_round)
                score_std = 0

        return (score, score_std, folds_predicts, folds_y)

    def mean_score_from_df(self, data, folds_y):
        folds_scores = []
        for i in range(self.score_folds):
            y_pred = data[f"y_pred_fold_{i}"].mean()
            y = folds_y[f"y_fold_{i}"]
            score_model = round(self.metric(y, y_pred), self.metric_round)
            folds_scores.append(score_model)

        all_models_score = round(np.mean(folds_scores), self.metric_round)
        all_models_score_std = round(np.std(folds_scores), self.metric_round + 2)
        return (all_models_score, all_models_score_std)

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

        cv_config = {
            #'X':X,
            #'y':y,
            "trial": trial,
            "estimator": self.opt_model,
            "target_encoders": self._select_target_encoders_names,
            "folds": self.folds,
            "score_folds": self.score_folds,
            "n_repeats": 1,
            "metric": self.metric,
            "metric_round": self.metric_round,
            "random_state": self._random_state,
        }

        self.select_columns = None
        if self.feature_selection:
            self.select_columns = self._opt_feature_selector(X.columns, trial=trial)
            score, score_std, folds_predicts, folds_y = self.stack_cv(
                X[self.select_columns], y, **cv_config
            )
        else:
            score, score_std, folds_predicts, folds_y = self.stack_cv(X, y, **cv_config)

        if not self._pruned_cv:
            # trail_config
            trail_config = {
                "trial_number": trial.number,
                "model_score": score,
                "score_std": score_std,
                "model_name": self.opt_model.__name__,
                "model_param": self.opt_model.model_param,
                "columns": self.select_columns,
                "folds": self.folds,
                "score_folds": self.score_folds,
            }
            trail_config.update(folds_predicts)
            self.history_trials.append(trail_config)

            # print(f'Single Model Score: {score}, std: {score_std}')

            self.history_trials_dataframe = pd.DataFrame(
                self.history_trials
            ).sort_values("model_score", ascending=True)
            if self.direction == "maximize":
                self.history_trials_dataframe = pd.DataFrame(
                    self.history_trials
                ).sort_values("model_score", ascending=False)

            # all mean
            models_score, models_score_std = self.mean_score_from_df(
                self.history_trials_dataframe, folds_y
            )
            print(f"ALL MODELS MEAN Score: {models_score}, std: {models_score_std}")

            models_score, models_score_std = self.mean_score_from_df(
                self.history_trials_dataframe.head(10), folds_y
            )
            print(f"TOP10 MODELS MEAN Score: {models_score}, std: {models_score_std}")
            print("#" * 50)

        return (score, score_std)


class StackingClassifier(Stacking):
    _type_of_estimator = "classifier"


class StackingRegressor(Stacking):
    _type_of_estimator = "regression"


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
                "LinearModel",
                "LightGBM",
                # "ExtraTrees",
                # "RandomForest",
                # "MLP",
            ],
            **params,
        )

        timeout_model_2 = (timeout * 0.25) - (time.time() - start_step_0)
        history = self.model_2.opt(
            X=X,
            y=y,
            timeout=timeout_model_2,
            verbose=verbose,
        )
        # self.model_2.save(name="model_2", folder=TMP_FOLDER)
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
        self.predict_model_2 = self.model_2.predict(X_source)

        ####################################################
        # STEP 3
        # Blend
        predicts = (self.predict_model_1 + self.predict_model_2) / 2
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
        dir_tmp = folder + "AutoML_tmp/"
        Path(dir_tmp).mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(folder + name + ".zip", dir_tmp)
        model = joblib.load(dir_tmp + "AutoML" + ".pkl")
        model.model_2 = model.model_2.load(name="model_2", folder=dir_tmp)
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
