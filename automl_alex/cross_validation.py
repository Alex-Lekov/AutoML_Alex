"""
Cross-validation is a method for evaluating an analytical model and its behavior on independent data. 
When evaluating the model, the available data is split into k parts. 
Then the model is trained on k − 1 pieces of data, and the rest of the data is used for testing. 
The procedure is repeated k times; in the end, each of the k pieces of data is used for testing. 
The result is an assessment of the effectiveness of the selected model with the most even use of the available data.
"""
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

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

from automl_alex._logger import *
from ._encoders import *

predict_proba_metrics = ["roc_auc_score", "log_loss", "brier_score_loss"]
TMP_FOLDER = ".automl-alex_tmp/"


class CrossValidation(object):
    """Allows you to wrap any of your models in a Cross-validation.

    Examples
    --------
    >>> from automl_alex import LightGBMClassifier, DataPrepare, CrossValidation
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
    >>> # Model
    >>> model = LightGBMClassifier()
    >>>
    >>> # CrossValidation
    >>> cv = CrossValidation(estimator=model,)
    >>> score, score_std = cv.fit_score(X_train, y_train, print_metric=True)
    >>> cv.fit(X_train, y_train)
    >>> predicts = cv.predict_test(X_test)
    >>> print('Test AUC: ', round(sklearn.metrics.roc_auc_score(y_test, predicts),4))
    """

    __name__ = "CrossValidation"
    _fit_models = False
    fited_models = {}
    """dictionary with trained models"""
    estimator = None
    """model"""
    _fit_target_enc = {}

    def __init__(
        self,
        estimator: Callable,  # model
        target_encoders_names: List[str] = [],
        folds: int = 7,
        score_folds: int = 5,
        n_repeats: int = 1,
        metric: Optional[Callable] = None,
        print_metric: bool = False,
        metric_round: int = 4,
        random_state: int = 42,
    ):
        """
        Parameters
        ----------
        estimator : Callable
            model object from automl_alex.models
            The object to use to fit model.
        target_encoders_names : List[str]
            name encoders (from automl_alex._encoders.target_encoders_names)
        folds : int, optional
            Number of folds., by default 7
        score_folds : int, optional
            Number of score folds, by default 5
        n_repeats : int, optional
            Number of times cross-validator needs to be repeated, by default 1
        metric : Optional[Callable], optional
            you can use standard metrics from sklearn.metrics or add custom metrics.
            If None, the metric is selected from the type of estimator:
            classifier: sklearn.metrics.roc_auc_score
            regression: sklearn.metrics.mean_squared_error., by default None
        print_metric : bool, optional
            print metric, by default False
        metric_round : int, optional
            round metric score, by default 4
        random_state : int, optional
            Controls the generation of the random states for each repetition, by default 42
        """
        self.estimator = estimator
        self.folds = folds
        self.score_folds = score_folds
        self.n_repeats = n_repeats
        self.print_metric = print_metric
        self.metric_round = metric_round
        self.target_encoders_names = target_encoders_names

        if metric is None:
            if estimator._type_of_estimator == "classifier":
                self.metric = sklearn.metrics.roc_auc_score
            elif estimator._type_of_estimator == "regression":
                self.metric = sklearn.metrics.mean_squared_error
        else:
            self.metric = metric

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

    def _clean_temp_folder(self):
        Path(TMP_FOLDER).mkdir(parents=True, exist_ok=True)
        if os.path.isdir(TMP_FOLDER + "cross-v_tmp"):
            shutil.rmtree(TMP_FOLDER + "cross-v_tmp")
        os.mkdir(TMP_FOLDER + "cross-v_tmp")

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[list, np.array, pd.DataFrame],
        cat_features: Optional[List[str]] = None,
    ):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : pd.DataFrame
            data (pd.DataFrame, shape (n_samples, n_features))
        y : Union[list, np.array, pd.DataFrame]
            target
        cat_features : Optional[List[str]], optional
            features name list. if None -> Auto-detection categorical_features, by default None
        """
        self._clean_temp_folder()
        # cat_features
        if cat_features is None:
            self.cat_features = X.columns[(X.nunique() < (len(X) // 100))]
        else:
            self.cat_features = cat_features

        self.cv_split_idx = [
            (train_idx, valid_idx) for (train_idx, valid_idx) in self.skf.split(X, y)
        ]

        for i, (train_idx, valid_idx) in enumerate(self.cv_split_idx):
            train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
            # Target Encoder
            if len(self.target_encoders_names) > 0:
                train_x_copy = train_x[self.cat_features].copy()
                for target_enc_name in self.target_encoders_names:
                    self._fit_target_enc[
                        f"{target_enc_name} _fold_{i}"
                    ] = copy.deepcopy(
                        target_encoders_names[target_enc_name](drop_invariant=True)
                    )

                    self._fit_target_enc[
                        f"{target_enc_name} _fold_{i}"
                    ] = self._fit_target_enc[f"{target_enc_name} _fold_{i}"].fit(
                        train_x_copy, train_y
                    )

                    data_encodet = self._fit_target_enc[
                        f"{target_enc_name} _fold_{i}"
                    ].transform(train_x_copy)
                    data_encodet = data_encodet.add_prefix(target_enc_name + "_")

                    train_x = train_x.join(data_encodet.reset_index(drop=True))
                train_x_copy = None
                train_x.fillna(0, inplace=True)

            # Fit
            self.estimator.fit(X_train=train_x, y_train=train_y)
            self.fited_models[
                f"model_{self.estimator.__name__}_fold_{i}"
            ] = copy.deepcopy(self.estimator)
        self._fit_models = True
        return self

    def predict_test(self, X_test):
        if not self._fit_models:
            raise Exception("No fit models")

        stacking_y_pred_test = np.zeros(len(X_test))

        for i in range(self.folds * self.n_repeats):
            X_test_tmp = X_test.copy()
            # Target Encoder
            if len(self.target_encoders_names) > 0:
                X_cat_features = X_test_tmp[self.cat_features].copy()
                for target_enc_name in self.target_encoders_names:
                    data_encodet = self._fit_target_enc[
                        f"{target_enc_name} _fold_{i}"
                    ].transform(X_cat_features)
                    data_encodet = data_encodet.add_prefix(target_enc_name + "_")

                    X_test_tmp = X_test_tmp.join(data_encodet.reset_index(drop=True))
                X_test_tmp.fillna(0, inplace=True)
            # Predict
            y_pred_test = self.fited_models[
                f"model_{self.estimator.__name__}_fold_{i}"
            ].predict_or_predict_proba(X_test_tmp)
            stacking_y_pred_test += y_pred_test
        predict = stacking_y_pred_test / (self.folds * self.n_repeats)

        return predict

    def predict_train(self, X):
        if not self._fit_models:
            raise Exception("No fit models")

        stacking_y_pred_train = np.zeros(len(X))

        for i, (train_idx, valid_idx) in enumerate(self.cv_split_idx):
            val_x = X.iloc[valid_idx]
            # Target Encoder
            if len(self.target_encoders_names) > 0:
                val_x_copy = val_x[self.cat_features].copy()
                for target_enc_name in self.target_encoders_names:
                    data_encodet = self._fit_target_enc[
                        f"{target_enc_name} _fold_{i}"
                    ].transform(val_x_copy)
                    data_encodet = data_encodet.add_prefix(target_enc_name + "_")
                    val_x = val_x.join(data_encodet.reset_index(drop=True))
                val_x_copy = None
                val_x.fillna(0, inplace=True)

            y_pred = self.fited_models[
                f"model_{self.estimator.__name__}_fold_{i}"
            ].predict_or_predict_proba(val_x)
            stacking_y_pred_train[valid_idx] += y_pred

        predict = stacking_y_pred_train / self.n_repeats

        return predict

    def get_feature_importance(self, X):
        if not self._fit_models:
            raise Exception("No fit models")

        if not self.estimator._is_possible_feature_importance():
            raise Exception("Can't get the feature importance for this estimator")

        feature_importance_df = pd.DataFrame(np.zeros(len(X.columns)), index=X.columns)

        for i in range(self.folds * self.n_repeats):
            X_tmp = X.copy()
            # Target Encoder
            if len(self.target_encoders_names) > 0:
                X_cat_features = X[self.cat_features].copy()
                for target_enc_name in self.target_encoders_names:
                    data_encodet = self._fit_target_enc[
                        f"{target_enc_name} _fold_{i}"
                    ].transform(X_cat_features)
                    data_encodet = data_encodet.add_prefix(target_enc_name + "_")

                    X_tmp = X_tmp.join(data_encodet.reset_index(drop=True))
            X_tmp.fillna(0, inplace=True)
            # Get feature_importance
            if i == 0:
                feature_importance_df = self.fited_models[
                    f"model_{self.estimator.__name__}_fold_{i}"
                ].get_feature_importance(X_tmp)
            feature_importance_df["value"] += self.fited_models[
                f"model_{self.estimator.__name__}_fold_{i}"
            ].get_feature_importance(X_tmp)["value"]

        return feature_importance_df

    def fit_score(
        self,
        X: pd.DataFrame,
        y: Union[list, np.array, pd.DataFrame],
        cat_features: Optional[List[str]] = None,
        print_metric=None,
        trial=None,
    ):
        self._pruned_cv = False
        if print_metric is None:
            print_metric = self.print_metric

        # cat_features
        if cat_features is None:
            cat_features = X.columns[(X.nunique() < (len(X) // 100))]

        self.cv_split_idx = [
            (train_idx, valid_idx) for (train_idx, valid_idx) in self.skf.split(X, y)
        ]

        folds_scores = []

        for i, (train_idx, valid_idx) in enumerate(self.cv_split_idx):
            train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
            val_x, val_y = X.iloc[valid_idx], y.iloc[valid_idx]
            # Target Encoder
            if len(self.target_encoders_names) > 0:
                val_x_copy = val_x[cat_features].copy()
                train_x_copy = train_x[cat_features].copy()
                for target_enc_name in self.target_encoders_names:
                    target_enc = target_encoders_names[target_enc_name](
                        drop_invariant=True
                    )

                    data_encodet = target_enc.fit_transform(train_x_copy, train_y)
                    data_encodet = data_encodet.add_prefix(target_enc_name + "_")
                    train_x = train_x.join(data_encodet.reset_index(drop=True))
                    data_encodet = None

                    val_x_data_encodet = target_enc.transform(val_x_copy)
                    val_x_data_encodet = val_x_data_encodet.add_prefix(
                        target_enc_name + "_"
                    )
                    val_x = val_x.join(val_x_data_encodet.reset_index(drop=True))
                    val_x_data_encodet = None
                val_x_copy = None
                train_x_copy = None
                train_x.fillna(0, inplace=True)
                val_x.fillna(0, inplace=True)

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
                    self._pruned_cv = True
                    break

            # score_folds
            if i + 1 >= self.score_folds:
                break

        if self._pruned_cv:
            score = score_model
            score_std = 0
        else:
            if self.score_folds > 1:
                score = round(np.mean(folds_scores), self.metric_round)
                score_std = round(np.std(folds_scores), self.metric_round + 2)
            else:
                score = round(score_model, self.metric_round)
                score_std = 0

        if print_metric:
            print(
                f"\n Mean Score {self.metric.__name__} on {self.score_folds} Folds: {score} std: {score_std}"
            )

        return (score, score_std)

    def save(self, name="cv_dump", folder="./", verbose=1):
        if not self._fit_models:
            raise Exception("No fit models")

        dir_tmp = TMP_FOLDER + "cross-v_tmp/"
        self._clean_temp_folder()

        for i in range(self.folds * self.n_repeats):
            # Target Encoder
            if len(self.target_encoders_names) > 0:
                for target_enc_name in self.target_encoders_names:
                    joblib.dump(
                        self._fit_target_enc[f"{target_enc_name} _fold_{i}"],
                        f"{dir_tmp}{target_enc_name} _fold_{i}.pkl",
                    )
            # Models
            self.fited_models[f"model_{self.estimator.__name__}_fold_{i}"].save(
                f"{dir_tmp}model_{self.estimator.__name__}_fold_{i}", verbose=0
            )

        joblib.dump(self, dir_tmp + "CV" + ".pkl")

        shutil.make_archive(folder + name, "zip", dir_tmp)

        shutil.rmtree(dir_tmp)
        if verbose > 0:
            print("Save CrossValidation")

    def load(self, name="cv_dump", folder="./", verbose=1):
        self._clean_temp_folder()
        dir_tmp = TMP_FOLDER + "cross-v_tmp/"

        shutil.unpack_archive(folder + name + ".zip", dir_tmp)

        cv = joblib.load(dir_tmp + "CV" + ".pkl")

        for i in range(cv.folds * cv.n_repeats):
            # Target Encoder
            if len(self.target_encoders_names) > 0:
                for target_enc_name in self.target_encoders_names:
                    self._fit_target_enc[f"{target_enc_name} _fold_{i}"] = joblib.load(
                        f"{dir_tmp}{target_enc_name} _fold_{i}.pkl"
                    )
            # Models
            cv.fited_models[
                f"model_{self.estimator.__name__}_fold_{i}"
            ] = copy.deepcopy(
                cv.estimator.load(
                    f"{dir_tmp}model_{self.estimator.__name__}_fold_{i}", verbose=0
                )
            )

        shutil.rmtree(dir_tmp)
        if verbose > 0:
            print("Load CrossValidation")
        return cv
