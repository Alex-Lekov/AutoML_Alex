from automl_alex._base import ModelBase
import catboost
from catboost import Pool
import numpy as np
import pandas as pd


class CatBoost(ModelBase):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """

    __name__ = "CatBoost"

    def _init_default_model_param(self, model_param=None):
        """
        Default model_param
        """
        if model_param is None:
            model_param = {
                "verbose": 0,
                "task_type": "GPU" if self._gpu else "CPU",
                "random_seed": self._random_state,
            }
        return model_param

    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self._type_of_estimator == "classifier":
            model = catboost.CatBoostClassifier(**model_param)
        elif self._type_of_estimator == "regression":
            model = catboost.CatBoostRegressor(**model_param)
        return model

    def fit(self, X_train=None, y_train=None, cat_features=None):
        """
        Args:
            X (pd.DataFrame, shape (n_samples, n_features)): the input data
            y (pd.DataFrame, shape (n_samples, ) or (n_samples, n_outputs)): the target data
            cat_features (list)
        Return:
            self
        """
        y_train = self.y_format(y_train)

        if cat_features is not None:
            cat_dims = [X_train.columns.get_loc(i) for i in cat_features[:]]
            train_pool = Pool(X_train, label=y_train, cat_features=cat_dims)
        else:
            train_pool = Pool(
                X_train,
                label=y_train,
            )

        params = self.model_param.copy()
        self.model = self._init_model(model_param=params)
        self.model.fit(
            train_pool,
            verbose=False,
            plot=False,
        )
        train_pool = None
        return self

    def predict(self, X=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        if self.model is None:
            raise Exception("No fit models")

        if self._type_of_estimator == "classifier":
            predicts = np.round(self.model.predict(X), 0)
        elif self._type_of_estimator == "regression":
            predicts = self.model.predict(X)
        return predicts

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return True

    def predict_proba(self, X=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        if self.model is None:
            raise Exception("No fit models")

        if not self.is_possible_predict_proba():
            raise Exception("Model cannot predict probability distribution")
        return self.model.predict_proba(X)[:, 1]

    def _is_possible_feature_importance(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return True

    def get_feature_importance(
        self,
        X,
    ):
        """
        Return:
            list feature_importance
        """
        if not self._is_possible_feature_importance():
            raise Exception("Model cannot get feature_importance")

        fe_lst = self.model.get_feature_importance()
        return pd.DataFrame(fe_lst, index=X.columns, columns=["value"])

    # @staticmethod
    def get_model_opt_params(
        self,
        trial,
        opt_lvl,
    ):
        """
        Return:
            dict of DistributionWrappers
        """
        model_param = self._init_default_model_param()
        ################################# LVL 1 ########################################
        if opt_lvl >= 1:
            model_param["min_child_samples"] = trial.suggest_int(
                "cb_min_child_samples", 1, 100
            )
            model_param["depth"] = trial.suggest_int("cb_depth", 4, 10)

        ################################# LVL 2 ########################################
        if opt_lvl >= 2:
            model_param["bagging_temperature"] = trial.suggest_int(
                "cb_bagging_temperature",
                0,
                10,
            )
            model_param["subsample"] = trial.suggest_discrete_uniform(
                "cb_subsample", 0.1, 1.0, 0.1
            )

        ################################# LVL 3 ########################################
        if opt_lvl >= 3:
            if self._type_of_estimator == "classifier":
                model_param["objective"] = trial.suggest_categorical(
                    "cb_objective",
                    [
                        "Logloss",
                        "CrossEntropy",
                    ],
                )

            elif self._type_of_estimator == "regression":
                model_param["objective"] = trial.suggest_categorical(
                    "cb_objective",
                    [
                        "MAE",
                        "MAPE",
                        "Quantile",
                        "RMSE",
                    ],
                )

        ################################# LVL 4 ########################################
        if opt_lvl >= 4:
            model_param["depth"] = trial.suggest_int("cb_depth", 2, 16)
            model_param["l2_leaf_reg"] = trial.suggest_loguniform(
                "cb_l2_leaf_reg", 1e-8, 0.1
            )
            model_param["learning_rate"] = (
                trial.suggest_int("cb_learning_rate", 1, 100) / 1000
            )
            model_param["iterations"] = trial.suggest_int("cb_iterations", 1, 10) * 100

        ################################# Other ########################################
        return model_param

    def _is_model_start_opt_params(
        self,
    ):
        return True

    def get_model_start_opt_params(
        self,
    ):
        dafault_params = {
            "cb_depth": 6,
            "cb_min_child_samples": 1,
            "cb_learning_rate": 0.03,
        }
        return dafault_params


class CatBoostClassifier(CatBoost):
    _type_of_estimator = "classifier"
    __name__ = "CatBoostClassifier"


class CatBoostRegressor(CatBoost):
    _type_of_estimator = "regression"
    __name__ = "CatBoostRegressor"
