from automl_alex._base import ModelBase
import lightgbm as lgb
import numpy as np
import pandas as pd


class LightGBM(ModelBase):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """

    __name__ = "LightGBM"

    def _init_default_model_param(
        self,
    ):
        """
        Default model_param
        """
        model_param = {
            "random_seed": self._random_state,
            "num_iterations": 300,
            "verbose": -1,
            "device_type": "gpu" if self._gpu else "cpu",
        }

        if self._type_of_estimator == "classifier":
            model_param["objective"] = "binary"

        if self._type_of_estimator == "regression":
            model_param["objective"] = "regression"
        return model_param

    def fit(self, X_train=None, y_train=None, cat_features=None) -> None:
        """
        Args:
            X (pd.DataFrame, shape (n_samples, n_features)): the input data
            y (pd.DataFrame, shape (n_samples, ) or (n_samples, n_outputs)): the target data
        Return:
            self
        """
        y_train = self.y_format(y_train)

        dtrain = lgb.Dataset(
            X_train,
            y_train,
        )

        model_param = self.model_param.copy()
        num_iterations = model_param.pop("num_iterations")

        if cat_features is None:
            cat_features = "auto"

        self.model = lgb.train(
            model_param,
            dtrain,
            num_boost_round=num_iterations,
            categorical_feature=cat_features,
        )
        dtrain = None
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
        return self.model.predict(X)

    def _is_possible_feature_importance(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return True

    def get_feature_importance(
        self,
        X,
        importance_type="gain",
    ):
        """
        Return:
            list feature_importance
        """
        if not self._is_possible_feature_importance():
            raise Exception("Model cannot get feature_importance")

        fe_lst = self.model.feature_importance(importance_type=importance_type)
        return pd.DataFrame(fe_lst, index=X.columns, columns=["value"])

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
            model_param["num_leaves"] = trial.suggest_int(
                "lgbm_num_leaves", 2, 100, log=True
            )
            model_param["learning_rate"] = trial.suggest_float(
                "lgbm_learning_rate", 1e-2, 0.3, log=True
            )

        ################################# LVL 2 ########################################
        if opt_lvl == 2:
            model_param["num_iterations"] = trial.suggest_int(
                "lgbm_num_iterations", 300, 500, step=100
            )

        if opt_lvl >= 2:
            model_param["min_child_samples"] = trial.suggest_int(
                "lgbm_min_child_samples", 2, 100, log=True
            )
            model_param["bagging_fraction"] = trial.suggest_float(
                "lgbm_bagging_fraction", 0.4, 1.0, step=0.1
            )
            if model_param["bagging_fraction"] < 1.0:
                model_param["feature_fraction"] = trial.suggest_float(
                    "lgbm_feature_fraction", 0.4, 1.0, step=0.1
                )
                model_param["bagging_freq"] = trial.suggest_int(
                    "lgbm_bagging_freq",
                    2,
                    11,
                )

        ################################# LVL 3 ########################################
        if opt_lvl == 3:
            model_param["num_iterations"] = trial.suggest_int(
                "lgbm_num_iterations", 300, 1000, step=100
            )

        ################################# LVL 4 ########################################
        if opt_lvl == 4:
            model_param["learning_rate"] = trial.suggest_float(
                "lgbm_learning_rate", 1e-3, 0.1, log=True
            )

        if opt_lvl >= 4:
            model_param["boosting"] = trial.suggest_categorical(
                "lgbm_boosting",
                [
                    "gbdt",
                    "dart",
                ],
            )
            if model_param["boosting"] == "dart":
                model_param["early_stopping_rounds"] = 0
                model_param["uniform_drop"] = trial.suggest_categorical(
                    "lgbm_uniform_drop", [True, False]
                )
                model_param["xgboost_dart_mode"] = trial.suggest_categorical(
                    "lgbm_xgboost_dart_mode", [True, False]
                )
                model_param["drop_rate"] = trial.suggest_float(
                    "lgbm_drop_rate", 1e-8, 1.0, log=True
                )
                model_param["max_drop"] = trial.suggest_int("lgbm_max_drop", 0, 100)
                model_param["skip_drop"] = trial.suggest_float(
                    "lgbm_skip_drop", 1e-3, 1.0, log=True
                )

            model_param["num_iterations"] = trial.suggest_int(
                "lgbm_num_iterations",
                200,
                1500,
                step=100,
            )

            if self._type_of_estimator == "classifier":
                model_param["objective"] = trial.suggest_categorical(
                    "lgbm_objective",
                    [
                        "binary",
                        "cross_entropy",
                    ],
                )

            elif self._type_of_estimator == "regression":
                model_param["objective"] = trial.suggest_categorical(
                    "lgbm_objective",
                    [
                        "regression",
                        "regression_l1",
                        "mape",
                        "huber",
                        "quantile",
                    ],
                )

        ################################# LVL 5 ########################################
        if opt_lvl >= 5:
            model_param["max_cat_threshold"] = trial.suggest_int(
                "lgbm_max_cat_threshold", 1, 100
            )
            model_param["min_child_weight"] = trial.suggest_float(
                "lgbm_min_child_weight", 1e-6, 1.0, log=True
            )
            model_param["learning_rate"] = trial.suggest_float(
                "lgbm_learning_rate", 1e-5, 0.1, log=True
            )
            model_param["reg_lambda"] = trial.suggest_float(
                "lgbm_reg_lambda", 1e-8, 1.0, log=True
            )
            model_param["reg_alpha"] = trial.suggest_float(
                "lgbm_reg_alpha", 1e-8, 1.0, log=True
            )
            model_param["max_bin"] = (
                trial.suggest_int(
                    "lgbm_max_bin",
                    1,
                    5,
                )
                * 50
            )
            # self.model_param['extra_trees'] = trial.suggest_categorical('lgbm_extra_trees', [True, False])
            model_param["enable_bundle"] = trial.suggest_categorical(
                "lgbm_enable_bundle", [True, False]
            )

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
            "lgbm_num_leaves": 31,
            "lgbm_min_child_samples": 20,
            "lgbm_learning_rate": 0.1,
            "lgbm_num_iterations": 300,
        }
        return dafault_params


class LightGBMClassifier(LightGBM):
    _type_of_estimator = "classifier"


class LightGBMRegressor(LightGBM):
    _type_of_estimator = "regression"
