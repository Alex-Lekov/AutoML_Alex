from sklearn import ensemble, neural_network, linear_model, svm, neighbors
from automl_alex._base import ModelBase
import numpy as np

from warnings import simplefilter, filterwarnings
from sklearn.exceptions import ConvergenceWarning, DataConversionWarning

simplefilter("ignore", category=ConvergenceWarning)
filterwarnings(
    "ignore",
    category=ConvergenceWarning,
    message="^Maximum number of iteration reached",
)
filterwarnings(
    "ignore", category=ConvergenceWarning, message="^Liblinear failed to converge"
)
simplefilter("ignore", category=DataConversionWarning)


################################## LinearModel ##########################################################


class LinearModel(ModelBase):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """

    __name__ = "LinearModel"

    def _init_default_model_param(
        self,
    ):
        """
        Default model_param
        """
        model_param = {}
        return model_param

    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self._type_of_estimator == "classifier":
            model = linear_model.LogisticRegression(**model_param)
        elif self._type_of_estimator == "regression":
            model = linear_model.LinearRegression(**model_param)
        return model

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

        model_param["fit_intercept"] = trial.suggest_categorical(
            "lr_fit_intercept", [True, False]
        )
        if self._type_of_estimator == "classifier":
            ################################# LVL 1 ########################################
            if opt_lvl >= 1:
                model_param["C"] = trial.suggest_uniform("lr_C", 0.1, 100.0)

            ################################# LVL 2 ########################################
            if opt_lvl >= 2:
                model_param["solver"] = trial.suggest_categorical(
                    "lr_solver", ["lbfgs", "saga", "liblinear"]
                )
                model_param["tol"] = trial.suggest_uniform("lr_tol", 1e-6, 0.1)
                model_param["class_weight"] = trial.suggest_categorical(
                    "lr_class_weight", [None, "balanced"]
                )

                if model_param["solver"] == "saga":
                    model_param["penalty"] = trial.suggest_categorical(
                        "lr_penalty",
                        [
                            "l1",
                            "l2",
                            "elasticnet",
                        ],
                    )
                    if model_param["penalty"] == "elasticnet":
                        model_param["l1_ratio"] = trial.suggest_uniform(
                            "lr_l1_ratio", 0.0, 1.0
                        )
                        model_param["max_iter"] = 5000
                if model_param["solver"] == "liblinear":
                    model_param["n_jobs"] = 1
                if model_param["solver"] == "lbfgs":
                    model_param["max_iter"] = 5000
        return model_param

    def _is_model_start_opt_params(
        self,
    ):
        return False

    def fit(self, X_train=None, y_train=None, cat_features=None):
        """
        Args:
            X (pd.DataFrame, shape (n_samples, n_features)): the input data
            y (pd.DataFrame, shape (n_samples, ) or (n_samples, n_outputs)): the target data
        Return:
            self
        """
        y_train = self.y_format(y_train)

        self.model = self._init_model(model_param=self.model_param)
        self.model.fit(
            X_train,
            y_train,
        )
        return self

    def predict(self, X_test=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        if self.model is None:
            raise Exception("No fit models")

        return self.model.predict(X_test)

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return True

    def predict_proba(self, X_test):
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
        return self.model.predict_proba(X_test)[:, 1]


class LogisticRegressionClassifier(LinearModel):
    _type_of_estimator = "classifier"


class LinearRegression(LinearModel):
    _type_of_estimator = "regression"


########################################### SVM #####################################################


class LinearSVM(LinearModel):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """

    __name__ = "LinearSVM"

    def _init_default_model_param(self, model_param=None):
        """
        Default model_param
        """
        model_param = {
            "verbose": 0,
            "random_state": self._random_state,
        }
        return model_param

    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self._type_of_estimator == "classifier":
            model = svm.LinearSVC(**model_param)
        elif self._type_of_estimator == "regression":
            model = svm.LinearSVR(**model_param)
        return model

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
            model_param["tol"] = trial.suggest_uniform("svc_tol", 1e-7, 1e-2)
            model_param["C"] = trial.suggest_uniform("svc_C", 1e-4, 50.0)
            model_param["fit_intercept"] = trial.suggest_categorical(
                "svc_fit_intercept", [True, False]
            )

        ################################# LVL 2 ########################################
        if opt_lvl >= 2:
            if self._type_of_estimator == "classifier":
                model_param["loss"] = trial.suggest_categorical(
                    "svc_loss",
                    [
                        "hinge",
                        "squared_hinge",
                    ],
                )
                # model_param['class_weight'] = trial.suggest_categorical('svc_class_weight',[None, 'balanced'])
                if model_param["loss"] == "squared_hinge":
                    model_param["penalty"] = trial.suggest_categorical(
                        "svc_penalty",
                        [
                            "l2",
                            "l1",
                        ],
                    )
                    if model_param["penalty"] == "l2":
                        model_param["dual"] = trial.suggest_categorical(
                            "svc_dual", [True, False]
                        )
                    if model_param["penalty"] == "l1":
                        model_param["dual"] = False
        return model_param

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return False


class LinearSVClassifier(LinearSVM):
    _type_of_estimator = "classifier"


class LinearSVRegressor(LinearSVM):
    _type_of_estimator = "regression"


########################################### KNN #####################################################


class KNeighbors(LinearModel):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """

    __name__ = "KNeighbors"

    def _init_default_model_param(self, model_param=None):
        """
        Default model_param
        """
        model_param = {
            "n_jobs": -1,
        }
        return model_param

    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self._type_of_estimator == "classifier":
            model = neighbors.KNeighborsClassifier(**model_param)
        elif self._type_of_estimator == "regression":
            model = neighbors.KNeighborsRegressor(**model_param)
        return model

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
        # model opt params
        model_param["n_neighbors"] = trial.suggest_int("knn_n_neighbors", 2, 150)
        model_param["weights"] = trial.suggest_categorical(
            "knn_weights",
            [
                "uniform",
                "distance",
            ],
        )
        model_param["algorithm"] = trial.suggest_categorical(
            "knn_algorithm", ["auto", "ball_tree", "kd_tree", "brute"]
        )
        model_param["p"] = trial.suggest_categorical(
            "knn_p",
            [
                1,
                2,
            ],
        )
        if (
            model_param["algorithm"] == "ball_tree"
            or model_param["algorithm"] == "kd_tree"
        ):
            model_param["leaf_size"] = trial.suggest_int("knn_leaf_size", 2, 150)
        return model_param


class KNNClassifier(KNeighbors):
    _type_of_estimator = "classifier"


class KNNRegressor(KNeighbors):
    _type_of_estimator = "regression"


########################################### MLP #####################################################


class MLP(LinearModel):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """

    __name__ = "MLP"

    def _init_default_model_param(self, model_param=None):
        """
        Default model_param
        """
        model_param = {
            "verbose": 0,
            "hidden_layer_sizes": 150,
            "random_state": self._random_state,
            "max_iter": 500,
            "early_stopping": True,
            "n_iter_no_change": 50,
        }
        return model_param

    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self._type_of_estimator == "classifier":
            model = neural_network.MLPClassifier(**model_param)
        elif self._type_of_estimator == "regression":
            model = neural_network.MLPRegressor(**model_param)
        return model

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
        # model params
        model_param.update(
            {
                "hidden_layer_sizes": trial.suggest_int("mlp_hidden_layer_sizes", 1, 10)
                * 50,
                "alpha": trial.suggest_uniform("mlp_alpha", 1e-6, 1.0),
                "learning_rate": trial.suggest_categorical(
                    "mlp_learning_rate", ["adaptive", "constant", "invscaling"]
                ),
                "tol": trial.suggest_uniform("mlp_tol", 1e-6, 1e-1),
            }
        )
        return model_param


class MLPClassifier(MLP):
    _type_of_estimator = "classifier"


class MLPRegressor(MLP):
    _type_of_estimator = "regression"


########################################### RandomForest #####################################################


class RandomForest(LinearModel):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """

    __name__ = "RandomForest"

    def _init_default_model_param(self, model_param=None):
        """
        Default model_param
        """
        model_param = {
            "verbose": 0,
            "random_state": self._random_state,
            "n_jobs": -1,
        }
        return model_param

    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self._type_of_estimator == "classifier":
            model = ensemble.RandomForestClassifier(**model_param)
        elif self._type_of_estimator == "regression":
            model = ensemble.RandomForestRegressor(**model_param)
        return model

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
            model_param["min_samples_split"] = trial.suggest_int(
                "rf_min_samples_split", 2, 100
            )
            model_param["max_depth"] = trial.suggest_int(
                "rf_max_depth", 10, 100, step=10
            )

        ################################# LVL 2 ########################################
        if opt_lvl >= 2:
            model_param["n_estimators"] = trial.suggest_int(
                "rf_n_estimators", 100, 1000, step=100
            )
            # model_param['max_features'] = trial.suggest_categorical('rf_max_features', [
            #    'auto',
            #    'sqrt',
            #    'log2'
            #    ])

        ################################# LVL 3 ########################################
        if opt_lvl >= 3:
            model_param["bootstrap"] = trial.suggest_categorical(
                "rf_bootstrap", [True, False]
            )
            if model_param["bootstrap"]:
                model_param["oob_score"] = trial.suggest_categorical(
                    "rf_oob_score", [True, False]
                )
            # if self._type_of_estimator == 'classifier':
            #    model_param['class_weight'] = trial.suggest_categorical('rf_class_weight',[None, 'balanced'])
        return model_param


class RandomForestClassifier(RandomForest):
    _type_of_estimator = "classifier"


class RandomForestRegressor(RandomForest):
    _type_of_estimator = "regression"


########################################### ExtraTrees #####################################################


class ExtraTrees(RandomForest):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """

    __name__ = "ExtraTrees"

    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self._type_of_estimator == "classifier":
            model = ensemble.ExtraTreesClassifier(**model_param)
        elif self._type_of_estimator == "regression":
            model = ensemble.ExtraTreesRegressor(**model_param)
        return model

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
            model_param["min_samples_split"] = trial.suggest_int(
                "ext_min_samples_split", 2, 100
            )
            model_param["max_depth"] = trial.suggest_int(
                "ext_max_depth", 10, 100, step=10
            )

        ################################# LVL 2 ########################################
        if opt_lvl >= 2:
            model_param["n_estimators"] = trial.suggest_int(
                "ext_n_estimators", 100, 1000, step=100
            )
            # model_param['max_features'] = trial.suggest_categorical('rf_max_features', [
            #    'auto',
            #    'sqrt',
            #    'log2'
            #    ])

        ################################# LVL 3 ########################################
        if opt_lvl >= 3:
            model_param["bootstrap"] = trial.suggest_categorical(
                "ext_bootstrap", [True, False]
            )
            if model_param["bootstrap"]:
                model_param["oob_score"] = trial.suggest_categorical(
                    "ext_oob_score", [True, False]
                )
            # if self._type_of_estimator == 'classifier':
            #    model_param['class_weight'] = trial.suggest_categorical('rf_class_weight',[None, 'balanced'])
        return model_param



class ExtraTreesClassifier(ExtraTrees):
    _type_of_estimator = "classifier"


class ExtraTreesRegressor(ExtraTrees):
    _type_of_estimator = "regression"
