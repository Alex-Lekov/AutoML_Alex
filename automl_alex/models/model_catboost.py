from .base import *
import catboost
from catboost import Pool
import numpy as np


class CatBoost(ModelBase):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """
    __name__ = 'CatBoost'

    def _init_default_model_param(self, model_param=None):
        """
        Default model_param
        """
        if model_param is None:
            model_param = {'verbose': 0, 
                                'task_type': 'GPU' if self._gpu else 'CPU',
                                'random_seed': self._random_state,
                                }
        return(model_param)


    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self.type_of_estimator == 'classifier':
            model = catboost.CatBoostClassifier(**model_param)
        elif self.type_of_estimator == 'regression':
            model = catboost.CatBoostRegressor(**model_param)
        return(model)


    #@staticmethod
    def get_model_opt_params(self, trial, opt_lvl, len_data, metric_name):
        """
        Return:
            dict of DistributionWrappers
        """
        model_param = self._init_default_model_param()
        ################################# LVL 1 ########################################
        if opt_lvl == 1:
            model_param['depth'] = trial.suggest_categorical('cb_depth', [6, 10])

        if opt_lvl >= 1:
            if len_data > 1000:
                model_param['min_child_samples'] = trial.suggest_int('cb_min_child_samples', 2, \
                                                                    (len_data//100))
            else:
                model_param['min_child_samples'] = trial.suggest_int('cb_min_child_samples', 2, 10)
            

        ################################# LVL 2 ########################################
        if opt_lvl == 2:
            model_param['depth'] = trial.suggest_categorical('cb_depth', [4, 6, 10])

        if opt_lvl >= 2:
            model_param['bagging_temperature'] = trial.suggest_int('cb_bagging_temperature', 0, 10,)
            model_param['subsample'] = trial.suggest_discrete_uniform('cb_subsample', 0.1, 1.0, 0.1)
        
        ################################# LVL 3 ########################################
        if opt_lvl == 3:
            model_param['depth'] = trial.suggest_categorical('cb_depth', [4, 6, 8, 10])

        if opt_lvl >= 3:
            if self.type_of_estimator == 'classifier':
                model_param['objective'] = trial.suggest_categorical('cb_objective', 
                        [
                        'Logloss', 
                        'CrossEntropy',
                        ])

            elif self.type_of_estimator == 'regression':
                model_param['objective'] = trial.suggest_categorical('cb_objective', 
                    [
                    'MAE',
                    'MAPE',
                    'Quantile',
                    'RMSE',
                    ])

            if model_param['objective'] == 'Logloss':
                if metric_name not in ['roc_auc_score', 'log_loss', 'brier_score_loss']:
                    model_param['scale_pos_weight'] = trial.suggest_discrete_uniform('cb_scale_pos_weight', 0.1, 1., 0.1)
        

        ################################# LVL 4 ########################################
        if opt_lvl >= 4:
            model_param['depth'] = trial.suggest_int('cb_depth', 2, 16)
            model_param['l2_leaf_reg'] = trial.suggest_loguniform('cb_l2_leaf_reg', 1e-8, .1)
            model_param['learning_rate'] = trial.suggest_int('cb_learning_rate', 1, 100)/1000
            model_param['iterations'] = trial.suggest_int('cb_iterations', 1, 10)*100

        ################################# Other ########################################
        return(model_param)

    def fit(self, X_train=None, y_train=None,):
        """
        Args:
            X (pd.DataFrame, shape (n_samples, n_features)): the input data
            y (pd.DataFrame, shape (n_samples, ) or (n_samples, n_outputs)): the target data
        Return:
            self (Class)
        """
        y_train = self.y_format(y_train)
        train_pool = Pool(X_train, label=y_train,)

        params = self.model_param.copy()
        self.model = self._init_model(model_param=params)
        self.model.fit(train_pool, verbose=False, plot=False,)
        train_pool=None
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

        if self.type_of_estimator == 'classifier':
            predicts = np.round(self.model.predict(X),0)
        elif self.type_of_estimator == 'regression':
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

    def get_feature_importance(self, train_x,):
        """
        Return:
            list feature_importance
        """
        if not self._is_possible_feature_importance(): 
            raise Exception("Model cannot get feature_importance")
        fe_lst = self.model.get_feature_importance()
        return (pd.DataFrame(fe_lst, index=train_x.columns, columns=['value']))


class CatBoostClassifier(CatBoost):
    type_of_estimator='classifier'
    __name__ = 'CatBoostClassifier'


class CatBoostRegressor(CatBoost):
    type_of_estimator='regression'
    __name__ = 'CatBoostRegressor'