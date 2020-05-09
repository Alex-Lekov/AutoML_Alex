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

    def _init_wrapper_params(self, wrapper_params=None):
        """
        Default wrapper_params
        """
        self.wrapper_params = {
            'need_norm_data': False,
            'early_stopping': True,
            }
        if wrapper_params is not None:
            self.wrapper_params = wrapper_params

    def _init_model_param(self, model_param=None):
        """
        Default model_param
        """
        if model_param is None:
            self.model_param = {'verbose': 0, 
                                'early_stopping_rounds': 50,
                                'task_type': 'GPU' if self._gpu else 'CPU',
                                'random_seed': self._random_state,
                                }
            if self.wrapper_params['early_stopping']:
                self.model_param['iterations'] = 1000
                if self.metric is not None:
                    if self.metric.__name__ == 'roc_auc_score':
                        self.model_param['eval_metric'] = 'AUC'
        else:
            self.model_param = model_param


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


    @staticmethod
    def get_model_opt_params(self, trial):
        """
        Return:
            dict of DistributionWrappers
        """
        self._init_model_param()
        ################################# LVL 1 ########################################
        if self._opt_lvl == 1:
            self.model_param['depth'] = trial.suggest_categorical('cb_depth', [6, 10])

        if self._opt_lvl >= 1:
            self.model_param['min_child_samples'] = trial.suggest_int('cb_min_child_samples', 1, \
                                                                        (len(self.X_train)//100))

       ################################# LVL 2 ########################################
        if self._opt_lvl == 2:
            self.model_param['depth'] = trial.suggest_categorical('cb_depth', [4, 6, 10])

        if self._opt_lvl >= 2:
            self.model_param['bagging_temperature'] = trial.suggest_int('cb_bagging_temperature', 0, 10,)
            self.model_param['subsample'] = trial.suggest_discrete_uniform('cb_subsample', 0.1, 1.0, 0.1)
        
        ################################# LVL 3 ########################################
        if self._opt_lvl == 3:
            self.model_param['depth'] = trial.suggest_categorical('cb_depth', [4, 6, 8, 10])

        if self._opt_lvl >= 3:
            if self.type_of_estimator == 'classifier':
                self.model_param['objective'] = trial.suggest_categorical('cbobjective', 
                        [
                        'Logloss', 
                        'CrossEntropy',
                        ])

            elif self.type_of_estimator == 'regression':
                self.model_param['objective'] = trial.suggest_categorical('cb_objective', 
                    [
                    'MAE',
                    'MAPE',
                    'Quantile',
                    'RMSE',
                    ])

            if self.model_param['objective'] == 'Logloss':
                if self.metric.__name__ not in ['roc_auc_score', 'log_loss', 'brier_score_loss']:
                    self.model_param['scale_pos_weight'] = trial.suggest_discrete_uniform('cb_scale_pos_weight', 0.1, 1., 0.1)
        

        ################################# LVL 4 ########################################
        if self._opt_lvl >= 4:
            self.model_param['depth'] = trial.suggest_int('cb_depth', 2, 16)
            self.model_param['l2_leaf_reg'] = trial.suggest_loguniform('cb_l2_leaf_reg', 1e-8, .1)
            self.model_param['learning_rate'] = trial.suggest_int('cb_learning_rate', 1, 100)/1000
            
            if not self.wrapper_params['early_stopping']:
                self.model_param['iterations'] = trial.suggest_int('cb_iterations', 1, 10)*100

        ################################# Other ########################################

    def _fit(self, X_train=None, y_train=None, X_test=None, y_test=None,):
        """
        Args:
            X (pd.DataFrame, shape (n_samples, n_features)): the input data
            y (pd.DataFrame, shape (n_samples, ) or (n_samples, n_outputs)): the target data
        Return:
            self
        """
        if (X_train is None) or (y_train is None):
            X_train = self.X_train
            y_train = self.y_train

        train_pool = Pool(X_train, label=y_train,)

        if self.wrapper_params['early_stopping'] and (X_test is not None):
            validate_pool = Pool(X_test, label=y_test,)
            self.model = self._init_model(model_param=self.model_param)
            self.model.fit(train_pool, 
                            eval_set = validate_pool,
                            use_best_model = True,
                            verbose = False,
                            plot=False,)
        else:
            params = self.model_param.copy()
            early_stopping_rounds = params.pop('early_stopping_rounds')
            self.model = self._init_model(model_param=params)
            self.model.fit(train_pool, verbose=False, plot=False,)

        return self

    def _predict(self, X=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        if self.model is None:
            raise Exception("No fit models")

        if X is None:
            X = self.X_test

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

    def _predict_proba(self, X):
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


class CatBoostClassifier(CatBoost):
    type_of_estimator='classifier'
    __name__ = 'CatBoostClassifier'


class CatBoostRegressor(CatBoost):
    type_of_estimator='regression'
    __name__ = 'CatBoostRegressor'