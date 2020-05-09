from .base import *
import xgboost as xgb
import numpy as np


class XGBoost(ModelBase):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """
    __name__ = 'XGBoost'

    def _init_wrapper_params(self, wrapper_params=None):
        """
        Default wrapper_params
        """
        self.wrapper_params = {
            'need_norm_data':False,
            'early_stopping':False,
            }
        if wrapper_params is not None:
            self.wrapper_params = wrapper_params

    def _init_model_param(self, model_param=None):
        """
        Default model_param
        """
        if model_param is None:
            self.model_param = {
                                'verbosity': 0,
                                'early_stopping_rounds': 100,
                                'n_estimators': 200,
                                'random_state': self._random_state,
                                }
            if self.wrapper_params['early_stopping']:
                self.model_param['n_estimators'] = 1000
        else:
            self.model_param = model_param
    
    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self.type_of_estimator == 'classifier':
            model = xgb.XGBClassifier(**model_param)
        elif self.type_of_estimator == 'regression':
            model = xgb.XGBRegressor(**model_param)
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
            self.model_param['max_depth'] = trial.suggest_int('xgb_max_depth', 2, 8,)
            self.model_param['min_child_weight'] = trial.suggest_int('xgb_min_child_weight', 1, 10)
        
        ################################# LVL 2 ########################################
        if self._opt_lvl == 2:
            self.model_param['max_depth'] = trial.suggest_int('xgb_max_depth', 2, 12,)
            self.model_param['min_child_weight'] = trial.suggest_int('xgb_min_child_weight', 1, (len(self.X_train)//100))
                       
        if self._opt_lvl >= 2:
            self.model_param['learning_rate'] = trial.suggest_int('xgb_learning_rate', 1, 100)/1000
            self.model_param['subsample'] = trial.suggest_discrete_uniform('xgb_subsample', 0.1, 1., 0.1)
            self.model_param['colsample_bytree'] = trial.suggest_discrete_uniform('xgb_colsample_bytree', 0.1, 1., 0.1)
            
        ################################# LVL 3 ########################################
        if self._opt_lvl >= 3:
            self.model_param['booster'] = trial.suggest_categorical('xgb_booster', ['gbtree', 'dart', 'gblinear'])
            
            if self.model_param['booster'] == 'dart' or self.model_param['booster'] == 'gbtree':
                self.model_param['min_child_weight'] = trial.suggest_int('xg_min_child_weight', 1, (len(self.X_train)//100))
                self.model_param['max_depth'] = trial.suggest_int('xgb_max_depth', 1, 20)
                self.model_param['gamma'] = trial.suggest_loguniform('xgb_gamma', 1e-6, 1.0)
                self.model_param['grow_policy'] = trial.suggest_categorical('xgb_grow_policy', ['depthwise', 'lossguide'])
            
            if self.model_param['booster'] == 'dart':
                self.model_param['early_stopping_rounds'] = 0
                self.model_param['sample_type'] = trial.suggest_categorical('xgb_sample_type', ['uniform', 'weighted'])
                self.model_param['normalize_type'] = trial.suggest_categorical('xgb_normalize_type', ['tree', 'forest'])
                self.model_param['rate_drop'] = trial.suggest_loguniform('xgb_rate_drop', 1e-8, 1.0)
                self.model_param['skip_drop'] = trial.suggest_loguniform('xgb_skip_drop', 1e-8, 1.0)

            if not self.wrapper_params['early_stopping']:
                self.model_param['n_estimators'] = trial.suggest_int('xgb_n_estimators', 1, 10,)*100

        ################################# LVL 4 ########################################
        if self._opt_lvl >= 4:
            if self.type_of_estimator == 'regression':
                self.model_param['objective'] = trial.suggest_categorical('xgb_objective', 
                    [
                    'reg:squarederror',
                    'reg:squaredlogerror',
                    'reg:logistic',
                    ])
            
            self.model_param['lambda'] = trial.suggest_loguniform('xg_lambda', 1e-8, 1.0)
            self.model_param['regalpha_alpha'] = trial.suggest_loguniform('XG_alpha', 1e-8, 1.0)
        
        ################################# Other ########################################
        if self.type_of_estimator == 'classifier':
            if self.metric.__name__ not in ['roc_auc_score', 'log_loss', 'brier_score_loss']:
                self.model_param['scale_pos_weight'] = trial.suggest_discrete_uniform('xgb_scale_pos_weight', 0.1, 1., 0.1)


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

        params = self.model_param.copy()
        early_stopping_rounds = params.pop('early_stopping_rounds')
       
        self.model = self._init_model(model_param=params)
        if self.wrapper_params['early_stopping'] and (X_test is not None):
            self.model.fit(
                X_train, 
                y_train,
                eval_set=(X_test, y_test,),
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
                )
        else:
            self.model.fit(
                X_train, 
                y_train,
                verbose=False,
                )

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
            if self.wrapper_params['early_stopping']:
                predicts = np.round(self.model.predict(X, ntree_limit=self.model.best_ntree_limit),0)
            else:
                predicts = np.round(self.model.predict(X),0)

        elif self.type_of_estimator == 'regression':
            if self.wrapper_params['early_stopping']:
                predicts = self.model.predict(X, ntree_limit=self.model.best_ntree_limit)
            else:
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

        if self.wrapper_params['early_stopping']:
            predicts = self.model.predict(X, ntree_limit=self.model.best_ntree_limit)
        else:
            predicts = self.model.predict(X)
        return predicts


class XGBoostClassifier(XGBoost):
    type_of_estimator='classifier'
    __name__ = 'XGBoostClassifier'


class XGBoostRegressor(XGBoost):
    type_of_estimator='regression'
    __name__ = 'XGBoostRegressor'