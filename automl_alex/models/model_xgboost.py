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

    def _init_default_wrapper_params(self,):
        """
        Default wrapper_params
        """
        wrapper_params = {
            'early_stopping':False,
            }
        return(wrapper_params)

    def _init_default_model_param(self,):
        """
        Default model_param
        """
        model_param = {
                        'verbosity': 0,
                        'early_stopping_rounds': 100,
                        'n_estimators': 200,
                        'random_state': self._random_state,
                        }
        if self.wrapper_params['early_stopping']:
            model_param['n_estimators'] = 1000
        return(model_param)

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

    #@staticmethod
    def get_model_opt_params(self, trial, model, opt_lvl, metric_name):
        """
        Return:
            dict of DistributionWrappers
        """
        model_param = model._init_default_model_param()
        ################################# LVL 1 ########################################
        if opt_lvl == 1:
            model_param['max_depth'] = trial.suggest_int('xgb_max_depth', 2, 8,)
            model_param['min_child_weight'] = trial.suggest_int('xgb_min_child_weight', 2, 7)
        
        ################################# LVL 2 ########################################
        if opt_lvl == 2:
            model_param['max_depth'] = trial.suggest_int('xgb_max_depth', 2, 12,)
            model_param['min_child_weight'] = trial.suggest_int('xgb_min_child_weight', 2, 10)
        
        if opt_lvl >= 2:
            model_param['learning_rate'] = trial.suggest_int('xgb_learning_rate', 1, 100)/1000
            model_param['subsample'] = trial.suggest_discrete_uniform('xgb_subsample', 0.1, 1., 0.1)
            model_param['colsample_bytree'] = trial.suggest_discrete_uniform('xgb_colsample_bytree', 0.1, 1., 0.1)
            
        ################################# LVL 3 ########################################
        if opt_lvl >= 3:
            model_param['booster'] = trial.suggest_categorical('xgb_booster', ['gbtree', 'dart', 'gblinear'])
            
            if model_param['booster'] == 'dart' or model_param['booster'] == 'gbtree':
                if len(model._data.X_train) > 1000:
                    model_param['min_child_weight'] = trial.suggest_int('xgb_min_child_weight', 2, (len(model._data.X_train)//100))
                else:
                    model_param['min_child_weight'] = trial.suggest_int('xgb_min_child_weight', 2, 10)
                model_param['max_depth'] = trial.suggest_int('xgb_max_depth', 1, 20)
                model_param['gamma'] = trial.suggest_loguniform('xgb_gamma', 1e-6, 1.0)
                model_param['grow_policy'] = trial.suggest_categorical('xgb_grow_policy', ['depthwise', 'lossguide'])
            
            if model_param['booster'] == 'dart':
                model_param['early_stopping_rounds'] = 0
                model_param['sample_type'] = trial.suggest_categorical('xgb_sample_type', ['uniform', 'weighted'])
                model_param['normalize_type'] = trial.suggest_categorical('xgb_normalize_type', ['tree', 'forest'])
                model_param['rate_drop'] = trial.suggest_loguniform('xgb_rate_drop', 1e-8, 1.0)
                model_param['skip_drop'] = trial.suggest_loguniform('xgb_skip_drop', 1e-8, 1.0)

            if not model.wrapper_params['early_stopping']:
                model_param['n_estimators'] = trial.suggest_int('xgb_n_estimators', 1, 10,)*100

        ################################# LVL 4 ########################################
        if opt_lvl >= 4:
            if model.type_of_estimator == 'regression':
                model_param['objective'] = trial.suggest_categorical('xgb_objective', 
                    [
                    'reg:squarederror',
                    'reg:squaredlogerror',
                    'reg:logistic',
                    ])
            
            model_param['lambda'] = trial.suggest_loguniform('xg_lambda', 1e-8, 1.0)
            model_param['regalpha_alpha'] = trial.suggest_loguniform('XG_alpha', 1e-8, 1.0)
        
        ################################# Other ########################################
        if model.type_of_estimator == 'classifier':
            if model.metric.__name__ not in ['roc_auc_score', 'log_loss', 'brier_score_loss']:
                model_param['scale_pos_weight'] = trial.suggest_discrete_uniform('xgb_scale_pos_weight', 0.1, 1., 0.1)
        return(model_param)


    def _fit(self, model=None, X_train=None, y_train=None, X_test=None, y_test=None,):
        """
        Args:
            X (pd.DataFrame, shape (n_samples, n_features)): the input data
            y (pd.DataFrame, shape (n_samples, ) or (n_samples, n_outputs)): the target data
        Return:
            self
        """
        if model is None:
            model = self

        if (X_train is None) or (y_train is None):
            X_train = model._data.X_train
            y_train = model._data.y_train

        params = model.model_param.copy()
        early_stopping_rounds = params.pop('early_stopping_rounds')

        model.model = model._init_model(model_param=params)
        if model.wrapper_params['early_stopping'] and (X_test is not None):
            model.model.fit(
                X_train, 
                y_train,
                eval_set=(X_test, y_test,),
                early_stopping_rounds=early_stopping_rounds,
                verbose=False,
                )
        else:
            model.model.fit(
                X_train, 
                y_train,
                verbose=False,
                )
        return model

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
            X = self._data.X_test

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
        
        if X is None:
            X = self._data.X_test

        if not self.is_possible_predict_proba(): 
            raise Exception("Model cannot predict probability distribution")

        if self.wrapper_params['early_stopping']:
            predicts = self.model.predict(X, ntree_limit=self.model.best_ntree_limit)
        else:
            predicts = self.model.predict(X)
        return predicts


class XGBoostClassifier(XGBoost):
    type_of_estimator='classifier'


class XGBoostRegressor(XGBoost):
    type_of_estimator='regression'