from automl_alex.base import ModelBase
import xgboost as xgb
import numpy as np


class XGBoost(ModelBase):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """
    __name__ = 'XGBoost'

    def _init_default_model_param(self,):
        """
        Default model_param
        """
        model_param = {
                        'verbosity': 0,
                        'n_estimators': 300,
                        'random_state': self._random_state,
                        }
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
    def get_model_opt_params(self, trial, opt_lvl):
        """
        Return:
            dict of DistributionWrappers
        """
        model_param = self._init_default_model_param()
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
                model_param['min_child_weight'] = trial.suggest_int('xgb_min_child_weight', 2, 100)
                model_param['max_depth'] = trial.suggest_int('xgb_max_depth', 1, 20)
                model_param['gamma'] = trial.suggest_loguniform('xgb_gamma', 1e-6, 1.0)
                model_param['grow_policy'] = trial.suggest_categorical('xgb_grow_policy', ['depthwise', 'lossguide'])
            
            if model_param['booster'] == 'dart':
                model_param['early_stopping_rounds'] = 0
                model_param['sample_type'] = trial.suggest_categorical('xgb_sample_type', ['uniform', 'weighted'])
                model_param['normalize_type'] = trial.suggest_categorical('xgb_normalize_type', ['tree', 'forest'])
                model_param['rate_drop'] = trial.suggest_loguniform('xgb_rate_drop', 1e-8, 1.0)
                model_param['skip_drop'] = trial.suggest_loguniform('xgb_skip_drop', 1e-8, 1.0)

            model_param['n_estimators'] = trial.suggest_int('xgb_n_estimators', 1, 10,)*100

        ################################# LVL 4 ########################################
        if opt_lvl >= 4:
            if self.type_of_estimator == 'regression':
                model_param['objective'] = trial.suggest_categorical('xgb_objective', 
                    [
                    'reg:squarederror',
                    'reg:squaredlogerror',
                    'reg:logistic',
                    ])
            
            model_param['lambda'] = trial.suggest_loguniform('xg_lambda', 1e-8, 1.0)
            model_param['regalpha_alpha'] = trial.suggest_loguniform('XG_alpha', 1e-8, 1.0)
        
        ################################# Other ########################################
        return(model_param)

    
    def _is_model_start_opt_params(self,):
        return(False)


    def fit(self, X_train=None, y_train=None, cat_features=None):
        """
        Args:
            X (pd.DataFrame, shape (n_samples, n_features)): the input data
            y (pd.DataFrame, shape (n_samples, ) or (n_samples, n_outputs)): the target data
        Return:
            self (Class)
        """
        params = self.model_param.copy()

        self.model = self._init_model(model_param=params)

        self.model.fit(
            X_train, 
            y_train,
            verbose=False,
            )
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

    def predict_proba(self, X):
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

        predicts = self.model.predict(X)
        return predicts

    def _is_possible_feature_importance(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return False


class XGBoostClassifier(XGBoost):
    type_of_estimator='classifier'


class XGBoostRegressor(XGBoost):
    type_of_estimator='regression'