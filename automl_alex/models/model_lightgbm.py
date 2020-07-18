from .base import *
import lightgbm as lgb
import numpy as np
import pandas as pd


class LightGBM(ModelBase):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """
    __name__ = 'LightGBM'
    
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
        model_param = {'random_seed': self._random_state,
                            'early_stopping_rounds': 50,
                            'num_iterations': 200,
                            'verbose': -1,
                            'device_type': 'gpu' if self._gpu else 'cpu',
                            }
        
        if self.type_of_estimator == 'classifier':
            model_param['objective'] = 'binary'

        if self.wrapper_params['early_stopping']:
            model_param['num_iterations'] = 1000
            if self.metric is not None:
                if self.metric.__name__ == 'roc_auc_score':
                    model_param['metric'] = 'auc'
        return(model_param)

    #@staticmethod
    def get_model_opt_params(self, trial, model, opt_lvl, metric_name):
        """
        Return:
            dict of DistributionWrappers
        """
        model_param = model._init_default_model_param()
        ################################# LVL 1 ########################################
        if opt_lvl == 1:
            model_param['num_leaves'] = trial.suggest_int('lgbm_num_leaves', 2, 50,)
            
        if opt_lvl >= 1:
            if len(self._data.X_train) > 1000:
                model_param['min_child_samples'] = trial.suggest_int('lgbm_min_child_samples', 2, \
                                                                        (len(self._data.X_train)//100))
            else:
                model_param['min_child_samples'] = trial.suggest_int('lgbm_min_child_samples', 2, 7)

        ################################# LVL 2 ########################################
        if opt_lvl == 2:
            model_param['learning_rate'] = trial.suggest_int('lgbm_learning_rate', 1, 11)/100
            model_param['num_leaves'] = trial.suggest_int('lgbm_num_leaves', 2, 50,)
            if not model.wrapper_params['early_stopping']:
                model_param['num_iterations'] = trial.suggest_int('lgbm_num_iterations', 1, 3,)*100

        if opt_lvl >= 2:
            model_param['bagging_fraction'] = trial.suggest_discrete_uniform('lgbm_bagging_fraction', 0.4, 1., 0.1)
            if model_param['bagging_fraction'] < 1.:
                model_param['feature_fraction'] = trial.suggest_discrete_uniform('lgbm_feature_fraction', 0.3, 1., 0.1)
                model_param['bagging_freq'] = trial.suggest_int('lgbm_bagging_freq', 2, 11,)
        
        ################################# LVL 3 ########################################
        if opt_lvl == 3:
            model_param['learning_rate'] = trial.suggest_int('lgbm_learning_rate', 1, 100)/1000
            if not model.wrapper_params['early_stopping']:
                model_param['num_iterations'] = trial.suggest_int('lgbm_num_iterations', 1, 11,)*100
        
        if opt_lvl >= 3:
            model_param['num_leaves'] = trial.suggest_int('lgbm_num_leaves', 2, 100,)
        
        ################################# LVL 4 ########################################
        if opt_lvl == 4:
            model_param['learning_rate'] = trial.suggest_loguniform('lgbm_learning_rate', 1e-3, .1)

        if opt_lvl >= 4:
            model_param['boosting'] = trial.suggest_categorical('lgbm_boosting', ['gbdt', 'dart',])
            if model_param['boosting'] == 'dart':
                model_param['early_stopping_rounds'] = 0
                model_param['uniform_drop'] = trial.suggest_categorical('lgbm_uniform_drop', [True, False])
                model_param['xgboost_dart_mode'] = trial.suggest_categorical('lgbm_xgboost_dart_mode', [True, False])
                model_param['drop_rate'] = trial.suggest_loguniform('lgbm_drop_rate', 1e-8, 1.0)
                model_param['max_drop'] = trial.suggest_int('lgbm_max_drop', 0, 100)
                model_param['skip_drop'] = trial.suggest_loguniform('lgbm_skip_drop', 1e-3, 1.0)

            model_param['num_iterations'] = trial.suggest_int('lgbm_num_iterations', 1, 6,)*1000

            if model.type_of_estimator == 'classifier':
                model_param['objective'] = trial.suggest_categorical('lgbm_objective', 
                    [
                    'binary', 
                    'cross_entropy',
                    ])

            elif model.type_of_estimator == 'regression':
                model_param['objective'] = trial.suggest_categorical('lgbm_objective', 
                    [
                    'regression',
                    'regression_l1',
                    'mape',
                    'huber',
                    'quantile',
                    ])

        ################################# LVL 5 ########################################
        if opt_lvl >= 5:
            model_param['max_cat_threshold'] = trial.suggest_int('lgbm_max_cat_threshold', 1, 100)
            model_param['min_child_weight'] = trial.suggest_loguniform('lgbm_min_child_weight', 1e-6, 1.0)
            model_param['learning_rate'] = trial.suggest_loguniform('lgbm_learning_rate', 1e-5, .1)
            model_param['reg_lambda'] = trial.suggest_loguniform('lgbm_reg_lambda', 1e-8, 1.0)
            model_param['reg_alpha'] = trial.suggest_loguniform('lgbm_reg_alpha', 1e-8, 1.0)
            model_param['max_bin'] = trial.suggest_int('lgbm_max_bin', 1, 5,)*50
            #self.model_param['extra_trees'] = trial.suggest_categorical('lgbm_extra_trees', [True, False])
            model_param['enable_bundle'] = trial.suggest_categorical('lgbm_enable_bundle', [True, False])

        ################################# Other ########################################
        if model.type_of_estimator == 'classifier':
            if metric_name not in ['roc_auc_score', 'log_loss', 'brier_score_loss']:
                model_param['scale_pos_weight'] = trial.suggest_discrete_uniform('lgbm_scale_pos_weight', 0.1, 1., 0.1)

        return(model_param)
            

    def _fit(self, model=None, X_train=None, y_train=None, X_test=None, y_test=None,):
        """
        Args:
            X (pd.DataFrame, shape (n_samples, n_features)): the input data
            y (pd.DataFrame, shape (n_samples, ) or (n_samples, n_outputs)): the target data
        Return:
            model (Class)
        """
        if model is None:
            model = self

        if (X_train is None) or (y_train is None):
            X_train = model._data.X_train
            y_train = model._data.y_train
        
        dtrain = lgb.Dataset(X_train, y_train,)
        params = model.model_param.copy()
        num_iterations = params.pop('num_iterations')

        if model.wrapper_params['early_stopping'] and (X_test is not None):
            dtest = lgb.Dataset(X_test, y_test,)
            model.model = lgb.train(params,
                                    dtrain,
                                    num_boost_round=num_iterations,
                                    valid_sets=(dtrain, dtest),
                                    verbose_eval=False,
                                    )
        else:
            early_stopping_rounds = params.pop('early_stopping_rounds')
            model.model = lgb.train(
                params, 
                dtrain, 
                num_boost_round=num_iterations, 
                verbose_eval=False,
                )

        dtrain=None
        dtest=None
        
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


    def _predict_proba(self, X=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        if X is None:
            X = self._data.X_test

        if self.model is None:
            raise Exception("No fit models")
        if not self.is_possible_predict_proba(): 
            raise Exception("Model cannot predict probability distribution")
        return self.model.predict(X)


    def is_possible_feature_importance(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return True

    def _get_feature_importance(self, train_x, importance_type='gain',):
        """
        Return:
            list feature_importance
        """
        if not self.is_possible_feature_importance(): 
            raise Exception("Model cannot get feature_importance")
        fe_lst = self.model.feature_importance(importance_type=importance_type)
        return (pd.DataFrame(fe_lst, index=train_x.columns))


class LightGBMClassifier(LightGBM):
    type_of_estimator='classifier'


class LightGBMRegressor(LightGBM):
    type_of_estimator='regression'