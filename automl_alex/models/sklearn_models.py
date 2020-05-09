import sklearn
from sklearn import ensemble, neural_network, linear_model, svm, neighbors
from .base import *

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

################################## LogRegClassifier ##########################################################

class LinearModel(ModelBase):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """
    __name__ = 'LinearModel'

    def _init_wrapper_params(self, wrapper_params=None):
        """
        Default wrapper_params
        """
        self.wrapper_params = {
                'need_norm_data':True,
                'scaler_name':'StandardScaler',
                }
        if wrapper_params is not None:
            self.wrapper_params = wrapper_params

    def _init_model_param(self, model_param=None):
        """
        Default model_param
        """
        if model_param is None:
            self.model_param = {'verbose':0,}
        else: 
            self.model_param = model_param

    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self.type_of_estimator == 'classifier':
            model = linear_model.LogisticRegression(**model_param)
        elif self.type_of_estimator == 'regression':
            model = linear_model.LinearRegression(**model_param)
        return(model)

    @staticmethod
    def get_model_opt_params(self, trial):
        """
        Return:
            dict of DistributionWrappers
        """
        # if need norm data
        if self.wrapper_params['need_norm_data']:
            self._need_scaler_params(trial)

        # model Default params
        self._init_model_param()

        self.model_param['fit_intercept'] = trial.suggest_categorical('lr_fit_intercept',[True, False])
        if self.type_of_estimator == 'classifier':
            ################################# LVL 1 ########################################
            if self._opt_lvl >= 1:
                self.model_param['C'] = trial.suggest_uniform('lr_C', 0.0, 100.0)
            
            ################################# LVL 2 ########################################
            if self._opt_lvl >= 2:
                self.model_param['solver'] = trial.suggest_categorical('lr_solver', ['lbfgs', 'saga', 'liblinear'])
                self.model_param['tol'] = trial.suggest_uniform('lr_tol', 1e-6, 0.1)
                self.model_param['class_weight'] = trial.suggest_categorical('lr_class_weight',[None, 'balanced'])

                if self.model_param['solver'] == 'saga':
                    self.model_param['penalty'] = trial.suggest_categorical('lr_penalty', ['l1', 'l2', 'elasticnet',])
                    if self.model_param['penalty'] == 'elasticnet':
                        self.model_param['l1_ratio'] = trial.suggest_uniform('lr_l1_ratio', 0.0, 1.0)
                        self.model_param['max_iter'] = 5000
                if self.model_param['solver'] == 'liblinear':
                    self.model_param['n_jobs'] = 1
                if self.model_param['solver'] == 'lbfgs':
                    self.model_param['max_iter'] = 5000

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
            
        self.model = self._init_model(model_param=self.model_param)
        self.model.fit(X_train, y_train,)
        return self

    def _predict(self, X_test=None):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """           
        if self.model is None:
            raise Exception("No fit models")
        
        if X_test is None:
            X_test = self.X_test

        return self.model.predict(X_test)

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return True

    def _predict_proba(self, X_test=None):
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
    type_of_estimator='classifier'
    __name__ = 'LogisticRegressionClassifier'


class LinearRegression(LinearModel):
    type_of_estimator='regression'
    __name__ = 'LinearRegression'


################################## SGD ##########################################################


class SGD(LinearModel):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """
    __name__ = 'SGD'

    def _init_wrapper_params(self, wrapper_params=None):
        """
        Default wrapper_params
        """
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning, message="^Maximum number of iteration reached")
        
        self.wrapper_params = {'need_norm_data': True, 
                                'norm_data': True, 
                                'scaler_name': 'StandardScaler'
                                }
        if wrapper_params is not None:
            self.wrapper_params = wrapper_params

    def _init_model_param(self, model_param=None):
        """
        Default model_param
        """
        if model_param is None:
            self.model_param = {'max_iter': 5000,
                                'verbose':0,
                                'fit_intercept': True,
                                'random_state':self._random_state,}
        else: 
            self.model_param = model_param


    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self.type_of_estimator == 'classifier':
            model = linear_model.SGDClassifier(**model_param)
        elif self.type_of_estimator == 'regression':
            model = linear_model.SGDRegressor(**model_param)
        return(model)

    @staticmethod
    def get_model_opt_params(self, trial):
        """
        Return:
            dict of DistributionWrappers
        """
        # if need norm data
        if self.wrapper_params['need_norm_data']:
            self._need_scaler_params(trial)

        # model Default params
        self._init_model_param()
        ################################# LVL 1 ########################################
        if self._opt_lvl >= 1:
            self.model_param['penalty'] = trial.suggest_categorical('sgb_penalty', ['l2', 'l1', 'elasticnet',])
            self.model_param['tol'] = trial.suggest_uniform('sgb_tol', 1e-6, 0.1)
            self.model_param['alpha'] = trial.suggest_uniform('sgb_alpha', 1e-6, 1.0)
            self.model_param['l1_ratio'] = trial.suggest_uniform('sgb_l1_ratio', 0.0, 1.0)
            self.model_param['eta0'] = trial.suggest_uniform('sgb_eta0', 0.0, 1.0)
            self.model_param['fit_intercept'] = trial.suggest_categorical('sgb_fit_intercept',[True, False])
            self.model_param['early_stopping'] = trial.suggest_categorical('sgb_early_stopping',[True, False])
        
        ################################# LVL 2 ########################################
        if self._opt_lvl >= 2:
            if self.type_of_estimator == 'classifier':
                self.model_param['loss'] = trial.suggest_categorical('sgb_loss', [
                        'log', 
                        'hinge', 
                        'modified_huber', 
                        'squared_hinge', 
                        'perceptron'
                        ])
                self.model_param['class_weight'] = trial.suggest_categorical('sgb_class_weight',[None, 'balanced'])
       
            elif self.type_of_estimator == 'regression':
                self.model_param['loss'] = trial.suggest_categorical('sgb_loss', [
                    'squared_loss', 
                    'huber', 
                    'epsilon_insensitive', 
                    'squared_epsilon_insensitive'
                    ])

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return False


class SGDClassifier(SGD):
    type_of_estimator='classifier'
    __name__ = 'SGDClassifier'


class SGDRegressor(SGD):
    type_of_estimator='regression'
    __name__ = 'SGDRegressor'


########################################### SVM #####################################################


class LinearSVM(LinearModel):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """
    __name__ = 'LinearSVM'

    def _init_wrapper_params(self, wrapper_params=None):
        """
        Default wrapper_params
        """
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning, message="^Liblinear failed to converge")
        
        self.wrapper_params = {'need_norm_data': True, 
                                'norm_data': True, 
                                'scaler_name': 'StandardScaler'
                                }
        if wrapper_params is not None:
            self.wrapper_params = wrapper_params

    def _init_model_param(self, model_param=None):
        """
        Default model_param
        """
        if model_param is None:
            self.model_param = {
                'verbose':0,
                'random_state': self._random_state,
                }
        else: 
            self.model_param = model_param

    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self.type_of_estimator == 'classifier':
            model = svm.LinearSVC(**model_param)
        elif self.type_of_estimator == 'regression':
            model = svm.LinearSVR(**model_param)
        return(model)

    @staticmethod
    def get_model_opt_params(self, trial):
        """
        Return:
            dict of DistributionWrappers
        """
        # if need norm data
        if self.wrapper_params['need_norm_data']:
            self._need_scaler_params(trial)

        # model Default params
        self._init_model_param()
        ################################# LVL 1 ########################################
        if self._opt_lvl >= 1:
            self.model_param['tol'] = trial.suggest_uniform('svc_tol', 1e-7, 1e-2)
            self.model_param['C'] = trial.suggest_uniform('svc_C', 1e-4, 50.0)
            self.model_param['fit_intercept'] = trial.suggest_categorical('svc_fit_intercept',[True, False])
        
        ################################# LVL 2 ########################################
        if self._opt_lvl >= 2:
            if self.type_of_estimator == 'classifier':
                self.model_param['loss'] = trial.suggest_categorical('svc_loss', ['hinge', 'squared_hinge',])
                self.model_param['class_weight'] = trial.suggest_categorical('svc_class_weight',[None, 'balanced'])
                if self.model_param['loss'] == 'squared_hinge':
                    self.model_param['penalty'] = trial.suggest_categorical('svc_penalty', ['l2', 'l1',])
                    if self.model_param['penalty'] == 'l2':
                        self.model_param['dual'] = trial.suggest_categorical('svc_dual',[True, False])
                    if self.model_param['penalty'] == 'l1':
                        self.model_param['dual'] = False

    def is_possible_predict_proba(self):
        """
        Return:
            bool, whether model can predict proba
        """
        return False


class LinearSVClassifier(LinearSVM):
    type_of_estimator='classifier'
    __name__ = 'LinearSVClassifier'


class LinearSVRegressor(LinearSVM):
    type_of_estimator='regression'
    __name__ = 'LinearSVRegressor'


########################################### KNN #####################################################


class KNeighbors(LinearModel):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """
    __name__ = 'KNeighbors'

    def _init_model_param(self, model_param=None):
        """
        Default model_param
        """
        if model_param is None:
            self.model_param = {'n_jobs': -1,}
        else: 
            self.model_param = model_param


    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self.type_of_estimator == 'classifier':
            model = neighbors.KNeighborsClassifier(**model_param)
        elif self.type_of_estimator == 'regression':
            model = neighbors.KNeighborsRegressor(**model_param)
        return(model)

    @staticmethod
    def get_model_opt_params(self, trial):
        """
        Return:
            dict of DistributionWrappers
        """
        # if need norm data
        if self.wrapper_params['need_norm_data']:
            self._need_scaler_params(trial)

        # model Default params
        self._init_model_param()
        # model opt params
        self.model_param['n_neighbors'] = trial.suggest_int('knn_n_neighbors', 2, 150)
        self.model_param['weights'] = trial.suggest_categorical('knn_weights', ['uniform', 'distance',])
        self.model_param['algorithm'] = trial.suggest_categorical('knn_algorithm', ['auto', 'ball_tree', 'kd_tree','brute'])
        self.model_param['p'] = trial.suggest_categorical('knn_p', [1, 2,])
        if self.model_param['algorithm'] == 'ball_tree' or self.model_param['algorithm'] == 'kd_tree':
            self.model_param['leaf_size'] = trial.suggest_int('knn_leaf_size', 2, 150)
        

class KNNClassifier(KNeighbors):
    type_of_estimator='classifier'
    __name__ = 'KNNClassifier'


class KNNRegressor(KNeighbors):
    type_of_estimator='regression'
    __name__ = 'KNNRegressor'


########################################### MLP #####################################################

        
class MLP(LinearModel):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """
    __name__ = 'MLP'
        
    def _init_model_param(self, model_param=None):
        """
        Default model_param
        """
        if model_param is None:
            self.model_param = {
                'verbose':0,
                'random_state': self._random_state,
                'max_iter': 1000,
                'early_stopping': True,
                }
        else: 
            self.model_param = model_param

    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self.type_of_estimator == 'classifier':
            model = neural_network.MLPClassifier(**model_param)
        elif self.type_of_estimator == 'regression':
            model = neural_network.MLPRegressor(**model_param)
        return(model)

    @staticmethod
    def get_model_opt_params(self, trial):
        """
        Return:
            dict of DistributionWrappers
        """
        # if need norm data
        if self.wrapper_params['need_norm_data']:
            self._need_scaler_params(trial)

        # model Default params
        self._init_model_param()
        # model params
        self.model_param.update({
                            'hidden_layer_sizes': trial.suggest_int('mlp_hidden_layer_sizes', 1, 10)*50,
                            'solver': trial.suggest_categorical('mlp_solver', ['sgd', 'adam']),
                            'alpha': trial.suggest_uniform('mlp_alpha', 1e-6, 1.0),
                            'learning_rate': trial.suggest_categorical('mlp_learning_rate', ['adaptive', 'constant', 'invscaling']),
                            'tol': trial.suggest_uniform('mlp_tol', 1e-6, 1e-1),
                            })
        

class MLPClassifier(MLP):
    type_of_estimator='classifier'
    __name__ = 'MLPClassifier'


class MLPRegressor(MLP):
    type_of_estimator='regression'
    __name__ = 'MLPRegressor'


########################################### RandomForest #####################################################


class RandomForest(LinearModel):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """
    __name__ = 'RandomForest'

    def _init_wrapper_params(self, wrapper_params=None):
        """
        Default wrapper_params
        """
        self.wrapper_params = {
                'need_norm_data':False,
                }
        if wrapper_params is not None:
            self.wrapper_params = wrapper_params

    def _init_model_param(self, model_param=None):
        """
        Default model_param
        """
        if model_param is None:
            self.model_param = {
                'verbose': 0,
                'random_state': self._random_state,
                'n_jobs': -1,
                }
        else: 
            self.model_param = model_param

    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self.type_of_estimator == 'classifier':
            model = ensemble.RandomForestClassifier(**model_param)
        elif self.type_of_estimator == 'regression':
            model = ensemble.RandomForestRegressor(**model_param)
        return(model)

    @staticmethod
    def get_model_opt_params(self, trial):
        """
        Return:
            dict of DistributionWrappers
        """
        # model Default params
        self._init_model_param()
        ################################# LVL 1 ########################################
        if self._opt_lvl >= 1:
            self.model_param['min_samples_split'] = trial.suggest_int('rf_min_samples_split', 2, \
                                                                        (len(self.X_train)//100))
            self.model_param['max_depth'] = trial.suggest_int('rf_max_depth', 1, 10,)*10

        ################################# LVL 2 ########################################
        if self._opt_lvl >= 2:
            self.model_param['n_estimators'] = trial.suggest_int('rf_n_estimators', 1, 10,)*100
            self.model_param['max_features'] = trial.suggest_categorical('rf_max_features', [
                'auto', 
                'sqrt', 
                'log2'
                ])
        
        ################################# LVL 3 ########################################
        if self._opt_lvl >= 3:
            self.model_param['bootstrap'] = trial.suggest_categorical('rf_bootstrap',[True, False])
            if self.model_param['bootstrap']:
                self.model_param['oob_score'] = trial.suggest_categorical('rf_oob_score',[True, False])
            if self.type_of_estimator == 'classifier':
                self.model_param['class_weight'] = trial.suggest_categorical('rf_class_weight',[None, 'balanced'])


class RandomForestClassifier(RandomForest):
    type_of_estimator='classifier'
    __name__ = 'RandomForestClassifier'


class RandomForestRegressor(RandomForest):
    type_of_estimator='regression'
    __name__ = 'RandomForestRegressor'


########################################### ExtraTrees #####################################################


class ExtraTrees(LinearModel):
    """
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """
    __name__ = 'ExtraTrees'

    def _init_wrapper_params(self, wrapper_params=None):
        """
        Default wrapper_params
        """
        self.wrapper_params = {
                'need_norm_data':False,
                }
        if wrapper_params is not None:
            self.wrapper_params = wrapper_params

    def _init_model_param(self, model_param=None):
        """
        Default model_param
        """
        if model_param is None:
            self.model_param = {
                'verbose': 0,
                'random_state': self._random_state,
                'n_jobs': -1,
                }
        else: 
            self.model_param = model_param

    def _init_model(self, model_param=None):
        """
        sets new model,
        Args:
            params: : parameters for model.
        """
        if self.type_of_estimator == 'classifier':
            model = ensemble.ExtraTreesClassifier(**model_param)
        elif self.type_of_estimator == 'regression':
            model = ensemble.ExtraTreesRegressor(**model_param)
        return(model)

    @staticmethod
    def get_model_opt_params(self, trial):
        """
        Return:
            dict of DistributionWrappers
        """
        # model Default params
        self._init_model_param()
        ################################# LVL 1 ########################################
        if self._opt_lvl >= 1:
            self.model_param['min_samples_split'] = trial.suggest_int('ext_min_samples_split', 2, \
                                                                        (len(self.X_train)//100))
            self.model_param['max_depth'] = trial.suggest_int('ext_max_depth', 1, 10,)*10

        ################################# LVL 2 ########################################
        if self._opt_lvl >= 2:
            self.model_param['n_estimators'] = trial.suggest_int('ext_n_estimators', 1, 10,)*100
            self.model_param['max_features'] = trial.suggest_categorical('ext_max_features', [
                'auto', 
                'sqrt', 
                'log2'
                ])
        
        ################################# LVL 3 ########################################
        if self._opt_lvl >= 3:
            self.model_param['bootstrap'] = trial.suggest_categorical('ext_bootstrap',[True, False])
            if self.model_param['bootstrap']:
                self.model_param['oob_score'] = trial.suggest_categorical('ext_oob_score',[True, False])
            if self.type_of_estimator == 'classifier':
                self.model_param['class_weight'] = trial.suggest_categorical('ext_class_weight',[None, 'balanced'])


class ExtraTreesClassifier(ExtraTrees):
    type_of_estimator='classifier'
    __name__ = 'ExtraTreesClassifier'


class ExtraTreesRegressor(ExtraTrees):
    type_of_estimator='regression'
    __name__ = 'ExtraTreesRegressor'