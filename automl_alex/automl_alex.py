from sklearn.metrics import *
from tqdm import tqdm
import pandas as pd
import time
import joblib
import automl_alex
from .models import *
from .cross_validation import *
from .data_prepare import *
from .encoders import *
from .optimizer import *
from pathlib import Path
from .logger import *

TMP_FOLDER = '.automl-alex_tmp/'

##################################### BestSingleModel ################################################

# in progress...

##################################### ModelsReview ################################################


class ModelsReview(object):
    """
    ModelsReview...
    """
    __name__ = 'ModelsReview'

    def __init__(self,  
                    type_of_estimator=None, # classifier or regression
                    metric=None,
                    metric_round=4,
                    gpu=False, 
                    random_state=42
                    ):
        self._gpu = gpu
        self._random_state = random_state
        if type_of_estimator is not None:
            self.type_of_estimator = type_of_estimator

        if metric is None:
            if self.type_of_estimator == 'classifier':
                self._metric = sklearn.metrics.roc_auc_score
            elif self.type_of_estimator == 'regression':
                self._metric = sklearn.metrics.mean_squared_error
        else:
            self._metric = metric
        
        self._metric_round = metric_round


    @logger.catch
    def fit(self,
        X_train=None, 
        y_train=None, 
        X_test=None, 
        y_test=None,
        models_names=None,
        verbose=3,
        ):
        """
        Fit models (in list models_names) whis default params
        """
        logger_print_lvl(verbose)
        result = pd.DataFrame(columns=['Model_Name', 'Score', 'Time_Fit_Sec'])
        score_ls = []
        time_ls = []
        if models_names is None:
            self.models_names = automl_alex.models.all_models.keys()
        else:
            self.models_names = models_names

        result['Model_Name'] = self.models_names
        
        if verbose > 0:
            disable_tqdm = False
        else: 
            disable_tqdm = True
        for model_name in tqdm(self.models_names, disable=disable_tqdm):
            # Model
            start_time = time.time()
            model_tmp = automl_alex.models.all_models[model_name](
                                            gpu=self._gpu, 
                                            random_state=self._random_state,
                                            type_of_estimator=self.type_of_estimator)
            # fit
            model_tmp.fit(X_train, y_train)
            # Predict
            if (self._metric.__name__ in predict_proba_metrics) and (model_tmp.is_possible_predict_proba()):
                y_pred = model_tmp.predict_proba(X_test)
            else:
                y_pred = model_tmp.predict(X_test)

            score_model = round(self._metric(y_test, y_pred), self._metric_round)
            score_ls.append(score_model)
            iter_time = round((time.time() - start_time),2)
            time_ls.append(iter_time)
            model_tmp = None

        result['Score'] = score_ls
        result['Time_Fit_Sec'] = time_ls
        self.result = result
        return(result)

class ModelsReviewClassifier(ModelsReview):
    type_of_estimator='classifier'


class ModelsReviewRegressor(ModelsReview):
    type_of_estimator='regression'


##################################### BestSingleModel ##################################

class BestSingleModel(Optimizer):
    '''
    Trying to find which model work best on our data
    '''
    __name__ = 'BestSingleModel'


class BestSingleModelClassifier(BestSingleModel):
    type_of_estimator='classifier'


class BestSingleModelRegressor(BestSingleModel):
    type_of_estimator='regression'


##################################### Stacking #########################################

# in progress...

##################################### AutoML #########################################

class AutoML(object):
    '''
    in progress AutoML
    '''
    __name__ = 'AutoML'
    de = None

    def __init__(self,  
                type_of_estimator=None, # classifier or regression
                metric=None,
                metric_round=4,
                combined_score_opt=True,
                gpu=False, 
                random_state=42,
                ):
        self._gpu = gpu
        self._random_state = random_state

        if type_of_estimator is not None:
            self.type_of_estimator = type_of_estimator

        if metric is not None:
            self.metric = metric
        else:
            if self.type_of_estimator == 'classifier':
                self.metric = sklearn.metrics.roc_auc_score
                self.direction = 'maximize'
            elif self.type_of_estimator == 'regression':
                self.metric = sklearn.metrics.mean_squared_error
                self.direction = 'minimize'

        self._combined_score_opt = combined_score_opt
        self._metric_round = metric_round


    @logger.catch
    def fit(self,
        X, 
        y,
        timeout=500, # optimization time in seconds
        auto_parameters=True,
        folds=7,
        score_folds=2,
        opt_lvl=2,
        early_stoping=100,
        feature_selection=True,
        verbose=3,
        ):
        logger_print_lvl(verbose)
        X_source = X.copy()
        ####################################################
        # STEP 0
        start_step_0 = time.time()
        logger.info('> Start Fit Base Model')
        if timeout < 400:
            logger.warning("! Not enough time to find the optimal parameters. \n \
                    Please, Increase the 'timeout' parameter for normal optimization. (min 500 sec)")

        self.cat_features=X.columns[(X.dtypes == 'object') | (X.dtypes == 'category')]
        X[self.cat_features] = X[self.cat_features].astype('str')
        X.fillna(0, inplace=True)
        X[self.cat_features] = X[self.cat_features].astype('category')

        self.model_1 = automl_alex.CatBoost(
            type_of_estimator=self.type_of_estimator, 
            random_state=self._random_state,
            gpu=self._gpu,
            verbose=verbose,
            )
        self.model_1 = self.model_1.fit(X, y, cat_features=self.cat_features.tolist())
        X = None

        ####################################################
        # STEP 1
        start_step_1 = time.time()
        logger.info('> DATA PREPROC')
        self.de_1 = automl_alex.DataPrepare(
            cat_encoder_names=['OneHotEncoder', 'CountEncoder'],
            #outliers_threshold=3,
            normalization=True,
            random_state=self._random_state, 
            verbose=verbose
            )
        X = self.de_1.fit_transform(X_source,)
        if self.de_1.cat_features is not None:
            X = X.drop(self.de_1.cat_features, axis = 1)

        params = {
                'metric' : self.metric,
                'metric_round' :self._metric_round,
                'auto_parameters':auto_parameters,
                'folds':folds,
                'score_folds':score_folds,
                'opt_lvl':opt_lvl,
                'early_stoping':early_stoping,
                'type_of_estimator':self.type_of_estimator,
                'random_state':self._random_state,
                'gpu':self._gpu,
                #'iteration_check': False,
                }

        # logger.info(50*'#')
        # logger.info('> Start Fit Models 2')
        # logger.info(50*'#')
        # # Model 2
        # self.model_2 = automl_alex.BestSingleModel(
        #     models_names = ['LinearModel',],
        #     feature_selection=False,
        #     **params,
        #     )

        # history = self.model_2.opt(X,y, timeout=100, verbose=verbose)
        # self.model_2.save(name='model_2', folder=TMP_FOLDER,)

        logger.info(50*'#')
        logger.info('> Start Fit Models 3')
        logger.info(50*'#')
        # Model 3
        self.model_3 = automl_alex.MLP(
            type_of_estimator=self.type_of_estimator, 
            random_state=self._random_state,
            verbose=verbose,
            )
        self.model_3 = self.model_3.fit(X, y)

        X = None

        total_step_1 = (time.time() - start_step_1)

        ####################################################
        # STEP 2
        start_step_2 = time.time()

        logger.info('> DATA PREPROC')

        self.de_2 = DataPrepare(
            cat_encoder_names=['HelmertEncoder','CountEncoder','HashingEncoder'],
            #outliers_threshold=3,
            normalization=False,
            random_state=self._random_state, 
            verbose=verbose
            )
        X = self.de_2.fit_transform(X_source)
        #X = X.drop(self.de_2.cat_features, axis = 1)

        logger.info(50*'#')
        logger.info('> Start Fit Models 4')

        self.model_4 = automl_alex.CatBoost(
            type_of_estimator=self.type_of_estimator, 
            random_state=self._random_state,
            gpu=self._gpu,
            verbose=verbose,
            )

        self.model_4 = self.model_4.fit(X, y)

        total_step_2 = (time.time() - start_step_2)

        ####################################################
        # STEP 3
        # Model 2 - 3
        start_step_3 = time.time()

        logger.info(50*'#')
        logger.info('> Start Fit Models 5')
        logger.info(50*'#')

        time_to_opt = (timeout - (time.time()-start_step_0)) - 60
        time.sleep(0.1)

        self.model_5 = automl_alex.BestSingleModel(
            models_names = ['LightGBM', 'ExtraTrees'],
            feature_selection=feature_selection,
            **params,
            )

        history = self.model_5.opt(X,y, timeout=time_to_opt, verbose=verbose)
        self.model_5.save(name='model_5', folder=TMP_FOLDER,)

        total_step_4 = (time.time() - start_step_3)

        ####################################################
        logger.info(50*'#')
        logger.info('> Finish!')


    @logger.catch
    def predict(self, X=None, verbose=0):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        if self.model_1 is None:
            raise Exception("No fit models")

        X_source = X.copy()
        ####################################################
        # STEP 0
        X[self.cat_features] = X[self.cat_features].astype('str')
        X.fillna(0, inplace=True)
        X[self.cat_features] = X[self.cat_features].astype('category')

        # MODEL 1
        self.predict_model_1 = self.model_1.predict_or_predict_proba(X)

        ####################################################
        # STEP 1
        X = self.de_1.transform(X_source, verbose=verbose)
        if self.de_1.cat_features is not None:
            X = X.drop(self.de_1.cat_features, axis = 1)

        # MODEL 2
        # self.model_2 = self.model_2.load(name='model_2', folder=TMP_FOLDER,)
        # self.predict_model_2 = self.model_2.predict(X)

        # MODEL 3
        self.predict_model_3 = self.model_3.predict_or_predict_proba(X)

        ####################################################
        # STEP 2
        X = self.de_2.transform(X_source, verbose=verbose)
        #X = X.drop(self.de_2.cat_features, axis = 1)

        # MODEL 4
        self.predict_model_4 = self.model_4.predict_or_predict_proba(X)
        
        # MODEL 5
        self.model_5 = self.model_5.load(name='model_5', folder=TMP_FOLDER,)
        self.predict_model_5 = self.model_5.predict(X)
        
        ####################################################
        # STEP 4
        # Blend
        predicts = (
            self.predict_model_1
            #+self.predict_model_2
            +self.predict_model_3
            +self.predict_model_4
            +self.predict_model_5)/4
        return (predicts)


    @logger.catch
    def save(self, name='AutoML_dump', folder='./'):
        dir_tmp = folder+"AutoML_tmp/"
        Path(dir_tmp).mkdir(parents=True, exist_ok=True)
        self.de_1.save(name='DataPrepare_1_dump', folder=dir_tmp)
        self.de_2.save(name='DataPrepare_2_dump', folder=dir_tmp)
        joblib.dump(self, dir_tmp+'AutoML'+'.pkl')
        #self.model_2.save(name='model_2', folder=dir_tmp,)
        self.model_5.save(name='model_5', folder=dir_tmp,)
        shutil.make_archive(folder+name, 'zip', dir_tmp)
        shutil.rmtree(dir_tmp)
        logger.info('Save AutoML')


    @logger.catch
    def load(self, name='AutoML_dump', folder='./'):
        dir_tmp = folder+"AutoML_tmp/"
        Path(dir_tmp).mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(folder+name+'.zip', dir_tmp)
        model = joblib.load(dir_tmp+'AutoML'+'.pkl')
        model.de_1 = DataPrepare()
        model.de_1 = model.de_1.load('DataPrepare_1_dump', folder=dir_tmp)
        model.de_2 = DataPrepare()
        model.de_2 = model.de_2.load('DataPrepare_2_dump', folder=dir_tmp)
        #model.model_2 = model.model_2.load(name='model_2', folder=dir_tmp,)
        model.model_5 = model.model_5.load(name='model_5', folder=dir_tmp,)
        shutil.rmtree(dir_tmp)
        logger.info('Load AutoML')
        return(model)


class AutoMLClassifier(AutoML):
    type_of_estimator='classifier'
    __name__ = 'AutoMLClassifier'

class AutoMLRegressor(AutoML):
    type_of_estimator='regression'
    __name__ = 'AutoMLRegressor'