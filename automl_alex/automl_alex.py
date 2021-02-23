from sklearn.metrics import *
from tqdm import tqdm
import pandas as pd
import time
import joblib
from .models import *
from .data_prepare import *
from .encoders import *
from pathlib import Path


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
    

    def fit(self,
        X_train=None, 
        y_train=None, 
        X_test=None, 
        y_test=None,
        models_names=None,
        verbose=1,
        ):
        """
        Fit models (in list models_names) whis default params
        """

        result = pd.DataFrame(columns=['Model_Name', 'Score', 'Time_Fit_Sec'])
        score_ls = []
        time_ls = []
        if models_names is None:
            self.models_names = all_models.keys()
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
            model_tmp = all_models[model_name](
                                            gpu=self._gpu, 
                                            random_state=self._random_state,
                                            type_of_estimator=self.type_of_estimator)
            # fit
            model_tmp = model_tmp.fit(X_train, y_train)
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
                combined_score_opt=False,
                gpu=False, 
                random_state=42
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


    def fit(self,
        X, 
        y,
        timeout=200, # optimization time in seconds
        auto_parameters=True,
        folds=10,
        score_folds=2,
        opt_lvl=2,
        early_stoping=100,
        feature_selection=True,
        verbose=1,
        ):
        ####################################################
        # STEP 1
        # DATA PREPROC
        start_step_1 = time.time()

        self.de = DataPrepare(
            cat_encoder_names=['HelmertEncoder','CountEncoder','HashingEncoder'],
            #outliers_threshold=3,
            normalization=False,
            random_state=self._random_state, 
            verbose=verbose
            )
        X = self.de.fit_transform(X)

        total_step_1 = (time.time() - start_step_1)

        ####################################################
        # STEP 2
        # Model 1
        start_step_2 = time.time()
        if verbose > 0:
            print('> Start Fit Base Models')

        self.model_1 = CatBoost(
            type_of_estimator=self.type_of_estimator, 
            random_state=self._random_state,
            gpu=self._gpu,
            )
        self.model_1 = self.model_1.fit(X, y)

        total_step_2 = (time.time() - start_step_2)

        ####################################################
        # STEP 3
        # Model 4
        start_step_3 = time.time()
        if verbose > 0:
            print(50*'#')
            print('> Start Opt Model')

        time_to_opt = (timeout - (time.time()-start_step_1)) - 60
        time.sleep(0.1)

        self.model_4 = LightGBM(
            type_of_estimator=self.type_of_estimator, 
            random_state=self._random_state,
            gpu=self._gpu,
            )

        params = {'X':X,
                'y':y,
                'metric' : self.metric,
                'combined_score_opt' : self._combined_score_opt,
                'metric_round' :self._metric_round,
                'auto_parameters':auto_parameters,
                'folds':folds,
                'score_folds':score_folds,
                'opt_lvl':opt_lvl,
                'early_stoping':early_stoping,
                'feature_selection':feature_selection,
                'timeout':time_to_opt, # optimization time in seconds
                'verbose':verbose}

        try:
            history = self.model_4.opt(**params)
            
            self._select_features_model_4 = list(history['columns'].iloc[0].values)
            self.best_params_model_4 = history['model_param'].iloc[0]

        except:
            self._select_features_model_4 = list(X.columns.values)
            self.best_params_model_4 = self.model_4._init_default_model_param()
            print("Not enough time to find the optimal parameters. \n \
                    Please, Increase the 'timeout' parameter for normal optimization.")

        self.model_4 = LightGBM(
            type_of_estimator=self.type_of_estimator, 
            random_state=self._random_state,
            gpu=self._gpu,
            )
        self.model_4.model_param = self.best_params_model_4
        self.model_4 = self.model_4.fit(X[self._select_features_model_4], y, model_param=self.best_params_model_4)

        total_step_3 = (time.time() - start_step_3)

        ####################################################
        # STEP 4
        # Model 2 - 3
        self.scaler = StandardScaler().fit(X)
        X = self.scaler.transform(X)

        # self.model_2 = LinearModel(
        #     type_of_estimator=self.type_of_estimator, 
        #     random_state=self._random_state,
        #     )
        # self.model_2 = self.model_2.fit(X, y)

        # Model 3
        self.model_3 = MLP(
            type_of_estimator=self.type_of_estimator, 
            random_state=self._random_state
            )
        self.model_3 = self.model_3.fit(X, y)

        if verbose > 0:
            print(50*'#')
            print('> Finish!')

        return(self)


    def predict(self, X=None, verbose=0):
        """
        Args:
            X (np.array, shape (n_samples, n_features)): the input data
        Return:
            np.array, shape (n_samples, n_classes)
        """
        if self.model_1 is None:
            raise Exception("No fit models")

        ####################################################
        # STEP 1
        # DATA PREPROC
        X = self.de.transform(X, verbose=verbose)

        ####################################################
        # STEP 2
        # Models

        if self.type_of_estimator == 'classifier':
            predict_model_1 = self.model_1.predict_proba(X)
            predict_model_4 = self.model_4.predict_proba(X[self._select_features_model_4])
            X = self.scaler.transform(X)
            #predict_model_2 = self.model_2.predict_proba(X)
            predict_model_3 = self.model_3.predict_proba(X)
        else:
            predict_model_1 = self.model_1.predict(X)
            predict_model_4 = self.model_4.predict(X[self._select_features_model_4])
            X = self.scaler.transform(X)
            #predict_model_2 = self.model_2.predict(X)
            predict_model_3 = self.model_3.predict(X)

        ####################################################
        # STEP 3
        # Blend
        predicts = ((predict_model_1*2)+predict_model_3+predict_model_4)/4
        return predicts


    def save(self, name='AutoML_dump', folder='./'):
        dir_tmp = folder+"AutoML_tmp/"
        Path(dir_tmp).mkdir(parents=True, exist_ok=True)
        self.de.save(name='DataPrepare_dump', folder=dir_tmp)
        joblib.dump(self, dir_tmp+'AutoML'+'.pkl')
        shutil.make_archive(dir+name, 'zip', dir_tmp)
        shutil.rmtree(dir_tmp)
        print('Save AutoML')


    def load(self, name='AutoML_dump', folder='./'):
        dir_tmp = folder+"AutoML_tmp/"
        Path(dir_tmp).mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(dir+name+'.zip', dir_tmp)
        model = joblib.load(dir_tmp+'AutoML'+'.pkl')
        model.de = DataPrepare()
        model.de = model.de.load('DataPrepare_dump', folder=dir_tmp)
        shutil.rmtree(dir_tmp)
        print('Load AutoML')
        return(model)


class AutoMLClassifier(AutoML):
    type_of_estimator='classifier'
    __name__ = 'AutoMLClassifier'

class AutoMLRegressor(AutoML):
    type_of_estimator='regression'
    __name__ = 'AutoMLRegressor'