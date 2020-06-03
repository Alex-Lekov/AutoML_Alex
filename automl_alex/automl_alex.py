from sklearn.metrics import *
import sklearn
from tqdm import trange
from tqdm import tqdm
import optuna
import pandas as pd
import numpy as np
import time
import sys
from .models import *
from .databunch import DataBunch
from .encoders import *


class AutoMLBase(ModelBase):
    """
    Base class for a specific ML algorithm implementation factory,
    i.e. it defines algorithm-specific hyperparameter space and generic methods for model training & inference
    """
    def fit(self, dataset):
        """
        Args:
            dataset: the input data,
                dataset.y may be None
        Return:
            np.array, shape (n_samples, ): predictions
        """
        raise NotImplementedError("Pure virtual class.")

    def predict(self, dataset):
        """
        Args:
            dataset : the input data,
                dataset.y may be None
        Return:
            np.array, shape (n_samples, ): predictions
        """
        raise NotImplementedError("Pure virtual class.")

    def _init_wrapper_params(self, wrapper_params=None):
        pass

    def _init_model_param(self, model_param=None):
        pass


##################################### BestSingleModel ################################################


class BestSingleModel(XGBoost):
    """
    Trying to find which model work best on our data
    Args:
        params (dict or None): parameters for model.
            If None default params are fetched.
    """
    def _make_model(self, model_name, databunch=None, model_param=None, wrapper_params=None,):
        '''
        Make new model and choose model library from 'model_name'
        '''
        if databunch is None:
            databunch=self._data

        model = all_models[model_name](
            databunch=databunch, 
            cv = self._cv,
            score_cv_folds = self._score_cv_folds,
            opt_lvl=self._opt_lvl,
            metric=self.metric,
            combined_score_opt=self._combined_score_opt,
            metric_round=self._metric_round,
            model_param=model_param, 
            wrapper_params=wrapper_params,
            gpu=self._gpu, 
            random_state=self._random_state,
            type_of_estimator=self.type_of_estimator
            )
        return(model)

    def _opt_model(self, trial):
        '''
        now we can choose models in optimization
        '''
        model_name = trial.suggest_categorical('model_name', self.models_names)
        model = self._make_model(model_name,)
        model.get_model_opt_params(model, trial=trial)
        return(model)

    def opt(self, 
        timeout=1000, 
        early_stoping=100, 
        cold_start=100,
        direction='maximize',
        opt_lvl=3,
        cv=None,
        score_cv_folds=None,
        auto_parameters=True,
        opt_encoders=True,
        cat_encoder_names=None,
        target_cat_encoder_names=None,
        models_names=None, #list models_names for opt
        save_to_sqlite=False,
        verbose=1,
        ):
        '''
        Custom opt func whis list:models_names
        now we can choose models in optimization
        '''
        if cold_start is not None:
            self._cold_start = cold_start
        if self.direction is None:
            self.direction = direction
        if opt_lvl is not None:
            self._opt_lvl = opt_lvl
        if cv is not None:
            self._cv = cv
        if score_cv_folds is not None:
            self._score_cv_folds = score_cv_folds
        if auto_parameters is not None:
            self._auto_parameters = auto_parameters

        # Encoders
        self._opt_encoders_bool = opt_encoders
        if cat_encoder_names is not None:
            self.encoders_names = cat_encoder_names
        if target_cat_encoder_names is not None:
            self.target_encoder_names = target_cat_encoder_names

        if models_names is None:
            self.models_names = all_models.keys()
        else:
            self.models_names = models_names
            
        # Opt
        history = self._opt_core(
            timeout, 
            early_stoping, 
            save_to_sqlite,
            verbose)
        return(history)

    def _predict_preproc_model(self, model_cfg, cv):
        """
        custom function for predict, now we can choose model library
        """
        databunch = self._remake_encode_databunch(
                encoder_name=model_cfg['cat_encoder'], 
                target_encoder_name=model_cfg['target_encoder'],
                )
        model = self._make_model(model_cfg['model_name'], databunch=databunch)
        model.model_param = model_cfg['model_param']
        model.wrapper_params = model_cfg['wrapper_params']
        model._cv = cv
        return(model)


class BestSingleModelClassifier(BestSingleModel):
    type_of_estimator='classifier'


class BestSingleModelRegressor(BestSingleModel):
    type_of_estimator='regression'


##################################### ModelsReview ################################################


class ModelsReview(AutoMLBase):
    """
    ModelsReview...
    """
    __name__ = 'ModelsReview'

    def fit(self, 
        models_names=None,
        verbose=1,
        ):
        """
        Fit models (in list models_names) whis default params
        """
        history_fits = pd.DataFrame()
        if models_names is None:
            self.models_names = all_models.keys()
        else:
            self.models_names = models_names
        
        if verbose > 0:
            disable_tqdm = False
        else: 
            disable_tqdm = True
        for model_name in tqdm(self.models_names, disable=disable_tqdm):
            # Model
            model_tmp = all_models[model_name](databunch=self._data, 
                                            cv=self._cv,
                                            score_cv_folds = self._cv,
                                            metric=self.metric,
                                            direction=self.direction,
                                            metric_round=self._metric_round,
                                            combined_score_opt=self._combined_score_opt,
                                            gpu=self._gpu, 
                                            random_state=self._random_state,
                                            type_of_estimator=self.type_of_estimator)
            # fit
            config = model_tmp.fit()
            history_fits = history_fits.append(config, ignore_index=True)
            model_tmp = None
            self.history_trials_dataframe = history_fits
        return(history_fits)

    def opt(self, 
        timeout=1000, 
        early_stoping=100, 
        auto_parameters=True,
        direction=None,
        opt_encoders=False,
        cat_encoder_names=None,
        target_cat_encoder_names=None,
        verbose=1,
        models_names=None,
        ):
        if direction is not None:
            self.direction = direction
        if self.direction is None:
            raise Exception('Need direction for optimaze!')

        if models_names is None:
            self.models_names = all_models.keys()
        else:
            self.models_names = models_names

        self.history_trials_dataframe = pd.DataFrame()
        self.top1_models_cfgs = pd.DataFrame()
        self.top10_models_cfgs = pd.DataFrame()

        timeout_per_model = timeout//len(self.models_names)
        
        if verbose > 0:
            disable_tqdm = False
        else: disable_tqdm = True
        for model_name in tqdm(self.models_names, disable=disable_tqdm):
            start_unixtime = time.time()
            # Model
            model_tmp = all_models[model_name](databunch=self._data, 
                                            opt_lvl=self._opt_lvl,
                                            cv=self._cv,
                                            score_cv_folds = self._score_cv_folds,
                                            metric=self.metric,
                                            direction=self.direction,
                                            metric_round=self._metric_round,
                                            combined_score_opt=self._combined_score_opt,
                                            gpu=self._gpu, 
                                            random_state=self._random_state,
                                            type_of_estimator=self.type_of_estimator)
            # Opt
            time.sleep(0.1)
            history = model_tmp.opt(timeout=timeout_per_model,
                        early_stoping=early_stoping, 
                        auto_parameters=auto_parameters,
                        opt_encoders=opt_encoders,
                        cat_encoder_names=cat_encoder_names,
                        target_cat_encoder_names=cat_encoder_names,
                        verbose= (lambda x: 0 if x <= 1 else 1)(verbose),
                        )
            if verbose > 0:
                best_score = history.head(1)['model_score'].iloc[0]
                print('\n', model_name, ' Best Score: ', best_score)

            history = history.drop_duplicates(subset=['model_score', 'score_std'], keep='last')
            # Trials
            self.history_trials_dataframe = self.history_trials_dataframe.append(
                history,
                ignore_index=True,
                )

            # Top1:
            self.top1_models_cfgs = self.top1_models_cfgs.append(
                history.head(1), 
                ignore_index=True,
                )

            # Top10:
            self.top10_models_cfgs = self.top10_models_cfgs.append(
                history.head(10), 
                ignore_index=True,
                )

            # time dinamic
            sec_iter = start_unixtime - time.time()
            sec_dev = timeout_per_model - sec_iter
            if sec_dev > 10:
                timeout_per_model = timeout_per_model + (sec_dev // (len(self.models_names)))
            model_tmp = None

        return(self.history_trials_dataframe)

class ModelsReviewClassifier(ModelsReview):
    type_of_estimator='classifier'


class ModelsReviewRegressor(ModelsReview):
    type_of_estimator='regression'


##################################### Stacking #########################################


class Stacking(AutoMLBase):
    '''
    Stack top models
    '''
    __name__ = 'Stacking'

    def opt(self, 
            timeout=2000,
            early_stoping=100,
            cold_start=None,
            opt_lvl=3,
            cv=None,
            score_cv_folds=None,
            auto_parameters=True,
            stack_models_names=None,
            stack_top=20,
            meta_models_names=['MLP',],
            opt_encoders=False,
            cat_encoder_names=None,
            target_cat_encoder_names=None,
            verbose=1,):
        if self.direction is None:
            raise Exception('Need direction for optimaze!')
        if cv is not None:
            self._cv = cv
        if score_cv_folds is not None:
            self._score_cv_folds = score_cv_folds
        if self._cv < 2:
            raise Exception("Stacking no CV? O_o")

        # calc ~ time for opt
        #min_stack_model_timeout = 1000
        metamodel_opt_timeout = (timeout // 100)*20
        predict_timeout = 30*stack_top # time for predict
        if (metamodel_opt_timeout/(len(meta_models_names))) < 100:
            raise Exception(f"Please give me more time to optimize or reduce the number of meta models ('meta_models_names')")
        select_models_timeout = (timeout - predict_timeout) - metamodel_opt_timeout

        if stack_models_names is None:
            self.stack_models_names = all_models.keys()
        else:
            self.stack_models_names = stack_models_names
        self.meta_models_names = meta_models_names

        if verbose > 0:
            print("\n Step1: Opt StackingModels")
            time.sleep(0.2) # clean print 
    
        # Model
        model = BestSingleModel(databunch=self._data,
                                opt_lvl=self._opt_lvl,
                                cv=self._cv,
                                score_cv_folds = self._score_cv_folds,
                                auto_parameters = self._auto_parameters,
                                metric=self.metric,
                                direction=self.direction,
                                metric_round=self._metric_round,
                                combined_score_opt=self._combined_score_opt,
                                gpu=self._gpu, 
                                random_state=self._random_state,
                                type_of_estimator=self.type_of_estimator)

        # Opt
        history = model.opt(
            timeout=select_models_timeout, 
            early_stoping=early_stoping,
            cold_start=cold_start,
            opt_lvl=opt_lvl,
            cv=self._cv,
            direction=self.direction,
            score_cv_folds=score_cv_folds,
            auto_parameters=auto_parameters,
            models_names=self.stack_models_names,
            opt_encoders=opt_encoders,
            cat_encoder_names=cat_encoder_names,
            target_cat_encoder_names=target_cat_encoder_names,
            verbose= (lambda x: 0 if x <= 1 else 1)(verbose), )

        history = history.drop_duplicates(subset=['model_score', 'score_std'], keep='last')
        self.stack_models_cfgs = history.head(stack_top)

        if verbose > 0:
            print("\n Step2: Get new X_train from StackingModels")
            time.sleep(0.2) # clean print 
        # Predict
        predicts = model.predict(models_cfgs=self.stack_models_cfgs)
        self.stack_models_predicts = predicts
        
        # get new X_train
        self._data.X_train_predicts = pd.DataFrame([*predicts['predict_train'].values],).T
        self._data.X_test_predicts = pd.DataFrame([*predicts['predict_test'].values],).T
        
        # Score:
        if verbose > 0:
            score_mean_models = self.metric(self._data.y_train, predicts['predict_train'].mean())
            print(f'\n StackModels Mean {self.metric.__name__} Score Train: ', 
                round(score_mean_models, self._metric_round))
            print("\n Step3: MetaModels")
            time.sleep(0.1) # clean print 

        # Meta model
        x_train = self._data.X_train_predicts
        x_test = self._data.X_test_predicts

        metamodel = ModelsReview(x_train, self._data.y_train, x_test, 
                                clean_and_encod_data=False,
                                cat_encoder_name=None,
                                target_encoder_name=None,
                                clean_nan=False,
                                type_of_estimator=self.type_of_estimator, 
                                random_state=self._random_state,)

        review = metamodel.opt(
            timeout=metamodel_opt_timeout, 
            opt_encoders=False, 
            models_names=meta_models_names, 
            verbose=(lambda x: 0 if x <= 1 else 1)(verbose), 
            )

        models_cfgs = review
        models_cfgs = models_cfgs.drop_duplicates(subset=['model_score', 'score_std'], keep='last')
        stack_metamodels_cfgs = models_cfgs.sort_values('model_score', ascending=True).head(10)
        if metamodel.direction == 'maximize':
            stack_metamodels_cfgs = models_cfgs.sort_values('model_score', ascending=False).head(10)

        metamodel_predicts = metamodel.predict(models_cfgs=stack_metamodels_cfgs)
        meta_pred_test = metamodel_predicts['predict_test'].mean()
        meta_pred_train = metamodel_predicts['predict_train'].mean()
        
        if verbose > 0:
            # Score:
            score_meta_models = self.metric(self._data.y_train, meta_pred_train)
            print(f'MetaModels Mean {self.metric.__name__} Score Train : {score_meta_models}')
            print("\n Finish!")

        return (meta_pred_test, meta_pred_train)

    fit_predict = opt
    

class StackingClassifier(Stacking):
    type_of_estimator='classifier'
    __name__ = 'StackingClassifier'


class StackingRegressor(Stacking):
    type_of_estimator='regression'
    __name__ = 'StackingRegressor'



##################################### AutoML #########################################

class AutoML(AutoMLBase):
    '''
    in progress AutoML
    '''
    __name__ = 'AutoML'

    def opt(self, 
            timeout=2000,
            early_stoping=100,
            cold_start=None,
            opt_lvl=3,
            cv=None,
            score_cv_folds=None,
            auto_parameters=True,
            stack_models_names=None,
            stack_top=10,
            opt_encoders=True,
            cat_encoder_names=None,
            target_cat_encoder_names=None,
            verbose=1,):
        if self.direction is None:
            raise Exception('Need direction for optimaze!')
        if cv is not None:
            self._cv = cv
        if score_cv_folds is not None:
            self._score_cv_folds = score_cv_folds
        if self._cv < 2:
            raise Exception("Stacking no CV? O_o")

        # calc ~ time for opt
        #min_stack_model_timeout = 1000
        predict_timeout = 30*stack_top # time for predict
        select_models_timeout = (timeout - predict_timeout)
        if select_models_timeout < 200:
            raise Exception(f"Please give me more time to optimize or reduce the number of stack models ('stack_top')")

        if stack_models_names is None:
            self.stack_models_names = all_models.keys()
        else:
            self.stack_models_names = stack_models_names

        if verbose > 0:
            print("\n Opt StackingModels")
            time.sleep(0.2) # clean print 
    
        # Model
        model = BestSingleModel(databunch=self._data,
                                opt_lvl=self._opt_lvl,
                                cv=self._cv,
                                score_cv_folds = self._score_cv_folds,
                                auto_parameters = self._auto_parameters,
                                metric=self.metric,
                                direction=self.direction,
                                metric_round=self._metric_round,
                                combined_score_opt=self._combined_score_opt,
                                gpu=self._gpu, 
                                random_state=self._random_state,
                                type_of_estimator=self.type_of_estimator)

        # Opt
        history = model.opt(
            timeout=select_models_timeout, 
            early_stoping=early_stoping,
            cold_start=cold_start,
            opt_lvl=opt_lvl,
            cv=self._cv,
            direction=self.direction,
            score_cv_folds=score_cv_folds,
            auto_parameters=auto_parameters,
            models_names=self.stack_models_names,
            opt_encoders=opt_encoders,
            cat_encoder_names=cat_encoder_names,
            target_cat_encoder_names=target_cat_encoder_names,
            verbose= (lambda x: 0 if x <= 1 else 1)(verbose), )

        history = history.drop_duplicates(subset=['model_score', 'score_std'], keep='last')
        self.stack_models_cfgs = history.head(stack_top)

        if verbose > 0:
            print("\n Predict from StackingModels")
            time.sleep(0.2) # clean print 
        # Predict
        predicts = model.predict(models_cfgs=self.stack_models_cfgs)
        self.stack_models_predicts = predicts
        
        # Score:
        if verbose > 0:
            score_mean_models = self.metric(self._data.y_train, predicts['predict_train'].mean())
            print(f'\n StackModels Mean {self.metric.__name__} Score Train: ', 
                round(score_mean_models, self._metric_round))
            time.sleep(0.1) # clean print 

        pred_test = predicts['predict_test'].mean()
        pred_train = predicts['predict_train'].mean()

        return (pred_test, pred_train)

    fit_predict = opt
    


class AutoMLClassifier(AutoML):
    type_of_estimator='classifier'
    __name__ = 'AutoMLClassifier'


class AutoMLRegressor(AutoML):
    type_of_estimator='regression'
    __name__ = 'AutoMLRegressor'
