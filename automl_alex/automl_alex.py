from sklearn.metrics import *
from tqdm import tqdm
import pandas as pd
import time
from .models import *
from .databunch import DataBunch
from .encoders import *


##################################### BestSingleModel ################################################


class BestSingleModel(LightGBM):
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
        model.model_param = model.get_model_opt_params(trial=trial, 
            model=model, 
            opt_lvl=model._opt_lvl, 
            metric_name=model.metric.__name__,
            )
        return(model)

    def opt(self, 
        timeout=1000, 
        iterations=None,
        early_stoping=100, 
        cold_start=100,
        direction='maximize',
        opt_lvl=3,
        cv=None,
        score_cv_folds=None,
        auto_parameters=True,
        models_names=None, #list models_names for opt
        feature_selection=True,
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

        if models_names is None:
            self.models_names = all_models.keys()
        else:
            self.models_names = models_names
            
        # Opt
        history = self._opt_core(
            timeout, 
            early_stoping, 
            feature_selection,
            iterations=iterations,
            verbose=verbose,)
        return(history)

    def _predict_preproc_model(self, model_cfg, model,):
        """
        custom function for predict, now we can choose model library
        """
        model = self._make_model(model_cfg['model_name'], databunch=self._data)
        model.model_param = model_cfg['model_param']
        model.wrapper_params = model_cfg['wrapper_params']
        return(model)


class BestSingleModelClassifier(BestSingleModel):
    type_of_estimator='classifier'


class BestSingleModelRegressor(BestSingleModel):
    type_of_estimator='regression'


##################################### ModelsReview ################################################


class ModelsReview(BestSingleModel):
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
        auto_parameters=False,
        feature_selection=True,
        direction=None,
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
                        auto_parameters=False, # try to set the rules ourselves
                        score_cv_folds=3,
                        cold_start=50, 
                        opt_lvl=3,
                        feature_selection=feature_selection,
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

# in progress...

##################################### AutoML #########################################

class AutoML(BestSingleModel):
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
            feature_selection=True,
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
        select_models_timeout = (timeout - predict_timeout - 200)
        if select_models_timeout < 200:
            raise Exception(f"Please give me more time to optimize or reduce the number of stack models ('stack_top')")

        if stack_models_names is None:
            self.stack_models_names = all_models.keys()
        else:
            self.stack_models_names = stack_models_names

        ####################################################
        # STEP 1
        # Model 1
        if verbose > 0:
            print("\n Opt BestModels")
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
            feature_selection=feature_selection,
            verbose= (lambda x: 0 if x <= 1 else 1)(verbose), )

        history = history.drop_duplicates(subset=['model_score', 'score_std'], keep='last')
        stack_models_1_cfgs = history.head(stack_top)

        if verbose > 0:
            print("\n Predict from Models_1")
            time.sleep(0.2) # clean print 
        # Predict
        predicts_1 = model.predict(models_cfgs=stack_models_1_cfgs)
        
        # Score:
        score_mean_models_1 = self.metric(self._data.y_train, predicts_1['predict_train'].mean())
        if verbose > 0:
            print(f'\n Models_1 Mean {self.metric.__name__} Score Train: ', 
                round(score_mean_models_1, self._metric_round))
            time.sleep(0.1) # clean print 

        #############################################################
        # STEP 2
        # Model 2
        model_2 = LinearModel(databunch=self._data,
                                metric=self.metric,
                                direction=self.direction,
                                metric_round=self._metric_round,
                                combined_score_opt=self._combined_score_opt,
                                gpu=self._gpu, 
                                random_state=self._random_state,
                                type_of_estimator=self.type_of_estimator)

        # Opt
        history_2 = model_2.opt(
            iterations=50, 
            cv_folds=10,
            score_cv_folds = 5,
            opt_lvl=2,
            auto_parameters=False,
            cold_start=25,
            feature_selection=feature_selection,
            verbose= (lambda x: 0 if x <= 1 else 1)(verbose), )

        history_2 = history_2.drop_duplicates(subset=['model_score', 'score_std'], keep='last')
        stack_model_2_cfgs = history_2.head(stack_top//2)

        if verbose > 0:
            print("\n Predict from Models_2")
            time.sleep(0.2) # clean print 
        # Predict
        predicts_2 = model_2.predict(models_cfgs=stack_model_2_cfgs)
        
        # Score:
        score_mean_models_2 = self.metric(self._data.y_train, predicts_2['predict_train'].mean())
        if verbose > 0:
            print(f'\n Models_2 Mean {self.metric.__name__} Score Train: ', 
                round(score_mean_models_2, self._metric_round))
            time.sleep(0.1) # clean print 

        ###############################################################
        # STEP 3
        if self.direction is 'maximize':
            add_model_2 = (score_mean_models_1 <= (score_mean_models_2+(score_mean_models_1/10)))
        else:
            add_model_2 = (score_mean_models_1 >= (score_mean_models_2-(score_mean_models_1/10)))
        
        if add_model_2:
            self.stack_models_predicts = pd.concat([predicts_1, predicts_2], ignore_index=True, sort=False)
            self.stack_models_cfgs = pd.concat([stack_models_1_cfgs, stack_model_2_cfgs], ignore_index=True, sort=False)
        else:
            self.stack_models_predicts = predicts_1
            self.stack_models_cfgs = stack_models_1_cfgs
        
            
        pred_test = self.stack_models_predicts['predict_test'].mean()
        pred_train = self.stack_models_predicts['predict_train'].mean()

        if verbose > 0:
            score_mean_models = self.metric(self._data.y_train, pred_train)
            print(f'\n StackModels {self.metric.__name__} Score Train: ', \
                round(score_mean_models, self._metric_round))
            time.sleep(0.1) # clean print 

        return (pred_test, pred_train)

    fit_predict = opt
    


class AutoMLClassifier(AutoML):
    type_of_estimator='classifier'
    __name__ = 'AutoMLClassifier'


class AutoMLRegressor(AutoML):
    type_of_estimator='regression'
    __name__ = 'AutoMLRegressor'
