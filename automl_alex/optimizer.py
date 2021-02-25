
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import optuna

############################################
# IN progress....
class Optimizer(object):
    '''
    In this library, I tried to simplify the process of optimizing and iterating parameters as much as possible. 
    You just specify how much time you are willing to spend on optimization, 
    and the library will select the rules for cross-validation, optimization depth, 
    and other optimization parameters based on this time. 
    The more time you give it, the better it will be able to select parameters.

    Automatic hyperparameter optimization based on Optuna (https://optuna.org/) but with some modifications.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit model.

    timeout : int, default=200
        Optimization time in seconds

    metric : Class, default=None 
        you can use standard metrics from sklearn.metrics or add custom metrics.
        If None, the metric is selected from the type of estimator:
        'classifier': sklearn.metrics.roc_auc_score
        'regression': sklearn.metrics.mean_squared_error

    metric_round : int, default=4

    combined_score_opt : bool, default=True
        score_opt = score - std

    iterations : int, default=None
        The number of optimization iterations. If this argument is not given, as many trials run as possible in timeout

    cold_start : int, default=30
        In the cold_start parameter, we specify how many iterations we give for this warm-up. 
        before the algorithm starts searching for optimal parameters, it must collect statistics on the represented space.
        this is why it starts in the randomsample solution at the beginning. 
        The longer it works in this mode , the less likely it is to get stuck in the local minimum. 
        But if you take too long to warm up, you may not have time to search with a more "smart" algorithm. 
        Therefore, it is important to maintain a balance.

    opt_lvl : int, default=2 
        by limiting the optimization time, we will have to choose how deep we should optimize the parameters. 
        Perhaps some parameters are not so important and can only give a fraction of a percent. 
        By setting the opt_lvl parameter, you control the depth of optimization.
        in the code automl_alex.models.model_lightgbm.LightGBM you can find how parameters are substituted for iteration
    
    early_stoping : int, default=100
        stop optimization if no better parameters are found through iterations
    
    auto_parameters : bool, default=True
        If we don't want to select anything, we just set auto_parameters=True. Then the algorithm itself will select, depending on the time allotted to it, the optimal values for:
            *folds
            *score_folds
            *cold_start
            *opt_lvl
    
    feature_selection : bool, default=True
        add feature_selection in optimization
    
    iteration_check : bool, default=True
        check: is enough time to find the optimal parameters? 
        raise optimization if Possible iters < 100.
    
    verbose : int, default=1
    
    random_state : int, default=42
        RandomState instance
        Controls the generation of the random states for each repetition.
    '''
    __name__ = 'Optimizer'


    def __init__(
        self,
        estimator,
        ):
        self.estimator=estimator


    def auto_parameters_calc(self, possible_iters, verbose=1):
        """
        Automatic determination of optimization parameters depending on the number of possible iterations

        Args:
            possible_iters (int): possible_iters
            verbose (int): print status

        Return:
            early_stoping (int)
            cv (int)
            score_cv_folds (int)
            opt_lvl (int)
            cold_start (int)
        """
        if verbose > 0: 
                print('> Start Auto calibration parameters')
        
        early_stoping = 50
        folds = 5
        score_folds = 1
        opt_lvl = 1
        cold_start = 10
            
        if possible_iters > 100:
            folds = 5
            score_folds = 2
            opt_lvl = 1
            cold_start = possible_iters // 2
            early_stoping = 100

        if possible_iters > 200:
            score_folds = 2
            opt_lvl = 2
            cold_start = (possible_iters / score_folds) // 3

        if possible_iters > 300:
            folds = 10
            score_folds = 3
            cold_start = (possible_iters / score_folds) // 5

        if possible_iters > 900:
            score_folds = 5
            opt_lvl = 3
            early_stoping = cold_start * 2
        
        if possible_iters > 10000:
            opt_lvl = 4
            score_folds = 10
            cold_start = (possible_iters / score_folds) // 10
            early_stoping = cold_start * 2

        if possible_iters > 25000:
            opt_lvl = 5
            score_folds = 20
            cold_start = (possible_iters / score_folds) // 30
            early_stoping = cold_start * 2
        return(early_stoping, folds, score_folds, opt_lvl, cold_start,)


