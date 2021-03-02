'''
Models available for training
'''

from .sklearn_models import *
from .model_xgboost import XGBoost, XGBoostClassifier, XGBoostRegressor
from .model_lightgbm import LightGBM, LightGBMClassifier, LightGBMRegressor
from .model_catboost import CatBoost, CatBoostClassifier, CatBoostRegressor


all_models = {
        'LightGBM': LightGBM,
        'KNeighbors': KNeighbors,
        #'LinearSVM': LinearSVM,
        'LinearModel': LinearModel,
        'RandomForest': RandomForest,
        'ExtraTrees': ExtraTrees,
        'XGBoost': XGBoost,
        'CatBoost': CatBoost,
        'MLP': MLP,
        }