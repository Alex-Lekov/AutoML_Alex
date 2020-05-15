from .sklearn_models import *
from .model_xgboost import XGBoost, XGBoostClassifier, XGBoostRegressor
from .model_lightgbm import LightGBM, LightGBMClassifier, LightGBMRegressor
from .model_catboost import CatBoost, CatBoostClassifier, CatBoostRegressor
from .base import *

all_models = {
        'LightGBM': LightGBM,
        'KNeighbors': KNeighbors,
        'LinearSVM': LinearSVM,
        'LinearModel': LinearModel,
        'SGD': SGD,
        'RandomForest': RandomForest,
        'ExtraTrees': ExtraTrees,
        'XGBoost': XGBoost,
        'CatBoost': CatBoost,
        'MLP': MLP,
        }