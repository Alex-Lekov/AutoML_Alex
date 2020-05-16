import sys
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import automl_alex
from automl_alex.models import *

#
RANDOM_SEED = 42

@pytest.fixture
def get_data():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(data.data), 
                                                    pd.DataFrame(data.target), 
                                                    test_size=0.10, 
                                                    random_state=42,)
    data = DataBunch(X_train=X_train, 
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    random_state=RANDOM_SEED)
    return(data)

# ### Test Models #####################################################

def test_fit_predict_default_model(get_data):
    data = get_data
    for model_name in all_models.keys():
        test_model = all_models[model_name](databunch=data,
                                            metric=sklearn.metrics.roc_auc_score, 
                                            direction = 'maximize',
                                            type_of_estimator='classifier',
                                            random_state=RANDOM_SEED)
        config = test_model.fit()
        assert config is not None
        assert isinstance(config, pd.DataFrame)
        assert not config.empty

        predicts = test_model.predict()
        score = sklearn.metrics.roc_auc_score(data.y_test, predicts['predict_test'][0])
        assert score is not None
        assert 0.9 < score <= 1

        predicts = test_model.predict(cv=0)
        score = sklearn.metrics.roc_auc_score(data.y_test, predicts['predict_test'][0])
        assert score is not None
        assert 0.9 < score <= 1

def test_cross_val_score(get_data):
    data = get_data
    for model_name in all_models.keys():
        test_model = all_models[model_name](databunch=data,
                                            metric=sklearn.metrics.roc_auc_score, 
                                            direction = 'maximize',
                                            type_of_estimator='classifier',
                                            random_state=RANDOM_SEED)
        predict_test, predict_train = test_model.cv(predict=True, n_repeats=3)
        score = sklearn.metrics.roc_auc_score(data.y_test, predict_test)
        assert score is not None
        assert 0.9 < score <= 1

        test_model = all_models[model_name](databunch=data,
                                            metric=sklearn.metrics.roc_auc_score, 
                                            direction = 'maximize',
                                            cv=12,
                                            type_of_estimator='classifier',
                                            random_state=RANDOM_SEED)
        predict_test, predict_train = test_model.cv(predict=True, n_repeats=1)
        score = sklearn.metrics.roc_auc_score(data.y_test, predict_test)
        assert score is not None
        assert 0.9 < score <= 1


# def test_opt_default(get_data):
#     data = get_data
#     for model_name in all_models.keys():
#         test_model = all_models[model_name](databunch=data, random_state=RANDOM_SEED)
#         s = test_model.opt(timeout=100, verbose=0,)
#         assert s is not None
#         assert test_model.best_score > 0.95

