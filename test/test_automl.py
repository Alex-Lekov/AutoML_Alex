import sys
import numpy as np
import pandas as pd
import pytest
import sklearn
import sklearn.metrics
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import automl_alex
from automl_alex import ModelsReviewClassifier

RANDOM_SEED = 42


@pytest.fixture
def get_data():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(data.data), 
                                                    pd.DataFrame(data.target), 
                                                    test_size=0.10, 
                                                    random_state=42,)
    return(X_train, X_test, y_train, y_test)


def test_ModelsReviewClassifier(get_data):
    test_model = ModelsReviewClassifier(X_train, X_test, y_train, y_test,
                                        random_state=RANDOM_SEED)
    review = test_model.fit(verbose=1,)
    assert review is not None
    assert isinstance(review, pd.DataFrame)
    assert not review.empty
    predict = test_model.predict()
    score = roc_auc_score(data.y_test, predict[0]['predict_test'])
    assert score is not None
    assert score >= 0.98

#     # CV=5, timeout = 300sec
#     test_model = ModelsReviewClassifier(databunch=data,
#                                         cv=5,random_state=RANDOM_SEED)
#     review = test_model.opt(timeout=300,)
#     assert review is not None
#     assert isinstance(review, pd.DataFrame)
#     assert not test_model.trials_dataframe.empty
#     predict = test_model.predict()
#     score = roc_auc_score(data.y_test, predict[0]['predict_test'])
#     assert score is not None
#     assert score >= 0.98


# def test_BestSingleModelClassifier(get_data):
#     data = get_data
#     test_model = BestSingleModelClassifier(databunch=data, cv=0, random_state=RANDOM_SEED)
#     s = test_model.opt(timeout=200)
#     predict_test = test_model.predict(get_pred_train=False)
#     assert predict_test is not None
#     score = roc_auc_score(data.y_test, predict_test)
#     assert score is not None
#     assert score >= 0.98

#     # CV=5, timeout = 300sec
#     test_model = BestSingleModelClassifier(databunch=data, cv=5, random_state=RANDOM_SEED)
#     s = test_model.opt(timeout=500,)
#     predict_test = test_model.predict(get_pred_train=False)
#     assert predict_test is not None
#     score = roc_auc_score(data.y_test, predict_test)
#     assert score is not None
#     assert score >= 0.98


# def test_StackingClassifier(get_data):
#     data = get_data
#     test_model = StackingClassifier(databunch=data, 
#                                     cv=5,
#                                     random_state=RANDOM_SEED)
#     s = test_model.opt(timeout=300, verbose=0,)
#     predict_test = test_model.predict(get_pred_train=False,)
#     assert predict_test is not None
#     score = roc_auc_score(data.y_test, predict_test)
#     assert score is not None
#     assert score >= 0.99
