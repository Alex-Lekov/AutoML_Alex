import sys
import numpy as np
import pandas as pd
import pytest
import sklearn
import sklearn.metrics
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import automl_alex
from automl_alex import LightGBMClassifier
RANDOM_SEED = 42


@pytest.fixture
def get_data():
    dataset = fetch_openml(name='credit-g', version=1, as_frame=True)
    dataset.target = dataset.target.astype('category').cat.codes
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, 
                                                    dataset.target,
                                                    test_size=0.25, 
                                                    random_state=RANDOM_SEED,)

    data = automl_alex.databunch.DataBunch(X_train=X_train, 
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    random_state=RANDOM_SEED)
    return(data)


def test_cv_score(get_data):
    data = get_data
    model = LightGBMClassifier(databunch=data, random_state=RANDOM_SEED)
    score, score_std = model.cross_val(print_metric=True)
    assert score is not None
    score, score_std = model.cross_val_score(print_metric=True)
    assert score is not None