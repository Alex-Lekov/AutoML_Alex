import pandas as pd
import pytest
import sklearn
import sklearn.metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import automl_alex
from automl_alex import BestSingleModelClassifier, ModelsReviewClassifier, AutoMLClassifier

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


def test_ModelsReviewClassifier(get_data):
    data = get_data
    test_model = ModelsReviewClassifier(databunch=data, random_state=RANDOM_SEED)
    review = test_model.fit(verbose=1,)
    assert review is not None
    assert isinstance(review, pd.DataFrame)
    assert not review.empty
    predicts = test_model.predict()
    score = sklearn.metrics.roc_auc_score(data.y_test, predicts['predict_test'][0])
    assert score is not None
    assert score >= 0.75

    # timeout = 300sec
    test_model = ModelsReviewClassifier(databunch=data, random_state=RANDOM_SEED)
    review = test_model.opt(timeout=300, models_names=['LinearModel', 'LightGBM',])
    assert review is not None
    assert isinstance(review, pd.DataFrame)
    assert not test_model.top1_models_cfgs.empty
    assert not test_model.top10_models_cfgs.empty
    assert not test_model.history_trials_dataframe.empty
    predicts = test_model.predict(models_cfgs=test_model.top1_models_cfgs)
    score = sklearn.metrics.roc_auc_score(data.y_test, predicts['predict_test'][0])
    assert score is not None
    assert score >= 0.75


def test_BestSingleModelClassifier(get_data):
    data = get_data
    test_model = BestSingleModelClassifier(databunch=data, random_state=RANDOM_SEED)
    history = test_model.opt(timeout=500)
    assert history is not None
    assert isinstance(history, pd.DataFrame)
    predicts = test_model.predict()
    assert predicts is not None
    score = sklearn.metrics.roc_auc_score(data.y_test, predicts['predict_test'][0])
    assert score is not None
    assert score >= 0.78

    test_model = BestSingleModelClassifier(databunch=data, cv=10, score_cv_folds=3, random_state=RANDOM_SEED)
    history = test_model.opt(timeout=500, auto_parameters=False,)
    assert history is not None
    assert isinstance(history, pd.DataFrame)
    predicts = test_model.predict()
    assert predicts is not None
    score = sklearn.metrics.roc_auc_score(data.y_test, predicts['predict_test'][0])
    assert score is not None
    assert score >= 0.78


def test_AutoMLClassifier(get_data):
    data = get_data
    test_model = AutoMLClassifier(databunch=data, random_state=RANDOM_SEED)
    predict_test, predict_train = test_model.opt(timeout=1500, verbose=0,)
    assert predict_test is not None
    score = sklearn.metrics.roc_auc_score(data.y_test, predict_test)
    assert score is not None
    assert score >= 0.8
