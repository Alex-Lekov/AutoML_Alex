import pandas as pd
import pytest
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from automl_alex.models import *
from automl_alex.data_prepare import *
from automl_alex import ModelsReviewClassifier, AutoMLClassifier

#
RANDOM_SEED = 42
TMP_FOLDER = '.automl-alex_tmp/'

@pytest.fixture
def get_data():
    dataset = fetch_openml(name='adult', version=1, as_frame=True)
    # convert target to binary
    dataset.target = dataset.target.astype('category').cat.codes
    dataset.data.head(5)
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, 
                                                        dataset.target,
                                                        test_size=0.2, 
                                                        random_state=RANDOM_SEED,)
    de = DataPrepare()
    X_train = de.fit_transform(X_train)
    X_test = de.transform(X_test)
    return(X_train, X_test, y_train, y_test)


# ### Test ModelsReviewClassifier #####################################################
# Binary-Classification
def test_ModelsReviewClassifier(get_data):
    X_train, X_test, y_train, y_test = get_data

    model = ModelsReviewClassifier(
        metric = sklearn.metrics.roc_auc_score,
        random_state=RANDOM_SEED)

    # let's see what the results are for all available models with default settings
    review = model.fit(
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test,
        )
    #assert isinstance(review, pd.DataFrame)
    assert 1 < len(review)


def test_automl_default_classification():
    for data_id in [179,4135,]:
        dataset = fetch_openml(data_id=data_id, as_frame=True)
        dataset.target = dataset.target.astype('category').cat.codes
        if len(dataset.data) < 2000:
            crop = len(dataset.data)
        else:
            crop = 2000
        X_train, X_test, y_train, y_test = train_test_split(dataset.data[:crop], 
                                                            dataset.target[:crop],
                                                            test_size=0.2, 
                                                            random_state=RANDOM_SEED,)
        model = AutoMLClassifier(random_state=RANDOM_SEED,)
        model.fit(X_train, y_train, timeout=600)
        predicts = model.predict(X_test)

        score = round(sklearn.metrics.roc_auc_score(y_test, predicts),4)
        assert score is not None
        assert 0.5 < score <= 1

        model.save('AutoML_model_1', folder=TMP_FOLDER)
        model_new = AutoMLClassifier(random_state=RANDOM_SEED,)
        model_new = model_new.load('AutoML_model_1', folder=TMP_FOLDER)
        predicts = model_new.predict(X_test)
        score2 = round(sklearn.metrics.roc_auc_score(y_test, predicts),4)
        assert score2 is not None
        assert 0.5 < score2 <= 1
        assert (score-score2) == 0.
