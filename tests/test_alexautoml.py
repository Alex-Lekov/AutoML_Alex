import pandas as pd
import pytest
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

from automl_alex.models import *
from automl_alex.data_prepare import *
from automl_alex import ModelsReviewClassifier

#
RANDOM_SEED = 42

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