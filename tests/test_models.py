import pandas as pd
import pytest
import sklearn
from sklearn.model_selection import train_test_split
from automl_alex.models import *
from sklearn.datasets import fetch_openml
from automl_alex.data_prepare import *

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

# ### Test Models #####################################################
# Binary-Classification
def test_fit_predict_default_classification():
    for data_id in [179,1461,31,1471,151,1067,1046,1489,1494]:
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
        de = DataPrepare(normalization=True, verbose=0)
        X_train = de.fit_transform(X_train)
        X_test = de.transform(X_test)
        for model_name in all_models.keys():
            print(model_name)
            model = all_models[model_name](type_of_estimator='classifier',
                                                random_state=RANDOM_SEED)
            model.fit(X_train, y_train)
            if model.is_possible_predict_proba():
                predicts = model.predict_proba(X_test)
            else:
                predicts = model.predict(X_test)
            assert predicts is not None

            score = sklearn.metrics.roc_auc_score(y_test, predicts)
            print(model_name, score)
            assert score is not None
            assert 0.49 < score <= 1

            if model._is_possible_feature_importance():
                feature_importance = model.get_feature_importance(X_train)
                assert isinstance(feature_importance, pd.DataFrame)
                assert not feature_importance.empty
