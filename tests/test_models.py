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
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, 
                                                            dataset.target,
                                                            test_size=0.2, 
                                                            random_state=RANDOM_SEED,)
        de = DataPrepare(normalization=True, verbose=0)
        X_train = de.fit_transform(X_train)
        X_test = de.transform(X_test)
        for model_name in all_models.keys():
            print(model_name)
            model = all_models[model_name](type_of_estimator='classifier',
                                                random_state=RANDOM_SEED)
            model = model.fit(X_train, y_train)
            if model.is_possible_predict_proba():
                predicts = model.predict_proba(X_test)
            else:
                predicts = model.predict(X_test)
            assert predicts is not None

            score = sklearn.metrics.roc_auc_score(y_test, predicts)
            print(model_name, score)
            assert score is not None
            assert 0.5 < score <= 1

def test_cross_val_score_classification(get_data):
    for data_id in [179,1461,31,1471,151,1067,1046,1489,1494]:
        dataset = fetch_openml(data_id=data_id, as_frame=True)
        dataset.target = dataset.target.astype('category').cat.codes
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, 
                                                            dataset.target,
                                                            test_size=0.2, 
                                                            random_state=RANDOM_SEED,)
        de = DataPrepare(normalization=True,verbose=0)
        X_train = de.fit_transform(X_train)
        X_test = de.transform(X_test)
        for model_name in all_models.keys():
            print(model_name)
            model = all_models[model_name](type_of_estimator='classifier',
                                                random_state=RANDOM_SEED)
            #model = model.fit(X_train, y_train)

            result = model.cross_validation(
                X=X_train,
                y=y_train,
                X_test=X_test,
                folds=10, 
                score_folds=3,
                n_repeats=2,
                metric=sklearn.metrics.roc_auc_score,
                print_metric=True, 
                metric_round=4, 
                predict=False,
                get_feature_importance=False,
                )

            assert result is not None
            print(model_name, result['score'])
            assert result['score'] is not None
            assert 0.5 < result['score'] <= 1


def test_cross_val_predict_classification(get_data):
    for data_id in [179,1461,31,1471,151,1067,1046,1489,1494]:
        dataset = fetch_openml(data_id=data_id, as_frame=True)
        dataset.target = dataset.target.astype('category').cat.codes
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, 
                                                            dataset.target,
                                                            test_size=0.2, 
                                                            random_state=RANDOM_SEED,)
        de = DataPrepare(normalization=True,verbose=0)
        X_train = de.fit_transform(X_train)
        X_test = de.transform(X_test)
        for model_name in all_models.keys():
            print(model_name)
            model = all_models[model_name](type_of_estimator='classifier',
                                                random_state=RANDOM_SEED)
            #model = model.fit(X_train, y_train)

            result = model.cross_validation(
                X=X_train,
                y=y_train,
                X_test=X_test,
                folds=10, 
                score_folds=3,
                n_repeats=2,
                metric=sklearn.metrics.roc_auc_score,
                print_metric=True, 
                metric_round=4, 
                predict=True,
                get_feature_importance=True,
                )

            assert result is not None

            score = sklearn.metrics.roc_auc_score(y_test, result['test_predict'])
            print(model_name, score)
            assert score is not None
            assert 0.5 < score <= 1