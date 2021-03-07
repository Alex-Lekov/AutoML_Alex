import pandas as pd
import pytest
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

import automl_alex
from automl_alex.models import *
from automl_alex.data_prepare import *
from automl_alex.optimizer import *

#
RANDOM_SEED = 42


# ### Test #####################################################
# Binary-Classification

def test_bestsinglemodelclassifier_default_classification():
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

        optimizer = automl_alex.BestSingleModelClassifier(models_names = ['LinearModel','KNeighbors','RandomForest','LightGBM','ExtraTrees','MLP'])
        history = optimizer.opt(X_train, y_train, timeout=400, verbose=3)
        predicts = optimizer.predict(X_test)

        score = round(sklearn.metrics.roc_auc_score(y_test, predicts),4)
        assert score is not None
        assert 0.5 < score <= 1


def test_optimizer_default_classification():
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
        de = DataPrepare(normalization=True,verbose=0)
        X_train = de.fit_transform(X_train)
        X_test = de.transform(X_test)
        for model_name in all_models.keys():
            print(model_name)
            model = all_models[model_name](type_of_estimator='classifier',
                                                random_state=RANDOM_SEED)
            #model = model.fit(X_train, y_train)

            history = model.opt(X_train, y_train, timeout=100, verbose=3)
            predicts = model.predict(X_test)

            score = round(sklearn.metrics.roc_auc_score(y_test, predicts),4)
            print(score)
            assert score is not None
            assert 0.49 < score <= 1