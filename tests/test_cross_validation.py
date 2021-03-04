import pandas as pd
import pytest
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

import automl_alex
from automl_alex.models import *
from automl_alex.data_prepare import *
from automl_alex import CrossValidation

#
RANDOM_SEED = 42

# ### Test Cross Validation #####################################################
# Binary-Classification

def test_cross_val_score_classification():
    for data_id in [179,]:
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

            cv = CrossValidation(
                    estimator=model,
                    folds=10,
                    score_folds=3,
                    n_repeats=1,
                    metric=sklearn.metrics.roc_auc_score,
                    print_metric=False, 
                    metric_round=4, 
                    random_state=RANDOM_SEED
                    )
            # Score
            score, score_std = cv.fit_score(X_train, y_train, print_metric=True)
            print(model_name, score, score_std)
            assert score is not None
            assert 0.5 < score <= 1

            # Fit
            cv.fit(X_train, y_train)

            # Test
            predicts = cv.predict_test(X_test)
            assert predicts is not None

            score_cv1 = round(sklearn.metrics.roc_auc_score(y_test, predicts),4)
            assert score_cv1 is not None
            assert 0.5 < score_cv1 <= 1

            # Train
            predicts = cv.predict_train(X_train)
            assert predicts is not None

            score = round(sklearn.metrics.roc_auc_score(y_train, predicts),4)
            assert score is not None
            assert 0.5 < score <= 1

            if cv.estimator._is_possible_feature_importance():
                feature_importance = cv.get_feature_importance(X_train)
                assert isinstance(feature_importance, pd.DataFrame)
                assert not feature_importance.empty

            # SAVE LOAD
            cv.save('test_save')

            cv_2 = CrossValidation(estimator=model,)
            cv_2 = cv_2.load('test_save')
            predicts = cv_2.predict_test(X_test)
            assert predicts is not None

            score_cv2 = round(sklearn.metrics.roc_auc_score(y_test, predicts),4)
            assert score_cv2 is not None
            assert 0.5 < score_cv2 <= 1
            #assert score_cv1 != score_cv2