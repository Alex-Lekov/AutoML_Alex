import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

predict_proba_metrics = ['roc_auc_score', 'log_loss', 'brier_score_loss']


def cross_validation(
        model=None, 
        X=None,
        y=None,
        X_test=None,
        folds=10,
        score_folds=5,
        n_repeats=2,
        metric=None,
        print_metric=False, 
        metric_round=4, 
        predict=False,
        get_feature_importance=False,
        random_state=42
        ):
    """
    Cross-validation is a method for evaluating an analytical model and its behavior on independent data. 
    When evaluating the model, the available data is split into k parts. 
    Then the model is trained on k − 1 pieces of data, and the rest of the data is used for testing. 
    The procedure is repeated k times; in the end, each of the k pieces of data is used for testing. 
    The result is an assessment of the effectiveness of the selected model with the most even use of the available data.
    
    Args:
            X : array-like of shape (n_samples, n_features)
                The data to fit. Can be for example a list, or an array.
            y : array-like of shape (n_samples,) or (n_samples, n_outputs),
                The target variable to try to predict in the case of
                supervised learning.
            model : estimator object implementing 'fit'
                The object to use to fit the data.
            folds=10 :
            score_folds=5 :
            n_repeats=2 :
            metric : If None, the estimator's default scorer (if available) is used.
            print_metric=False :
            metric_round=4 (undefined):
            predict=False (undefined):
            get_feature_importance=False (undefined):
        
        Returns:
            result (dict)
    """

    if metric is None:
        if model.type_of_estimator == 'classifier':
            metric = sklearn.metrics.roc_auc_score
        elif model.type_of_estimator == 'regression':
            metric = sklearn.metrics.mean_squared_error

    if model.type_of_estimator == 'classifier':
        skf = RepeatedStratifiedKFold(
            n_splits=folds, 
            n_repeats=n_repeats,
            random_state=random_state,
            )
    else:
        skf = RepeatedKFold(
            n_splits=folds,
            n_repeats=n_repeats, 
            random_state=random_state,
            )

    folds_scores = []
    stacking_y_pred_train = np.zeros(len(X))
    stacking_y_pred_test = np.zeros(len(X_test))
    feature_importance_df = pd.DataFrame(np.zeros(len(X.columns)), index=X.columns)

    for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):

        train_x, train_y = X.iloc[train_idx], y.iloc[train_idx]
        val_x, val_y = X.iloc[valid_idx], y.iloc[valid_idx]

        # Fit
        model.fit(X_train=train_x, y_train=train_y,)

        # Predict
        if (metric.__name__ in predict_proba_metrics) and (model.is_possible_predict_proba()):
            y_pred = model.predict_proba(val_x)
            if predict:
                y_pred_test = model.predict_proba(X_test)
        else:
            y_pred = model.predict(val_x)
            if predict:
                y_pred_test = model.predict(X_test)

        score_model = metric(val_y, y_pred)
        folds_scores.append(score_model)

        if get_feature_importance:
            if i == 0:
                feature_importance_df = model.get_feature_importance(train_x)
            feature_importance_df['value'] += model.get_feature_importance(train_x)['value']

        if predict:
            stacking_y_pred_train[valid_idx] += y_pred
            stacking_y_pred_test += y_pred_test
        else:
            # score_folds
            if i+1 >= score_folds:
                break

    if predict:
        stacking_y_pred_train = stacking_y_pred_train / n_repeats
        stacking_y_pred_test = stacking_y_pred_test / (folds*n_repeats)
    
    if score_folds > 1 or predict:
        score = round(np.mean(folds_scores), metric_round)
        score_std = round(np.std(folds_scores), metric_round+2)
    else:
        score = round(score_model, metric_round)
        score_std = 0

    if print_metric:
        print(f'\n Mean Score {metric.__name__} on {i+1} Folds: {score} std: {score_std}')

    # Total
    result = {
        'score':score,
        'score_std':score_std,
        'test_predict':stacking_y_pred_test,
        'train_predict':stacking_y_pred_train,
        'feature_importance': dict(feature_importance_df['value']),
        }
    return(result)