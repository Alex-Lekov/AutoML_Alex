import numpy as np
import pandas as pd
import time
import datetime
import pytest
import sklearn

from sklearn.datasets import fetch_openml

import automl_alex
from automl_alex._logger import *
from automl_alex.models import *
from automl_alex.data_prepare import *
from automl_alex import AutoMLClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm
import sys

logger.info(f"automl_alex v: {automl_alex.__version__}")


RANDOM_SEED = 42
TIME_LIMIT = 3600  # 1h
CV = 5


def test_automl_classifier_bench():
    for data_id in [
        # 179,
        # 4135,
        1461,
        # 1226,
        # 31,
        1471,
        151,
        # 1067,
        # 1046,
        1489,
        1494,
    ]:
        dataset = fetch_openml(data_id=data_id, as_frame=True)
        dataset.target = dataset.target.astype("category").cat.codes

        logger.info("=" * 75)
        logger.info("LOAD DATASET")
        logger.info(f"Dataset: {data_id} {dataset.data.shape}")

        y = dataset.target
        X = dataset.data

        skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=42)

        metrics = []

        for count, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            # if count > 3:
            #    continue
            logger.info(f"START FOLD {count}")
            RANDOM_SEED = count
            EXPERIMENT = count
            np.random.seed(RANDOM_SEED)

            # shuffle columns for more randomization experiment
            columns_tmp = list(X.columns.values)
            np.random.shuffle(columns_tmp)
            X = X[columns_tmp]

            # Split
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            START_EXPERIMENT = time.time()

            model = AutoMLClassifier(
                random_state=RANDOM_SEED,
            )
            model.fit(X_train, y_train, timeout=TIME_LIMIT)

            predicts = model.predict(X_test)
            assert predicts is not None

            # model.save(f'AutoML_fold_{count}', folder='./result/')

            logger.info("*" * 75)
            logger.info(f"AUC: {round(roc_auc_score(y_test, predicts),4)}")

            logger.info(
                f"predict_model_1 AUC: {round(sklearn.metrics.roc_auc_score(y_test, model.predict_model_1),4)}"
            )
            logger.info(
                f"predict_model_2 AUC: {round(sklearn.metrics.roc_auc_score(y_test, model.predict_model_2),4)}"
            )
            # logger.info(f'predict_model_3 AUC: {round(sklearn.metrics.roc_auc_score(y_test, model.predict_model_3),4)}')
            # logger.info(f'predict_model_4 AUC: {round(sklearn.metrics.roc_auc_score(y_test, model.predict_model_4),4)}')
            # logger.info(f'predict_model_5 AUC: {round(sklearn.metrics.roc_auc_score(y_test, model.predict_model_5),4)}')
            logger.info("-" * 75)

            END_EXPERIMENT = time.time()

            metrics.append(
                {
                    "AUC": round(roc_auc_score(y_test, predicts), 4),
                    "log_loss": round(log_loss(y_test, predicts), 4),
                    "Accuracy": round(accuracy_score(y_test, predicts > 0.5), 4),
                    "Time_min": (END_EXPERIMENT - START_EXPERIMENT) // 60,
                    "Time": datetime.datetime.now(),
                }
            )

            pd.DataFrame(metrics).to_csv(
                f"./result/{data_id}_metrics.csv",
                index=False,
            )
            model = None