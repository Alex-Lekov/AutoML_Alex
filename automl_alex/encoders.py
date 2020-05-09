
import pandas as pd
import numpy as np
import time
import warnings
import os
from category_encoders import HashingEncoder, SumEncoder, OneHotEncoder, HelmertEncoder, OrdinalEncoder
from category_encoders import TargetEncoder, CatBoostEncoder, JamesSteinEncoder


################################################################
            #               Simple Encoders 
            #      (do not use information about target)
################################################################
class FrequencyEncoder():
    """
    FrequencyEncoder  
    Conversion of category into frequencies.
    Parameters
        ----------
    cols : list of categorical features.
    drop_invariant : not used
    """    
    def __init__(self, cols=None, drop_invariant=None):
        self.cols = cols
        self.counts_dict = None

    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        counts_dict = {}
        if self.cols is None:
            self.cols = X.columns
        for col in self.cols:
            values = X[col].value_counts(dropna=False).index
            counts = list(X[col].value_counts(dropna=False))
            counts_dict[col] = dict(zip(values, counts))
        self.counts_dict = counts_dict

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        counts_dict_test = {}
        res = []
        for col in self.cols:
            values = X[col].value_counts(dropna=False).index
            counts = list(X[col].value_counts(dropna=False))
            counts_dict_test[col] = dict(zip(values, counts))

            # if value is in "train" keys - replace "test" counts with "train" counts
            for k in [
                key
                for key in counts_dict_test[col].keys()
                if key in self.counts_dict[col].keys()
            ]:
                counts_dict_test[col][k] = self.counts_dict[col][k]

            res.append(X[col].map(counts_dict_test[col]).values.reshape(-1, 1))
        res = np.hstack(res)

        X[self.cols] = res
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        X = self.transform(X)
        return X



################################################################
            #                Target Encoders
################################################################

# in progress...


encoders_names = {
                'HashingEncoder': HashingEncoder,
                'SumEncoder': SumEncoder,
                'OneHotEncoder': OneHotEncoder,
                'HelmertEncoder': HelmertEncoder,
                'OrdinalEncoder': OrdinalEncoder,
                #'FrequencyEncoder': FrequencyEncoder,
                }


target_encoders_names = {
                        'TargetEncoder': TargetEncoder,
                        'CatBoostEncoder': CatBoostEncoder,
                        'JamesSteinEncoder': JamesSteinEncoder,
                        }