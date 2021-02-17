import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler

from .encoders import *

# disable chained assignments
pd.options.mode.chained_assignment = None 

class CleanNans(object):
    """
    Сlass for cleaning Nans
    """

    def __init__(self, method=0):
        """
        Fill Nans and add column, that there were nans in this column
        
        Args:
            method : {0, 'median', 'mean',}
        """
        self.method = method

    def fit(self, data, cols=None):
        """
        Fit fillna.

        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
            cols list() features: the input data
        Returns:
            self
        """
        if cols is not None:
            data = data[cols]
        
        self.nan_columns = list(data.columns[data.isnull().sum() > 0])
        if not self.nan_columns:     
            print('No nans features')

        if self.method is 'median':
            self.fill_value = data.median()
        elif self.method is 'mean':
            self.fill_value = data.mean()
        else:
            self.fill_value = 0
        return self

    def transform(self, data, cols=None):
        """
        Transform fillna.

        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
            cols list() features: the input data
        Returns:
            Clean data (pd.DataFrame, shape (n_samples, n_features))
        """
        if cols is not None:
            data = data[cols]

        if self.nan_columns:
            for nan_column in nan_columns:
                data[nan_column+'isNAN'] = pd.isna(data[nan_column]).astype('uint8')
            
            data.fillna(self.fill_value, inplace=True)
        else:
            print('No nans features')
        return data



    def clean_nans(self, data, cols=None):
        """
        Fill Nans and add column, that there were nans in this column
        
        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
            cols list() features: the input data
        Return:
            Clean data (pd.DataFrame, shape (n_samples, n_features))
        
        """
        if cols is not None:
            nan_columns = list(data[cols].columns[data[cols].isnull().sum() > 0])
            if nan_columns:
                for nan_column in nan_columns:
                    data[nan_column+'isNAN'] = pd.isna(data[nan_column]).astype('uint8')
                data.fillna(data.median(), inplace=True)
        return(data)

class DataPrepare(object):
    """
    Сlass for cleaning, encoding and processing your dataset
    """
    def __init__(self, 
                cat_features=None,
                clean_and_encod_data=True,
                cat_encoder_names=['HelmertEncoder',],
                clean_nan=True,
                num_generator_features=True,
                group_generator_features=False,
                frequency_enc_num_features=False,
                normalization=True,
                random_state=42,
                verbose=1):
        """
        Description of __init__

        Args:
            cat_features=None (list or None): 
            clean_and_encod_data=True (undefined):
            cat_encoder_names=None (list or None):
            clean_nan=True (undefined):
            num_generator_features=True (undefined):
            group_generator_features=True (undefined):
            random_state=42 (undefined):
        """
        self.random_state = random_state
        self.cat_encoder_names = cat_encoder_names


    def check_data_format(self, data):
        """
        Description of check_data_format:
            Check that data is not pd.DataFrame or empty

        Args:
            data (undefined): dataset
        Return:
            True or Exception
        """
        data_tmp = pd.DataFrame(data)
        if data_tmp is None or data_tmp.empty:
            raise Exception("data is not pd.DataFrame or empty")
        return(True)

    def clean_nans(self, data, cols=None):
        """
        Fill Nans and add column, that there were nans in this column
        
        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
            cols list() features: the input data
        Return:
            Clean data (pd.DataFrame, shape (n_samples, n_features))
        
        """
        if cols is not None:
            nan_columns = list(data[cols].columns[data[cols].isnull().sum() > 0])
            if nan_columns:
                for nan_column in nan_columns:
                    data[nan_column+'isNAN'] = pd.isna(data[nan_column]).astype('uint8')
                data.fillna(data.median(), inplace=True)
        return(data)

    def auto_detect_cat_features(self, data):
        """
        Description of _auto_detect_cat_features:
            Auto-detection categorical_features by simple rule:
            categorical feature == if feature nunique low 1% of data

        Args:
            data (pd.DataFrame): dataset
            
        Returns:
            cat_features (list): columns names cat features
        
        """
        #object_features = list(data.columns[data.dtypes == 'object'])
        cat_features = data.columns[(data.nunique(dropna=False) < len(data)//100) & \
            (data.nunique(dropna=False) >2)]
        #cat_features = list(set([*object_features, *cat_features]))
        return(cat_features)

    def encodet_features(self, data, cat_encoder_name) -> pd.DataFrame:
        """
        Description of _encode_features:
            Encode car features

        Args:
            data (pd.DataFrame):
            cat_encoder_name (str): cat Encoder name

        Returns:
            pd.DataFrame

        """
        if cat_encoder_name in cat_encoders_names.keys():
            encoder = cat_encoders_names[cat_encoder_name](drop_invariant=True) 

            if cat_encoder_name == 'HashingEncoder':
                encoder = cat_encoders_names[cat_encoder_name](n_components=int(np.log(len(data.columns))*1000), 
                                                        drop_invariant=True)

            data_encodet = encoder.fit_transform(data)
            data_encodet = data_encodet.add_prefix(cat_encoder_name + '_')
        else:
            raise Exception(f"{cat_encoder_name} not support!")
        return(data_encodet)