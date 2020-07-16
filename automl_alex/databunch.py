import pandas as pd
import numpy as np
import time
import warnings
import os
from itertools import combinations

from .encoders import *


class DataBunch(object):
    """
    Сlass for storing, cleaning and processing your dataset
    """
    def __init__(self, 
                    X_train=None, 
                    y_train=None,
                    X_test=None,
                    y_test=None,
                    cat_features=None,
                    clean_and_encod_data=True,
                    cat_encoder_names=['FrequencyEncoder', 'HelmertEncoder', 'HashingEncoder'],
                    clean_nan=True,
                    num_generator_features=True,
                    group_generator_features=True,
                    random_state=42):
        self.random_state = random_state
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_train_predicts = None
        self.X_test_predicts = None
        self.cat_features = None

        # Encoders
        if cat_encoder_names is None:
            self.cat_encoder_names = cat_encoders_names.keys()
        else:
            self.cat_encoder_names = cat_encoder_names
        
        # check X_train, y_train, X_test
        if self.check_data_format(X_train):
            self.X_train_source = pd.DataFrame(X_train)
        if X_test is not None:
            if self.check_data_format(X_test):
                self.X_test_source = pd.DataFrame(X_test)
                
        if y_train is None:
            raise Exception("No target data!")
        else: self.y_train = y_train
        
        if y_test is not None:
            self.y_test = y_test
        
        # add categorical features in DataBunch
        if cat_features is None:
            self.cat_features = self._auto_detect_cat_features(self.X_train_source)
        else:
            self.cat_features = list(cat_features)
        
        # preproc_data in DataBunch 
        if clean_and_encod_data:
            self.X_train, self.X_test = self.preproc_data(self.X_train_source, 
                                                            self.X_test_source, 
                                                            cat_features=self.cat_features,
                                                            cat_encoder_names=cat_encoder_names,
                                                            clean_nan=clean_nan,
                                                            num_generator_features=num_generator_features,
                                                            group_generator_features=group_generator_features,)
        else: 
            self.X_train, self.X_test = X_train, X_test
                                        
    
    def check_data_format(self, data):
        data_tmp = pd.DataFrame(data)
        if data_tmp is None or data_tmp.empty:
            raise Exception("data is not pd.DataFrame or empty")
        return(True)


    def clean_nans(self, data, cols=None):
        '''
        Fill Nans and add column, that there were nans in this column
        
        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
            cols list() features: the input data
        Return:
            Clean data (pd.DataFrame, shape (n_samples, n_features))
        '''
        if cols is not None:
            nan_columns = list(data[cols].columns[data[cols].isnull().sum() > 0])
            if nan_columns:
                for nan_column in nan_columns:
                    data[nan_column+'isNAN'] = pd.isna(data[nan_column]).astype('uint8')
                data.fillna(data.median(), inplace=True)
        return(data)


    def _auto_detect_cat_features(self, data):
        '''
        Auto-detection categorical_features by simple rule:
        categorical feature == if feature nunique low 1% of data
        '''
        #object_features = list(data.columns[data.dtypes == 'object'])
        cat_features = data.columns[(data.nunique(dropna=False) < len(data)//100) & \
            (data.nunique(dropna=False) >2)]
        #cat_features = list(set([*object_features, *cat_features]))
        return(cat_features)


    def _encode_features(self, data, cat_encoder_name):
        '''
        Encode car features
        '''
        if cat_encoder_name in cat_encoders_names.keys():
            encoder = cat_encoders_names[cat_encoder_name](drop_invariant=True) 

            if cat_encoder_name == 'HashingEncoder':
                encoder = cat_encoders_names[cat_encoder_name](n_components=int(np.log(len(data))*100), 
                                                        drop_invariant=True)
            if cat_encoder_name == 'FrequencyEncoder':
                encoder = cat_encoders_names['OrdinalEncoder']()
                data = encoder.fit_transform(data)
                encoder = cat_encoders_names['FrequencyEncoder']()

            data_encodet = encoder.fit_transform(data)
            data_encodet = data_encodet.add_prefix(cat_encoder_name + '_')
        else:
            raise Exception(f"{cat_encoder_name} not support!")
        return(data_encodet)


    def numeric_interaction_terms(self, df, columns):
        '''
        Numerical interaction generator features: A/B, A*B, A-B,
        '''
        fe_df = pd.DataFrame()
        for c in combinations(columns,2):
            fe_df['{}_/_{}'.format(c[0], c[1]) ] = (df[c[0]]*1.) / df[c[1]]
            fe_df['{}_*_{}'.format(c[0], c[1]) ] = df[c[0]] * df[c[1]]
            fe_df['{}_-_{}'.format(c[0], c[1]) ] = df[c[0]] - df[c[1]]
            #fe_df['{} + {}'.format(c[0], c[1]) ] = df[c[0]] + df[c[1]]
        return fe_df


    def group_encoder(self, data, cat_columns, num_columns):
        '''
        Group Encode car features on num features
        '''
        for num_col in num_columns:
            encoder = JamesSteinEncoder(drop_invariant=True)
            data_encodet = encoder.fit_transform(X=data[cat_columns], y=data[num_col].values)
            data_encodet = data_encodet.add_prefix('GroupEncoder_' + num_col + '_')
            data = pd.concat([
                        data.reset_index(drop=True), 
                        data_encodet.reset_index(drop=True)], 
                        axis=1,)
        return(data)


    def preproc_data(self, X_train=None, 
                        X_test=None, 
                        cat_features=None,
                        cat_encoder_names=None,
                        clean_nan=True,
                        num_generator_features=True,
                        group_generator_features=True):
        '''
        dataset preprocessing function
        '''
        # concat datasets for correct processing.
        df_train = X_train.copy()
        df_train['test'] = 0
        
        if X_test is not None:
            df_test = X_test.copy()
            df_test['test'] = 1
            data = df_train.append(df_test, sort=False).reset_index(drop=True) # concat
        else: data = df_train
        
        # object & num features
        object_features = list(data.columns[(data.dtypes == 'object') | (data.dtypes == 'category')])
        num_features = list(set(data.columns) - set(cat_features) - set(object_features) - {'test'})
        encodet_features_names = list(set(object_features + list(cat_features)))
        self.encodet_features_names = encodet_features_names

        # LabelEncoded Binary Features
        for feature in data.columns:
            if (feature is not 'test') and (data[feature].nunique(dropna=False) < 3):
                data[feature] = data[feature].astype('category').cat.codes
                #if len(encodet_features_names) > 0:
                #    if feature in encodet_features_names:
                #        encodet_features_names.remove(feature)
        
        # Encoding
        if encodet_features_names:
            for encoder_name in cat_encoder_names:
                data_encodet = self._encode_features(data[encodet_features_names], encoder_name,)
                data = pd.concat([
                    data.reset_index(drop=True), 
                    data_encodet.reset_index(drop=True)], 
                    axis=1,)

        # FrequencyEncoder num features
        if 'FrequencyEncoder' in cat_encoder_names:
            encoder = cat_encoders_names['FrequencyEncoder']()
            data_encodet = encoder.fit_transform(data[num_features])
            data_encodet = data_encodet.add_prefix('FrequencyEncoder' + '_')
            data = pd.concat([
                    data.reset_index(drop=True), 
                    data_encodet.reset_index(drop=True)], 
                    axis=1,)

        # Nans
        if clean_nan:
            data = self.clean_nans(data, cols=num_features)

        # Num Generator Features
        if num_generator_features:
            fe_df = self.numeric_interaction_terms(data[num_features], num_features)
            data = pd.concat([
                        data.reset_index(drop=True), 
                        fe_df.reset_index(drop=True)], 
                        axis=1,)

        # Group Encoder
        if group_generator_features:
            data = self.group_encoder(data, encodet_features_names, num_features)

        # Drop source cat features
        data.drop(columns=encodet_features_names, inplace=True)

        X_train = data.query('test == 0').drop(['test'], axis=1)
        X_test = data.query('test == 1').drop(['test'], axis=1)
        return(X_train, X_test)
