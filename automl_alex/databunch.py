import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import Normalizer, RobustScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler
import warnings
import os

#from .encoders import FrequencyEncoder
#from .automl_alex import *
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
                    cat_encoder_name='OneHotEncoder',
                    target_encoder_name='JamesSteinEncoder',
                    clean_nan=True,
                    random_state=42):
        self.random_state = random_state
        self.cat_encoder_name = cat_encoder_name
        self.target_encoder_name = target_encoder_name
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_train_predicts = None
        self.X_test_predicts = None
        self.cat_features = None
        
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
                                                            cat_encoder_name=cat_encoder_name,
                                                            clean_nan=clean_nan,)
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


    def preproc_data(self, X_train=None, 
                        X_test=None, 
                        cat_features=None,
                        cat_encoder_name=None,
                        clean_nan=True,):
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
            if cat_encoder_name in encoders_names.keys():
                encoder = encoders_names[cat_encoder_name](drop_invariant=True) 
                if cat_encoder_name == 'HashingEncoder':
                    encoder = encoders_names[cat_encoder_name](n_components=int(np.log(len(data))*100), 
                                                            drop_invariant=True)
                data_encodet = encoder.fit_transform(data[encodet_features_names])
                data_encodet = data_encodet.add_prefix(cat_encoder_name + '_')
            else:
                raise Exception(f"{cat_encoder_name} not support!")

            if self.target_encoder_name is not None:
                data = pd.concat([
                    data.reset_index(drop=True), 
                    data_encodet.reset_index(drop=True)], 
                    axis=1,)
            else:
                data = pd.concat([
                    data.drop(columns=encodet_features_names).reset_index(drop=True), 
                    data_encodet.reset_index(drop=True)], 
                    axis=1,)

                
        # Nans
        if clean_nan:
            data = self.clean_nans(data, cols=num_features)

        X_train = data.query('test == 0').drop(['test'], axis=1)
        X_test = data.query('test == 1').drop(['test'], axis=1)
        return(X_train, X_test)


    def use_scaler(self, train_x=None, val_x=None, test_x=None, name='StandardScaler'):
        """
        Args:
            train_x (pd.DataFrame, shape (n_samples, n_features)): the input data
            test_x (pd.DataFrame, shape (n_samples, n_features)): the input data
            name (str): name Scaler from sklearn.preprocessing
        Return:
            Scaled train_x test_x
        """
        preprocessor_dict = {
            'MaxAbsScaler': MaxAbsScaler(), 
            'MinMaxScaler': MinMaxScaler(),
            'Normalizer': Normalizer(),
            'RobustScaler': RobustScaler(),
            'StandardScaler': StandardScaler(),
            }
            
        scaler = preprocessor_dict[name]
        # train
        train_x = pd.DataFrame(scaler.fit_transform(train_x))
        # val
        if val_x is not None:
            val_x = pd.DataFrame(scaler.transform(val_x))
        # test
        if test_x is not None:
            test_x = pd.DataFrame(scaler.transform(test_x))
        return(train_x, val_x, test_x)

    def target_encodet(self, train_x, train_y, val_x, test_x=None):
        """
        Encodet data in TargetEncoder
        """
        if self.target_encoder_name is not None:
            if self.encodet_features_names:
                encoder = target_encoders_names[self.target_encoder_name](
                    cols=self.encodet_features_names,
                    drop_invariant=True) 
                train_x = encoder.fit_transform(
                    train_x.reset_index(drop=True), 
                    train_y.reset_index(drop=True),
                    )
                val_x = encoder.transform(val_x)
                if test_x is not None:
                    test_x = encoder.transform(test_x)
        return(train_x, val_x, test_x)
