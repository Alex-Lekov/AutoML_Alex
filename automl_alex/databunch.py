import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler

from .encoders import *

# disable chained assignments
pd.options.mode.chained_assignment = None 


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
                frequency_enc_num_features=True,
                normalization=True,
                random_state=42,
                verbose=1):
        """
        Description of __init__

        Args:
            X_train=None (undefined): dataset
            y_train=None (undefined): y 
            X_test=None (undefined): dataset
            y_test=None (undefined): y
            cat_features=None (list or None): 
            clean_and_encod_data=True (undefined):
            cat_encoder_names=None (list or None):
            clean_nan=True (undefined):
            num_generator_features=True (undefined):
            group_generator_features=True (undefined):
            random_state=42 (undefined):

        """
        self.random_state = random_state
        
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_train_predicts = None
        self.X_test_predicts = None
        self.cat_features = None

        # Encoders
        # if cat_encoder_names is None:
        #     self.cat_encoder_names = cat_encoders_names.keys()
        # else:
        #     self.cat_encoder_names = cat_encoder_names
        self.cat_encoder_names = cat_encoder_names

        # check X_train, y_train, X_test
        if self.check_data_format(X_train):
            self.X_train_source = pd.DataFrame(X_train)
        if X_test is not None:
            if self.check_data_format(X_test):
                self.X_test_source = pd.DataFrame(X_test)
                
        if y_train is not None:
            self.y_train = pd.DataFrame(y_train)
        else:
            raise Exception("No target data!")
        
        if y_test is not None:
            self.y_test = y_test
        
        if verbose > 0:   
            print('Source X_train shape: ', X_train.shape, '| X_test shape: ', X_test.shape)
            print('#'*50)
        
        # add categorical features in DataBunch
        if cat_features is None:
            self.cat_features = self.auto_detect_cat_features(self.X_train_source)
            if verbose > 0:
                print('Auto detect cat features: ', len(self.cat_features))
                
        else:
            self.cat_features = list(cat_features)
        
        # preproc_data in DataBunch 
        if clean_and_encod_data:
            if verbose > 0:
                print('> Start preprocessing Data')
            self.X_train, self.X_test = self.preproc_data(self.X_train_source, 
                                                            self.X_test_source, 
                                                            cat_features=self.cat_features,
                                                            cat_encoder_names=cat_encoder_names,
                                                            clean_nan=clean_nan,
                                                            num_generator_features=num_generator_features,
                                                            group_generator_features=group_generator_features,
                                                            frequency_enc_num_features=frequency_enc_num_features,
                                                            normalization=normalization,
                                                            verbose=verbose,)
        else: 
            self.X_train, self.X_test = X_train, X_test
                                        
    
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


    def gen_cat_encodet_features(self, data, cat_encoder_name) -> pd.DataFrame:
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
            if cat_encoder_name == 'FrequencyEncoder':
                encoder = cat_encoders_names['OrdinalEncoder']()
                data = encoder.fit_transform(data)
                encoder = cat_encoders_names['FrequencyEncoder']()

            data_encodet = encoder.fit_transform(data)
            data_encodet = data_encodet.add_prefix(cat_encoder_name + '_')
        else:
            raise Exception(f"{cat_encoder_name} not support!")
        return(data_encodet)


    def gen_numeric_interaction_features(self, 
                                        df, 
                                        columns, 
                                        operations=['/','*','-','+'],) -> pd.DataFrame:
        """
        Description of numeric_interaction_terms:
            Numerical interaction generator features: A/B, A*B, A-B,

        Args:
            df (pd.DataFrame):
            columns (list): num columns names
            operations (list): operations type

        Returns:
            pd.DataFrame

        """
        fe_df = pd.DataFrame()
        for c in combinations(columns,2):
            if '/' in operations:
                fe_df['{}_/_{}'.format(c[0], c[1]) ] = (df[c[0]]*1.) / df[c[1]]
            if '*' in operations:
                fe_df['{}_*_{}'.format(c[0], c[1]) ] = df[c[0]] * df[c[1]]
            if '-' in operations:
                fe_df['{}_-_{}'.format(c[0], c[1]) ] = df[c[0]] - df[c[1]]
            if '+' in operations:
                fe_df['{}_+_{}'.format(c[0], c[1]) ] = df[c[0]] + df[c[1]]
        return(fe_df)


    def gen_groupby_cat_encode_features(self, data, cat_columns, num_column) -> pd.DataFrame:
        """
        Description of group_encoder

        Args:
            data (pd.DataFrame): dataset
            cat_columns (list): cat columns names
            num_column (str): num column name

        Returns:
            pd.DataFrame

        """
        encoder = JamesSteinEncoder(drop_invariant=True)
        data_encodet = encoder.fit_transform(X=data[cat_columns], y=data[num_column].values)
        data_encodet = data_encodet.add_prefix('GroupEncoder_' + num_column + '_')
        return(data_encodet)


    def preproc_data(self, X_train=None, 
                        X_test=None, 
                        cat_features=None,
                        cat_encoder_names=None,
                        clean_nan=True,
                        num_generator_features=True,
                        group_generator_features=True,
                        frequency_enc_num_features=True,
                        normalization=True,
                        verbose=1,):
        """
        Description of preproc_data:
            dataset preprocessing function

        Args:
            X_train=None (pd.DataFrame):
            X_test=None (pd.DataFrame):
            cat_features=None (list):
            cat_encoder_names=None (list):
            clean_nan=True (Bool):
            num_generator_features=True (Bool):
            group_generator_features=True (Bool):
            
        Returns:
            X_train (pd.DataFrame)
            X_test (pd.DataFrame)

        """
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
        self.num_features_names = num_features
        self.binary_features_names = []

        # LabelEncoded Binary Features
        for feature in data.columns:
            if (feature != 'test') and (data[feature].nunique(dropna=False) < 3):
                data[feature] = data[feature].astype('category').cat.codes
                self.binary_features_names.append(feature)
                #if len(encodet_features_names) > 0:
                #    if feature in encodet_features_names:
                #        encodet_features_names.remove(feature)
                #        self.encodet_features_names = encodet_features_names
                        
        # Generator cat encodet features
        if encodet_features_names:
            if cat_encoder_names is None:
                for feature in encodet_features_names:
                    data[feature] = data[feature].astype('category').cat.codes
            else:
                if verbose > 0:
                    print('> Generate cat encodet features')
                for encoder_name in cat_encoder_names:
                    data_encodet = self.gen_cat_encodet_features(data[encodet_features_names], 
                                                                encoder_name,)
                    data = pd.concat([
                        data.reset_index(drop=True), 
                        data_encodet.reset_index(drop=True)], 
                        axis=1,)
                    if verbose > 0:
                        print(' + ', data_encodet.shape[1], ' Features from ', encoder_name)

        # Generate FrequencyEncoder num features
        if frequency_enc_num_features:
            if num_features:
                if verbose > 0:
                    print('> Generate Frequency Encode num features')
                encoder = cat_encoders_names['FrequencyEncoder']()
                data_encodet = encoder.fit_transform(data[num_features])
                data_encodet = data_encodet.add_prefix('FrequencyEncoder' + '_')
                data = pd.concat([data.reset_index(drop=True), 
                        data_encodet.reset_index(drop=True)], 
                        axis=1,)
                if verbose > 0:
                    print(' + ', data_encodet.shape[1], ' Frequency Encode Num Features ',)

        # Nans
        if clean_nan:
            if verbose > 0:
                print('> Clean Nans in num features')
            data = self.clean_nans(data, cols=num_features)

        # Generator interaction Num Features
        if num_generator_features:
            if len(num_features) > 1:
                if verbose > 0:
                    print('> Generate interaction Num Features')
                fe_df = self.gen_numeric_interaction_features(data[num_features], 
                                                            num_features,
                                                            operations=['/','*','-','+'],)
                data = pd.concat([data.reset_index(drop=True), 
                            fe_df.reset_index(drop=True)], 
                            axis=1,)
                if verbose > 0:
                    print(' + ', fe_df.shape[1], ' Interaction Features')

        # Generator Group Encoder Features
        if group_generator_features:
            if encodet_features_names and num_features:
                if verbose > 0:
                    print('> Generate Group Encoder Features')
                count = 0
                for num_col in num_features:
                    data_encodet = self.gen_groupby_cat_encode_features(
                        data,
                        encodet_features_names, 
                        num_col,)
                    data = pd.concat([data.reset_index(drop=True), 
                                data_encodet.reset_index(drop=True)], 
                                axis=1,)
                    count += data_encodet.shape[1]
                if verbose > 0:
                    print(' + ', count, ' Group cat Encoder Features')

        # Drop source cat features
        data.drop(columns=encodet_features_names, inplace=True)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0, inplace=True)
        
        X_train = data.query('test == 0').drop(['test'], axis=1)
        X_test = data.query('test == 1').drop(['test'], axis=1)

        # Normalization Data
        if normalization:
            if verbose > 0:
                print('> Normalization Features')
            columns_name = X_train.columns.values
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            X_train = pd.DataFrame(X_train, columns=columns_name)
            X_test = pd.DataFrame(X_test, columns=columns_name)

        if verbose > 0:
            print('#'*50)
            print('> Total Features: ', (X_train.shape[1]))
            print('#'*50)
            print('New X_train shape: ', X_train.shape, '| X_test shape: ', X_test.shape)
        
        return(X_train, X_test)
