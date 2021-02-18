import pandas as pd
import numpy as np
from itertools import combinations
import pickle

from .encoders import *

# disable chained assignments
pd.options.mode.chained_assignment = None 


class CleanNans(object):
    """
    Сlass for cleaning Nans
    """

    def __init__(self, method='median'):
        """
        Fill Nans and add column, that there were nans in this column
        
        Args:
            method : ['median', 'mean',]
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
        
        data = data._get_numeric_data()
        
        self.nan_columns = list(data.columns[data.isnull().sum() > 0])
        if not self.nan_columns:     
            print('No nans features')

        if self.method == 'median':
            self.fill_value = data.median()
        elif self.method == 'mean':
            self.fill_value = data.mean()
        else:
            raise ValueError('Wrong fill method')

        return self

    def transform(self, data, cols=None) -> pd.DataFrame:
        """Transforms the dataset.
        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
            cols list() features: the input data
        Returns:
            pandas.Dataframe of shape = (n_train, n_features)
                The train dataset with no missing values.
        """
        if cols is not None:
            data = data[cols]

        if self.nan_columns:
            for nan_column in self.nan_columns:
                data[nan_column+'_isNAN'] = pd.isna(data[nan_column]).astype('uint8')
            
            data.fillna(self.fill_value, inplace=True)
        else:
            raise ValueError('No nans features')

        return data

    def fit_transform(self, data, cols=None) -> pd.DataFrame:
        """Fit and transforms the dataset.
        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
            cols list() features: the input data
        Returns:
            pandas.Dataframe of shape = (n_train, n_features)
                The train dataset with no missing values.
        """
        self.fit(data, cols)

        return self.transform(data)

class DataPrepare(object):
    """
    Сlass for cleaning, encoding and processing your dataset
    """
    def __init__(self, 
                cat_features=None,
                clean_and_encod_data=True,
                cat_encoder_names=['HelmertEncoder','CountEncoder'],
                clean_nan=True,
                num_generator_features=True,
                #group_generator_features=False,
                #frequency_enc_num_features=False,
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
            random_state=42 (undefined):
        """
        self.random_state = random_state
        self.cat_encoder_names = cat_encoder_names
        self.verbose = verbose
        self._clean_and_encod_data = clean_and_encod_data
        self._clean_nan = clean_nan
        self._num_generator_features = num_generator_features
        self.cat_features = cat_features

        self.binary_encoder = None
        self.clean_nan_encoder = None
        self.cat_clean_ord_encoder = None

        self.fit_cat_encoders={}

    def check_data_format(self, data):
        """
        Description of check_data_format:
            Check that data is not pd.DataFrame or empty

        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
        Return:
            True or Exception
        """
        if (not isinstance(data, pd.DataFrame)) or data.empty:
            raise Exception("data is not pd.DataFrame or empty")

    def check_num_nans(self, data):
        """
        Description of check_num_nans:
            Check Nans in numeric features in data 

        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
        Return:
            True or Exception
        """
        data = data._get_numeric_data()
        return(len(list(data.columns[data.isnull().sum() > 0])) > 0)

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
        if len(cat_features) < 1:
            cat_features = None
        #cat_features = list(set([*object_features, *cat_features]))
        return(cat_features)
    
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

    def fit(self, data,):
        """
        Fit DataPrepare.

        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): 
                the input data
        Returns:
            self
        """
        ########### check_data_format ######################
        self.check_data_format(data)

        if self.verbose > 0:   
            print('Source data shape: ', data.shape,)
            print('#'*50)
            print('! START FIT preprocessing Data')

        data = data.reset_index(drop=True)
        ########### Detect type of features ######################

        if self.cat_features is None:
            self.cat_features = self.auto_detect_cat_features(data)
            if self.verbose > 0:
                print('- Auto detect cat features: ', len(self.cat_features))

        self.binary_features = data.columns[data.nunique(dropna=False) <= 2]
        self.num_features = list(set(data.select_dtypes('number').columns) - set(self.binary_features))


        ########### Binary Features ######################
        if len(self.binary_features) > 0:
            if self.verbose > 0:
                    print('> Binary Features')

            self.binary_encoder = OrdinalEncoder()
            self.binary_encoder = self.binary_encoder.fit(data[self.binary_features])

        ########### Categorical Features ######################
        if self.cat_features is not None:
            # Clean Categorical Features
            if self.verbose > 0:
                    print('> Clean Categorical Features')
            self.cat_clean_ord_encoder = OrdinalEncoder()
            self.cat_clean_ord_encoder = self.cat_clean_ord_encoder.fit(data[self.cat_features])
            data[self.cat_features] = self.cat_clean_ord_encoder.transform(data[self.cat_features])


            # Encode Categorical Features
            if self.verbose > 0:
                    print('> Encode Categorical Features.')

            for cat_encoder_name in self.cat_encoder_names:
                if self.verbose > 0:
                    print(' +', cat_encoder_name)

                if cat_encoder_name not in cat_encoders_names.keys():
                    raise Exception(f"{cat_encoder_name} not support!")

                self.fit_cat_encoders[cat_encoder_name] = cat_encoders_names[cat_encoder_name](cols=self.cat_features, drop_invariant=True)
                if cat_encoder_name == 'HashingEncoder':
                    self.fit_cat_encoders[cat_encoder_name] = cat_encoders_names[cat_encoder_name](
                            n_components=int(np.log(len(data.columns))*1000), 
                            drop_invariant=True)
                
                self.fit_cat_encoders[cat_encoder_name] = \
                    self.fit_cat_encoders[cat_encoder_name].fit(data[self.cat_features])

        ########### Numerical Features ######################

        # CleanNans
        if self._clean_nan:
            if self.check_num_nans(data):
                self.clean_nan_encoder = CleanNans()
                self.clean_nan_encoder = self.clean_nan_encoder.fit(data[self.num_features])
                if self.verbose:
                    print('> CleanNans, total nans columns:', \
                        len(self.clean_nan_encoder.nan_columns))
            else:
                if self.verbose:
                    print('  No nans features')

        ########### Final ######################
        if self.verbose:
            print('#'*50)
            print('! END FIT preprocessing Data')
        return self

    def transform(self, data) -> pd.DataFrame:
        """Transform dataset.
        Args:
            data (pd.DataFrame, shape = (n_samples, n_features)): 
                the input data
        Returns:
            data (pd.Dataframe, shape = (n_train, n_features)):
                The dataset with clean numerical and encoded categorical features.
        """
        if self.verbose > 0:
            start_columns = len(data.columns)
            print('#'*50)
            print('! Start Transform Data')

        data = data.reset_index(drop=True)

        ########### Binary Features ######################
        
        if self.binary_encoder:
            data[self.binary_features] = self.binary_encoder.transform(data[self.binary_features]).replace(2,0).astype('category')
            if self.verbose:
                print('> Clean Binary Features')

        ########### Categorical Features ######################
        if self.cat_features is not None:
            # Clean Categorical Features
            if self.verbose > 0:
                print('> Clean Categorical Features')
            data[self.cat_features] = self.cat_clean_ord_encoder.transform(data[self.cat_features])

            # Encode Categorical Features
            if self.verbose > 0:
                print('> Transform Categorical Features.')
            for cat_encoder_name in self.cat_encoder_names:
                data_encodet = self.fit_cat_encoders[cat_encoder_name].transform(data[self.cat_features])
                data_encodet = data_encodet.add_prefix(cat_encoder_name + '_')
                if self.verbose > 0:
                    print(' - Encoder:', cat_encoder_name, 'ADD features:', len(data_encodet.columns))
                data = data.join(data_encodet.reset_index(drop=True))
        

        ########### Numerical Features ######################
        # CleanNans
        if self.clean_nan_encoder:
            data = self.clean_nan_encoder.transform(data)
            if self.verbose:
                print('> Clean Nans')

        # Generator interaction Num Features
        if self._num_generator_features:
            if len(self.num_features) > 1:
                if self.verbose > 0:
                    print('> Generate interaction Num Features')
                fe_df = self.gen_numeric_interaction_features(data[self.num_features], 
                                                            self.num_features,
                                                            operations=['/','*','-','+'],)
                data = data.join(fe_df.reset_index(drop=True))
                if self.verbose > 0:
                    print(' ADD features:', fe_df.shape[1],)
        
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0, inplace=True)


        ########### Final ######################
        if self.verbose > 0:
            end_columns = len(data.columns)
            print('#'*50)
            print('Final data shape: ', data.shape,)
            print('Total ADD columns:', end_columns-start_columns)
        return data

    def fit_transform(self, data,) -> pd.DataFrame:
        """Fits and transforms the dataset.
        Args:
            data (pd.DataFrame, shape = (n_samples, n_features)): 
                the input data
        Returns:
            data (pd.Dataframe, shape = (n_train, n_features)):
                The dataset with clean numerical and encoded categorical features.
        """
        self.fit(data)
        return self.transform(data)

    def save(self, name):
        pickle.dump(self, open(name+'.pkl', 'wb'), protocol=4)
        print('Save DataPrepare')

    def load(self, name):
        return(pickle.load(open(name+'.pkl', 'rb')))