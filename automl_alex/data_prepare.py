'''
Data processing, cleaning, and encoding
'''
from typing import Optional
from typing import Callable
from typing import List
from typing import Tuple
import pandas as pd
import numpy as np
import random
from itertools import combinations
import joblib
import sys
import gc
from pathlib import Path
import shutil

import tensorflow.keras.layers as L
from tensorflow.keras import Model

from ._encoders import *
from ._logger import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# disable chained assignments
pd.options.mode.chained_assignment = None 

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class DenoisingAutoencoder(object):
    '''
    Denoising autoencoders (DAE) for numerical features. try to achieve a good representation by changing the reconstruction criterion
    https://en.wikipedia.org/wiki/Autoencoder#Denoising_autoencoder_(DAE)

    Examples
    --------
    >>> from automl_alex import DenoisingAutoencoder, CleanNans
    >>> import sklearn
    >>> # Get Dataset
    >>> dataset = sklearn.datasets.fetch_openml(name='adult', version=1, as_frame=True)
    >>> dataset.target = dataset.target.astype('category').cat.codes
    >>> X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    >>>                                             dataset.data, 
    >>>                                             dataset.target,
    >>>                                             test_size=0.2,)
    >>> 
    >>> # clean nans before use
    >>> cn = CleanNans()
    >>> clean_X_train = cn.fit_transform(X_train)
    >>> clean_X_test = cn.transform(X_test)
    >>>
    >>> # get Numeric Features
    >>> num_columns = list(clean_X_train.select_dtypes('number').columns)
    >>> 
    >>> nf = DenoisingAutoencoder()
    >>> new_features_X_train = nf.fit_transform(clean_X_train, num_columns)
    >>> new_features_X_test = nf.transform(clean_X_test)
    '''    
    autoencoder = None

    def __init__(self, verbose: int = 0) -> None:
        '''
        Parameters
        ----------
        verbose : int,
            print state, by default 0
        '''        
        self.verbose = verbose

    def _get_dae(self, caunt_columns: int, units: Optional[int] = 512,):
        # denoising autoencoder
        inputs = L.Input((caunt_columns,))
        x = L.Dense(units, activation='relu')(inputs) # 1500 original
        x = L.Dense(units, activation='relu')(x) # 1500 original
        x = L.Dense(units, activation='relu')(x) # 1500 original
        outputs = L.Dense(caunt_columns, activation='relu')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        return model


    @logger.catch
    def fit(self, data: pd.DataFrame, cols: Optional[List[str]] = None) -> None:
        '''
        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))
        cols : Optional[List[str]], optional
            cols list features, by default None

        Returns
        -------
        self

        Raises
        ------
        Exception
            No numerical features
        '''        
        if cols is not None:
            data = data[cols]
        
        data = data._get_numeric_data()
        self.columns = data.columns
        count_columns = len(self.columns)

        if count_columns < 1:
            raise ValueError('No numerical features')

        self.scaler = MinMaxScaler().fit(data)
        s_data = self.scaler.transform(data)

        units = 512
        if count_columns > 512:
            units = count_columns

        self.autoencoder = self._get_dae(count_columns, units=units)
        self.autoencoder.fit (s_data, s_data,
                    epochs=50,
                    batch_size=124,
                    shuffle=True,
                    verbose=self.verbose)
        return(self)


    @logger.catch
    def transform(self, data: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
        '''
        Transforms the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))
        cols : Optional[List[str]], optional
            cols list features, by default None

        Returns
        -------
        pd.DataFrame
            The dataset with Transform data

        Raises
        ------
        Exception
            if No fit autoencoder
        '''
        if self.autoencoder is None:
            raise Exception("No fit autoencoder")

        if cols is not None:
            data = data[cols]

        s_data = self.scaler.transform(data[self.columns])
        encodet_data = self.autoencoder.predict(s_data)
        encodet_data = pd.DataFrame(encodet_data, columns=self.columns)
        return(encodet_data)


    def fit_transform(self, data: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
        '''
        Fit and Transforms the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))
        cols : Optional[List[str]], optional
            cols list features, by default None

        Returns
        -------
        pd.DataFrame
            The dataset with Transform data

        Raises
        ------
        Exception
            No numerical features
        '''        
        self.fit(data, cols)
        return self.transform(data)



class CleanNans(object):
    """
    Сlass Fill Nans numerical columns, method : ['median', 'mean',]

    Examples
    --------
    >>> from automl_alex import CleanNans
    >>> import sklearn
    >>> # Get Dataset
    >>> dataset = sklearn.datasets.fetch_openml(name='adult', version=1, as_frame=True)
    >>> dataset.target = dataset.target.astype('category').cat.codes
    >>> X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    >>>                                             dataset.data, 
    >>>                                             dataset.target,
    >>>                                             test_size=0.2,)
    >>> 
    >>> cn = CleanNans()
    >>> clean_X_train = cn.fit_transform(X_train)
    >>> clean_X_test = cn.transform(X_test)

    """

    def __init__(self, method: str = 'median', verbose: int = 0) -> None:
        '''
        Parameters
        ----------
        method : str, 
            Fill Nans, method = ['median', 'mean',], by default 'median'
        verbose : int,
            print state, by default 0
        '''        
        self.method = method
        self.verbose = verbose


    @logger.catch
    def fit(self, data: pd.DataFrame, cols: Optional[List[str]] = None) -> None:
        '''

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))
        cols : Optional[List[str]], optional
            cols list features, by default None

        Returns
        -------
        self
        '''        
        if cols is not None:
            data = data[cols]
        
        data = data._get_numeric_data()

        if self.verbose:
            for col in data.columns:
                pct_missing = np.mean(data[col].isnull())
                if pct_missing > 0.25:
                    logger.warning('! Attention {} - {}% Nans!'.format(col, round(pct_missing*100)))
        
        self.nan_columns = list(data.columns[data.isnull().sum() > 0])
        if not self.nan_columns:     
            logger.info('No nans features')

        if self.method == 'median':
            self.fill_value = data.median()
        elif self.method == 'mean':
            self.fill_value = data.mean()
        else:
            raise ValueError('Wrong fill method')
        return(self)


    @logger.catch
    def transform(self, data: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
        '''Transforms the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))
        cols : Optional[List[str]], optional
            cols list features, by default None

        Returns
        -------
        pd.DataFrame
            The dataset with no missing values.
        '''
        if cols is not None:
            data = data[cols]

        if self.nan_columns:
            for nan_column in self.nan_columns:
                data[nan_column+'_isNAN'] = pd.isna(data[nan_column]).astype('uint8')
            
            data.fillna(self.fill_value, inplace=True)
        else:
            raise ValueError('No nans features')

        return data


    @logger.catch
    def fit_transform(self, data: pd.DataFrame, cols: Optional[List[str]] = None) -> pd.DataFrame:
        '''
        Fit and Transforms the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))
        cols : Optional[List[str]], optional
            cols list features, by default None

        Returns
        -------
        pd.DataFrame
            The dataset with no missing values.
        '''        
        self.fit(data, cols)

        return self.transform(data)


class NumericInteractionFeatures(object):
    '''
    Class for  Numerical interaction generator features: A/B, A*B, A-B,

    Examples
    --------
    >>> from automl_alex import NumericInteractionFeatures, CleanNans
    >>> import sklearn
    >>> # Get Dataset
    >>> dataset = sklearn.datasets.fetch_openml(name='adult', version=1, as_frame=True)
    >>> dataset.target = dataset.target.astype('category').cat.codes
    >>> X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    >>>                                             dataset.data, 
    >>>                                             dataset.target,
    >>>                                             test_size=0.2,)
    >>> 
    >>> # clean nans before use
    >>> cn = CleanNans()
    >>> clean_X_train = cn.fit_transform(X_train)
    >>> clean_X_test = cn.transform(X_test)
    >>>
    >>> # get Numeric Features
    >>> num_columns = list(clean_X_train.select_dtypes('number').columns)
    >>> 
    >>> nf = NumericInteractionFeatures()
    >>> new_features_X_train = nf.fit_transform(clean_X_train, num_columns)
    >>> new_features_X_test = nf.transform(clean_X_test)
    '''    
    _cols_combinations = None


    def __init__(self, operations: List[str] = ['/','*','-','+'], verbose: int = 0) -> None:
        '''
        Parameters
        ----------
        operations : List[str], optional
            generator operations, by default ['/','*','-','+']
        verbose : int, optional
            print state, by default 0
        '''     
        self.operations = operations
        self.verbose = verbose
        self.columns = None


    def fit(self, columns: List[str],) -> None:
        '''
        Fit: generate combinations features

        Parameters
        ----------
        columns : List[str]
            list features names

        Returns
        -------
        self
        '''        
        self.columns = columns
        self._cols_combinations = list(combinations(columns,2))
        return(self)


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Transforms the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))

        Returns
        -------
        pd.DataFrame
            dataset with new features (pd.DataFrame shape = (n_samples, n_features))

        Raises
        ------
        Exception
            if No fit cols_combinations
        '''        
        if self._cols_combinations is None:
            raise Exception("No fit cols_combinations")

        fe_df = pd.DataFrame()

        for col1 in self.columns:
            for col2 in self.columns:
                if col1 == col2:
                    continue
                else:
                    if '/' in self.operations:
                        fe_df['{}_/_{}'.format(col1, col2) ] = (df[col1]*1.) / df[col2]
                    if '-' in self.operations:
                        fe_df['{}_-_{}'.format(col1, col2) ] = df[col1] - df[col2]

        for c in self._cols_combinations:
            if '*' in self.operations:
                fe_df['{}_*_{}'.format(c[0], c[1]) ] = df[c[0]] * df[c[1]]
            if '+' in self.operations:
                fe_df['{}_+_{}'.format(c[0], c[1]) ] = df[c[0]] + df[c[1]]
        return(fe_df)


    def fit_transform(self, data: pd.DataFrame, columns: List[str],) -> pd.DataFrame:
        '''
        Fit and transforms the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))
        columns : List[str]
            list features names

        Returns
        -------
        pd.DataFrame
            dataset with new features (pd.DataFrame shape = (n_samples, n_features))
        '''        
        self.fit(columns)

        return self.transform(data)


class CleanOutliers(object):
    '''
    A class method that takes a data column and removes outliers from it
    I would like to provide two methods solution based on "z score" and solution based on "IQR".

    Something important when dealing with outliers is that one should try to use estimators as robust as possible. 
    try different values threshold and method

    Examples
    --------
    >>> from automl_alex import CleanOutliers
    >>> import sklearn
    >>> # Get Dataset
    >>> dataset = sklearn.datasets.fetch_openml(name='adult', version=1, as_frame=True)
    >>> dataset.target = dataset.target.astype('category').cat.codes
    >>> X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    >>>                                             dataset.data, 
    >>>                                             dataset.target,
    >>>                                             test_size=0.2,)
    >>> 
    >>> co = CleanOutliers()
    >>> clean_X_train = co.fit_transform(clean_X_train)
    >>> clean_X_test = co.transform(clean_X_test)

    '''
    _weight = {}

    def __init__(self, method: str = 'IQR', threshold: int = 2, verbose: int = 0) -> None:
        '''
        Parameters
        ----------
        method : str, optional
            method ['IQR', 'z_score',], by default 'IQR'
        threshold : int, optional
            threshold on method, by default 2
        verbose : int, optional
            print state, by default 0
        ''' 
        self.method = method
        self.threshold = threshold
        self.verbose = verbose


    def _IQR(self, data: pd.DataFrame, colum_name: str, threshold=1.5) -> Tuple[float, float]:
        '''
        Outlier detection by Interquartile Ranges Rule, also known as Tukey's test. 
        calculate the IQR ( 75th quantile - 25th quantile) 
        and the 25th 75th quantile. 
        Any value beyond:
            upper bound = 75th quantile + （IQR * threshold）
            lower bound = 25th quantile - （IQR * threshold）   
        are regarded as outliers. Default threshold is 1.5.

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))
        colum_name : str
            feature name
        threshold : float, optional
            threshold on method, by default 1.5

        Returns
        -------
        Tuple[float, float]
            lower_bound and upper_bound
        '''        
        quantile1,quantile3 = np.percentile(data[colum_name],[25,75])
        iqr_val = quantile3 - quantile1
        lower_bound = quantile1 - (threshold*iqr_val)
        upper_bound = quantile3 + (threshold*iqr_val)
        return(lower_bound, upper_bound)


    def _fit_z_score(self, data: pd.DataFrame, colum_name: str,) -> Tuple[float, float]:
        '''
        Z score is an important measurement or score that tells how many Standard deviation above or below a number is from the mean of the dataset
        Any positive Z score means the no. of standard deviation above the mean and a negative score means no. of standard deviation below the mean
        Z score is calculate by subtracting each value with the mean of data and dividing it by standard deviation
        

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))
        colum_name : str
            feature name

        Returns
        -------
        Tuple[float, float]
            median, median_absolute_deviation
        '''        
        median_y = data[colum_name].median()
        median_absolute_deviation_y = (np.abs(data[colum_name]-median_y)).median()
        return(median_y, median_absolute_deviation_y)


    def _get_z_score(self, median, mad, data, col, threshold=3):
        '''
        Its Modified z_score

        The goal of taking Z-scores is to remove the effects of the location and scale of the data, 
        allowing different datasets to be compared directly. 
        The intuition behind the Z-score method of outlier detection is that, once we’ve centred and rescaled the data, 
        anything that is too far from zero (the threshold is usually a Z-score of 3 or -3) should be considered an outlier.

        The Z-score method relies on the mean and standard deviation of a group of data to measure central tendency and dispersion. 
        This is troublesome, because the mean and standard deviation are highly affected by outliers – they are not robust. 
        In fact, the skewing that outliers bring is one of the biggest reasons for finding and removing outliers from a dataset!

        Another drawback of the Z-score method is that it behaves strangely in small datasets – in fact, the Z-score method will never detect an outlier if the dataset has fewer than 12 items in it. 
        This motivated the development of a modified Z-score method, which does not suffer from the same limitation

        A further benefit of the modified Z-score method is that it uses the median and MAD rather than the mean and standard deviation. 
        The median and MAD are robust measures of central tendency and dispersion, respectively.

        Default threshold is 3.
        '''
        modified_z_scores = 0.7413 *((data[col] - median)/mad)
        abs_z_scores = np.abs(modified_z_scores)
        filtered_entries = (abs_z_scores > threshold)
        return(filtered_entries)


    def fit(self, data: pd.DataFrame, columns: List[str],) -> None:
        '''
        Fit CleanOutliers

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))
        columns : List[str]
            list features names

        Returns
        -------
        self

        Raises
        ------
        ValueError
            Wrong method
        '''        
        if columns is not None:
            chek_columns = columns
        else:
            chek_columns = data._get_numeric_data().columns

        self._weight = {}

        for column in chek_columns:
            if self.method == 'IQR':
                lower_bound, upper_bound = self._IQR(data,colum_name=column,threshold=self.threshold)
                self._weight[column] = [lower_bound, upper_bound]
                #logger.info(self._weight)
                if self.verbose:
                    total_outliers = len(data[column][(data[column] < lower_bound) | (data[column] > upper_bound)])

            elif self.method == 'z_score':
                median, mad = self._fit_z_score(data, col=column)
                self._weight[column] = [median, mad]
                if self.verbose:
                    filtered_entries = self._get_z_score(median, mad, data, column, threshold=self.threshold)
                    total_outliers = filtered_entries.sum()
            else:
                raise ValueError('Wrong method')

            if self.verbose:
                if total_outliers > 0:
                    logger.info(f'Num of outlier detected: {total_outliers} in Feature {column}')
                    logger.info(f'Proportion of outlier detected: {round((100/(len(data)/total_outliers)),1)} %')
        return(self)


    def transform(self, data: pd.DataFrame,) -> pd.DataFrame:
        '''
        Transforms the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))

        Returns
        -------
        pd.DataFrame
            dataset with cleaned features (pd.DataFrame shape = (n_samples, n_features))
        ''' 
        #logger.info(self._weight)
        for weight_values in self._weight:
            if self.method == 'IQR':
                data.loc[data[weight_values] < self._weight[weight_values][0], weight_values] = self._weight[weight_values][0]
                data.loc[data[weight_values] > self._weight[weight_values][1], weight_values] = self._weight[weight_values][1]

                feature_name = weight_values+'_Is_Outliers_'+self.method
                data[feature_name] = 0
                data.loc[
                    (data[weight_values] < self._weight[weight_values][0]) | (data[weight_values] > self._weight[weight_values][1]), \
                    feature_name
                    ] = 1

            elif self.method == 'z_score':
                filtered_entries = self._get_z_score(
                    self._weight[weight_values][0], 
                    self._weight[weight_values][1], 
                    data, 
                    weight_values, 
                    threshold=self.threshold
                    )
                data.loc[filtered_entries, weight_values] = data[weight_values].median()

                feature_name = weight_values+'_Is_Outliers_'+self.method
                data[feature_name] = 0
                data.loc[filtered_entries, feature_name] = 1

        return data


    def fit_transform(self, data: pd.DataFrame, columns: List[str],) -> pd.DataFrame:
        '''
        Fit and transforms the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))
        columns : List[str]
            list features names

        Returns
        -------
        pd.DataFrame
            dataset with new features (pd.DataFrame shape = (n_samples, n_features))
        '''        
        self.fit(data, columns)
        return self.transform(data)


class DataPrepare(object):
    """
    Сlass for cleaning, encoding and processing your dataset

    Examples
    --------
    >>> from automl_alex import DataPrepare
    >>> import sklearn
    >>> # Get Dataset
    >>> dataset = sklearn.datasets.fetch_openml(name='adult', version=1, as_frame=True)
    >>> dataset.target = dataset.target.astype('category').cat.codes
    >>> X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    >>>                                             dataset.data, 
    >>>                                             dataset.target,
    >>>                                             test_size=0.2,)
    >>> 
    >>> de = DataPrepare()
    >>> clean_X_train = de.fit_transform(X_train)
    >>> clean_X_test = de.transform(X_test)

    """
    _clean_outliers_enc = None
    _binary_encoder = None
    _clean_nan_encoder = None
    _cat_clean_ord_encoder = None
    _fit_cat_encoders={}
    _fited = False


    def __init__(self, 
                cat_features: Optional[List[str]] = None,
                clean_and_encod_data: bool = True,
                cat_encoder_names: List[str] = ['HelmertEncoder','CountEncoder'],
                clean_nan: bool = True,
                clean_outliers: bool = True,
                outliers_method: str ='IQR', # method : ['IQR', 'z_score',]
                outliers_threshold: int = 2,
                drop_invariant: bool = True,
                num_generator_features: bool = True,
                operations_num_generator: List[str] = ['/','*','-',],
                num_denoising_autoencoder: bool = False,
                #group_generator_features=False,
                #frequency_enc_num_features=False,
                normalization: bool = False,
                reduce_memory: bool = False,
                random_state: int = 42,
                verbose: int = 3) -> None:
        '''
        Parameters
        ----------
        cat_features : Optional[List[str]], optional
            features name list. if None -> Auto-detection categorical_features, by default None
        clean_and_encod_data : bool, optional
            On or Off cleaning, by default True
        cat_encoder_names : List[str], optional
            name encoders (from automl_alex._encoders.cat_encoders_names), by default ['HelmertEncoder','CountEncoder']
        clean_nan : bool, optional
            On or Off, by default True
        clean_outliers : bool, optional
            On or Off, by default True
        outliers_method : str, optional
            method 'IQR' or 'z_score', by default 'IQR'
        drop_invariant : bool, optional
            drop invariant features, by default True
        num_generator_features : bool, optional
            generate num features, by default True
        operations_num_generator : List[str], optional
            operations for generate num features, by default ['/','*','-',]
        num_denoising_autoencoder : bool, optional
            generate num denoising autoencoder features, by default False
        normalization : bool, optional
            On or Off, by default True
        reduce_memory : bool, optional
            On or Off, by default False
        random_state : int, optional
            Controls the generation of the random states for each repetition, by default 42
        verbose : int, optional
            print state, by default 3
        '''
        self.random_state = random_state
        self.cat_encoder_names = cat_encoder_names

        self.verbose = verbose
        logger_print_lvl(self.verbose)

        self._clean_and_encod_data = clean_and_encod_data
        self._clean_nan = clean_nan
        self._clean_outliers = clean_outliers
        self._outliers_threshold = outliers_threshold
        self._drop_invariant = drop_invariant
        self._outliers_method = outliers_method
        self._num_generator_features = num_generator_features
        self._num_denoising_autoencoder = num_denoising_autoencoder
        self._operations_num_generator = operations_num_generator
        self._normalization = normalization
        self._reduce_memory = reduce_memory
        self.cat_features = cat_features


    def _check_data_format(self, data: pd.DataFrame) -> None:
        '''Check that data is not pd.DataFrame or empty

        Parameters
        ----------
        data : pd.DataFrame
            data (pd.DataFrame, shape (n_samples, n_features))

        Raises
        ------
        Exception
            if data is not pd.DataFrame or empty
        '''        
        if (not isinstance(data, pd.DataFrame)) or data.empty:
            raise Exception("data is not pd.DataFrame or empty")


    def _check_num_nans(self, data: pd.DataFrame) -> bool:
        """
        Check Nans in numeric features in data 

        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
        Return:
            bool: True or False
        """
        data = data._get_numeric_data()
        return(len(list(data.columns[data.isnull().sum() > 0])) > 0)


    def auto_detect_cat_features(self, data: pd.DataFrame) -> List[str]:
        '''
        Auto-detection categorical_features by simple rule:
        categorical feature == if feature nunique low 1% of data

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))

        Returns
        -------
        List[str]
            categorical features names
        '''        
        #object_features = list(data.columns[data.dtypes == 'object'])
        cat_features = data.columns[(data.nunique(dropna=False) < len(data)//100) & \
            (data.nunique(dropna=False) >2)]
        if len(cat_features) < 1:
            cat_features = None
        else:
            cat_features = list(cat_features)
        #cat_features = list(set([*object_features, *cat_features]))
        return(cat_features)


    def fit_transform(self, data: pd.DataFrame, verbose: bool = None) -> pd.DataFrame:
        '''
        Fit and transforms the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))
        verbose : bool, optional
            print info, by default None

        Returns
        -------
        pd.DataFrame
            The dataset with clean numerical and encoded categorical features.
            shape = (n_samples, n_features)
        '''        
        if verbose is not None:
            self.verbose =  verbose
        logger_print_lvl(self.verbose)
        ########### check_data_format ######################
        self._check_data_format(data)

        start_columns = len(data.columns)
        logger.info(f'Source data shape: {data.shape}')
        logger.info('#'*50)
        logger.info('! START preprocessing Data')

        data = data.reset_index(drop=True)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        ########### Drop invariant  features ######################
        if self._drop_invariant:
            self._drop_invariant_features = \
                data.columns[data.nunique(dropna=False) < 2]
            if len(self._drop_invariant_features) > 0:
                data.drop(self._drop_invariant_features, axis=1, inplace=True)

        ########### Detect type of features ######################

        if self.cat_features is None:
            self.cat_features = self.auto_detect_cat_features(data)
            if self.cat_features is not None:
                logger.info(f'- Auto detect cat features: {len(self.cat_features)}')

        self.binary_features = data.columns[data.nunique(dropna=False) <= 2]
        self.num_features = list(set(data.select_dtypes('number').columns) - set(self.binary_features))
        self.object_features = list(set(data.columns[(data.dtypes == 'object') | (data.dtypes == 'category')]) - set(self.binary_features))


        ########### Binary Features ######################
        if len(self.binary_features) > 0:
            logger.info('> Binary Features')

            self._binary_encoder = OrdinalEncoder()
            self._binary_encoder = self._binary_encoder.fit(data[self.binary_features])
            data[self.binary_features] = self._binary_encoder.transform(data[self.binary_features]).replace(2,0)
            

        ########### Categorical Features ######################
        #if self.cat_features is not None:
        # Clean Categorical Features
        if self.object_features is not None:
            logger.info('> Clean Categorical Features')
            self._cat_clean_ord_encoder = OrdinalEncoder()
            self._cat_clean_ord_encoder = self._cat_clean_ord_encoder.fit(data[self.object_features])
            data[self.object_features] = self._cat_clean_ord_encoder.transform(data[self.object_features])


        if self.cat_features is not None:
            # Encode Categorical Features
            logger.info('> Transform Categorical Features.')

            for cat_encoder_name in self.cat_encoder_names:

                if cat_encoder_name not in cat_encoders_names.keys():
                    raise Exception(f"{cat_encoder_name} not support!")

                self._fit_cat_encoders[cat_encoder_name] = cat_encoders_names[cat_encoder_name](cols=self.cat_features, drop_invariant=True)
                if cat_encoder_name == 'HashingEncoder':
                    self._fit_cat_encoders[cat_encoder_name] = cat_encoders_names[cat_encoder_name](
                            n_components=int(np.log(len(data.columns))*1000), 
                            drop_invariant=True)
                
                self._fit_cat_encoders[cat_encoder_name] = \
                    self._fit_cat_encoders[cat_encoder_name].fit(data[self.cat_features])

                data_encodet = self._fit_cat_encoders[cat_encoder_name].transform(data[self.cat_features])
                data_encodet = data_encodet.add_prefix(cat_encoder_name + '_')
                if self._reduce_memory: 
                    data_encodet = reduce_mem_usage(data_encodet)
                logger.info(f' - Encoder: {cat_encoder_name} ADD features: {len(data_encodet.columns)}')
                data = data.join(data_encodet.reset_index(drop=True))

        ########### Numerical Features ######################

        # CleanNans
        if self._clean_nan:
            if self._check_num_nans(data):
                self._clean_nan_encoder = CleanNans(verbose=self.verbose)
                self._clean_nan_encoder.fit(data[self.num_features])
                data = self._clean_nan_encoder.transform(data)
                logger.info(f'> CleanNans, total nans columns: {len(self._clean_nan_encoder.nan_columns)}')
            else:
                logger.info('  No nans features')

        # DenoisingAutoencoder
        if self._num_denoising_autoencoder:
            if len(self.num_features) > 2:
                logger.info('> Start fit DenoisingAutoencoder')
                self._autoencoder = DenoisingAutoencoder(verbose=self.verbose)
                data_encodet = self._autoencoder.fit_transform(data[self.num_features])
                data_encodet = data_encodet.add_prefix('DenoisingAutoencoder_')
                data = pd.concat([
                            data.reset_index(drop=True), 
                            data_encodet.reset_index(drop=True)], 
                            axis=1,)
                logger.info('> Add Denoising features')

        # CleanOutliers
        if self._clean_outliers:
            logger.info('> CleanOutliers',)
            self._clean_outliers_enc = CleanOutliers(
                threshold=self._outliers_threshold, 
                method=self._outliers_method,
                verbose=self.verbose)
            self._clean_outliers_enc.fit(data, columns=self.num_features)
            data = self._clean_outliers_enc.transform(data)

        # Generator interaction Num Features
        if self._num_generator_features:
            if len(self.num_features) > 1:
                logger.info('> Generate interaction Num Features')
                self.num_generator = NumericInteractionFeatures(operations=self._operations_num_generator,)
                self.num_generator.fit(list(self.num_features))
                fe_df = self.num_generator.transform(data[self.num_features])
                
                if self._reduce_memory:
                    fe_df = reduce_mem_usage(fe_df)
                data = pd.concat([
                        data.reset_index(drop=True), 
                        fe_df.reset_index(drop=True)], 
                        axis=1,)
                #data = data.join(fe_df.reset_index(drop=True))
                logger.info(f' ADD features: {fe_df.shape[1]}')

        
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.data_median_dict = data.median()
        data.fillna(self.data_median_dict, inplace=True)

        ########### Normalization ######################
        if self._normalization:
            logger.info('> Normalization Features')
            self.normalization_features = data.columns[data.nunique(dropna=False) > 2].values
            self.scaler = StandardScaler().fit(data[self.normalization_features])
            data_tmp = self.scaler.transform(data[self.normalization_features])
            data_tmp = pd.DataFrame(data_tmp, columns=self.normalization_features)
            data.drop(self.normalization_features, axis=1, inplace=True)
            data = pd.concat([
                        data.reset_index(drop=True), 
                        data_tmp.reset_index(drop=True)], 
                        axis=1,)
            #data[self.normalization_features] = data_tmp[self.normalization_features]
            data_tmp = None

        ########### reduce_mem_usage ######################
        if self._reduce_memory:
            logger.info('> Reduce_Memory')
            data = reduce_mem_usage(data, verbose=self.verbose)
        data.fillna(0, inplace=True)

        ########### Final ######################
        end_columns = len(data.columns)
        logger.info('#'*50)
        logger.info(f'Final data shape: {data.shape}')
        logger.info(f'Total ADD columns: {end_columns-start_columns}')
        logger.info('#'*50)
        self._fited = True
        return data


    def transform(self, data: pd.DataFrame, verbose: bool = None) -> pd.DataFrame:
        '''
        Transforms the dataset.

        Parameters
        ----------
        data : pd.DataFrame
            dataset (pd.DataFrame shape = (n_samples, n_features))
        verbose : bool, optional
            print info, by default None

        Returns
        -------
        pd.DataFrame
            The dataset with clean numerical and encoded categorical features.
            shape = (n_samples, n_features)

        Raises
        ------
        Exception
            if not fited fit_tranform
        '''        
        if not self._fited:
            raise Exception("not fited. use fit_tranform at first")

        if verbose is not None:
            self.verbose = verbose

        logger_print_lvl(self.verbose)

        ########### check_data_format ######################
        self._check_data_format(data)

        start_columns = len(data.columns)
        logger.info('#'*50)
        logger.info('! Start Transform Data')

        data = data.reset_index(drop=True)
        data.replace([np.inf, -np.inf], np.nan, inplace=True)

        ########### Drop invariant  features ######################
        if self._drop_invariant:
            if len(self._drop_invariant_features) > 0:
                data.drop(self._drop_invariant_features, axis=1, inplace=True)

        ########### Binary Features ######################
        
        if self._binary_encoder:
            data[self.binary_features] = self._binary_encoder.transform(data[self.binary_features]).replace(2,0)
            logger.info('> Clean Binary Features')

        ########### Categorical Features ######################
        # Clean Categorical Features
        if self.object_features is not None:
            logger.info('> Clean Categorical Features')
            data[self.object_features] = self._cat_clean_ord_encoder.transform(data[self.object_features])
        
        if self.cat_features is not None:
            # Encode Categorical Features
            logger.info('> Transform Categorical Features.')
            for cat_encoder_name in self.cat_encoder_names:
                data_encodet = self._fit_cat_encoders[cat_encoder_name].transform(data[self.cat_features])
                data_encodet = data_encodet.add_prefix(cat_encoder_name + '_')
                if self._reduce_memory:
                    data_encodet = reduce_mem_usage(data_encodet)
                if self.verbose > 0:
                    logger.info(f' - Encoder: {cat_encoder_name} ADD features: {len(data_encodet.columns)}')
                data = data.join(data_encodet.reset_index(drop=True))

        ########### Numerical Features ######################

        # CleanNans
        if self._clean_nan_encoder:
            data = self._clean_nan_encoder.transform(data)
            logger.info('> Clean Nans')

        # DenoisingAutoencoder
        if self._num_denoising_autoencoder:
            if len(self.num_features) > 2:
                data_encodet = self._autoencoder.transform(data[self.num_features])
                data_encodet = data_encodet.add_prefix('DenoisingAutoencoder_')
                data = pd.concat([
                            data.reset_index(drop=True), 
                            data_encodet.reset_index(drop=True)], 
                            axis=1,)
                logger.info('> Add Denoising features')

        # CleanOutliers
        if self._clean_outliers:
            data = self._clean_outliers_enc.transform(data)

        # Generator interaction Num Features
        if self._num_generator_features:
            if len(self.num_features) > 1:
                logger.info('> Generate interaction Num Features')
                fe_df = self.num_generator.transform(data[self.num_features])
                
                if self._reduce_memory:
                    fe_df = reduce_mem_usage(fe_df)
                data = pd.concat([
                        data.reset_index(drop=True), 
                        fe_df.reset_index(drop=True)
                        ], axis=1,)
                #data = data.join(fe_df.reset_index(drop=True))
                logger.info(f' ADD features: {fe_df.shape[1]}')
        
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(self.data_median_dict, inplace=True)

        ########### Normalization ######################
        if self._normalization:
            logger.info('> Normalization Features')
            data_tmp = self.scaler.transform(data[self.normalization_features])
            data_tmp = pd.DataFrame(data_tmp, columns=self.normalization_features)
            data.drop(self.normalization_features,axis=1,inplace=True)
            data = pd.concat([
                        data.reset_index(drop=True), 
                        data_tmp.reset_index(drop=True)], 
                        axis=1,)
            #data[self.normalization_features] = data_tmp[self.normalization_features]
            data_tmp=None

        ########### reduce_mem_usage ######################
        if self._reduce_memory:
            logger.info('> Reduce_Memory')
            data = reduce_mem_usage(data, verbose=self.verbose)
        data.fillna(0, inplace=True)

        ########### Final ######################

        end_columns = len(data.columns)
        logger.info('#'*50)
        logger.info(f'Final data shape: {data.shape}')
        logger.info(f'Total ADD columns: {end_columns-start_columns}')
        logger.info('#'*50)
        return data


    def save(self, name: str = 'DataPrepare_dump', folder: str = './') -> None:
        '''
        Save data prepare

        Parameters
        ----------
        name : str, optional
            file name, by default 'DataPrepare_dump'
        folder : str, optional
            target folder, by default './'
        '''

        dir_tmp = "./DataPrepare_tmp/"
        Path(dir_tmp).mkdir(parents=True, exist_ok=True)
        for cat_encoder_name in self.cat_encoder_names:
            joblib.dump(self._fit_cat_encoders[cat_encoder_name], \
                dir_tmp+cat_encoder_name+'.pkl')

        joblib.dump(self, dir_tmp+'DataPrepare'+'.pkl')

        shutil.make_archive(folder+name, 'zip', dir_tmp)

        shutil.rmtree(dir_tmp)
        logger.info('Save DataPrepare')


    def load(self, name: str = 'DataPrepare_dump', folder: str = './') -> Callable:
        '''
        Load data prepare

        Parameters
        ----------
        name : str, optional
            file name, by default 'DataPrepare_dump'
        folder : str, optional
            target folder, by default './'

        Returns
        -------
        automl_alex.DataPrepare
            Loaded DataPrepare
        '''        
        
        dir_tmp = "./DataPrepare_tmp/"
        Path(dir_tmp).mkdir(parents=True, exist_ok=True)

        shutil.unpack_archive(folder+name+'.zip', dir_tmp)

        de = joblib.load(dir_tmp+'DataPrepare'+'.pkl')

        for cat_encoder_name in de.cat_encoder_names:
            de._fit_cat_encoders[cat_encoder_name] = joblib.load(dir_tmp+cat_encoder_name+'.pkl')

        shutil.rmtree(dir_tmp)
        logger.info('Load DataPrepare')
        return(de)


@logger.catch
def reduce_mem_usage(df: pd.DataFrame, verbose=0) -> pd.DataFrame:
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    if verbose > 0:
        start_mem = df.memory_usage().sum() / 1024**2
        logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if (str(col_type)[:3] == 'int') or (str(col_type)[:5] == 'float'):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                #elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                #    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #    df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    if verbose > 0:
        end_mem = df.memory_usage().sum() / 1024**2
        logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df