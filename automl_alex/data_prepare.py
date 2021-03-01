import pandas as pd
import numpy as np
import random
from itertools import combinations
import joblib
import sys
import gc
from pathlib import Path
import shutil

from .encoders import *
from .logger import *
from sklearn.preprocessing import StandardScaler

# disable chained assignments
#pd.options.mode.chained_assignment = None 

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)



class CleanNans(object):
    """
    Сlass for cleaning Nans
    """

    def __init__(self, method='median', verbose=0):
        """
        Fill Nans and add column, that there were nans in this column
        
        Args:
            method : ['median', 'mean',]
        """
        self.method = method
        self.verbose = verbose


    @logger.catch
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

        return self


    @logger.catch
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


    @logger.catch
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


class NumericInteractionFeatures(object):
    """
    Сlass for  Numerical interaction generator features: A/B, A*B, A-B,
    """
    cols_combinations = None


    def __init__(self, operations=['/','*','-','+'], verbose=0):
        """
        Fill Nans and add column, that there were nans in this column
        
        Args:
            method : ['median', 'mean',]
        """
        self.operations = operations
        self.verbose = verbose
        self.columns = None


    def fit(self, columns,):
        """
        Fit.

        Args:
            columns (list): num columns names
        Returns:
            self
        """
        self.columns = columns
        self.cols_combinations = list(combinations(columns,2))
        return self


    def transform(self, df) -> pd.DataFrame:
        """Transforms the dataset.
        Args:
            df (pd.DataFrame, shape (n_samples, n_features)): the input data
            cols list() features: the input data
        Returns:
            pandas.Dataframe of shape = (n_train, n_features)
                Dataset with new features.
        """
        if self.cols_combinations is None:
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

        for c in self.cols_combinations:
            if '*' in self.operations:
                fe_df['{}_*_{}'.format(c[0], c[1]) ] = df[c[0]] * df[c[1]]
            if '+' in self.operations:
                fe_df['{}_+_{}'.format(c[0], c[1]) ] = df[c[0]] + df[c[1]]
        return(fe_df)


    def fit_transform(self, data, cols) -> pd.DataFrame:
        """Fit and transforms the dataset.
        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
            cols list() features: the input data
        Returns:
            pandas.Dataframe of shape = (n_train, n_features)
        """
        self.fit(cols)

        return self.transform(data)


class CleanOutliers(object):
    """
    Сlass for detect and remove outliers from your data. 
    I would like to provide two methods solution based on "z score" and solution based on "IQR".

    Something important when dealing with outliers is that one should try to use estimators as robust as possible. 
    try different values threshold and method
    """
    weight = {}

    def __init__(self, method='IQR', threshold=2, verbose=0):
        """
        Fill Nans and add column, that there were nans in this column
        
        Args:
            method : ['IQR', 'z_score',]
        """
        self.method = method
        self.threshold = threshold
        self.verbose = verbose


    def IQR(self, data, col, threshold=1.5):
        '''
        outlier detection by Interquartile Ranges Rule, also known as Tukey's test. 
        calculate the IQR ( 75th quantile - 25th quantile) 
        and the 25th 75th quantile. 
        Any value beyond:
            upper bound = 75th quantile + （IQR * threshold）
            lower bound = 25th quantile - （IQR * threshold）   
        are regarded as outliers. Default threshold is 1.5.
        '''
        
        quantile1,quantile3 = np.percentile(data[col],[25,75])
        iqr_val = quantile3 - quantile1
        lower_bound = quantile1 - (threshold*iqr_val)
        upper_bound = quantile3 + (threshold*iqr_val)
        return(lower_bound, upper_bound)


    def fit_z_score(self, data, col,):
        '''
        Z score is an important measurement or score that tells how many Standard deviation above or below a number is from the mean of the dataset
        Any positive Z score means the no. of standard deviation above the mean and a negative score means no. of standard deviation below the mean
        Z score is calculate by subtracting each value with the mean of data and dividing it by standard deviation
        '''
        median_y = data[col].median()
        median_absolute_deviation_y = (np.abs(data[col]-median_y)).median()
        return(median_y, median_absolute_deviation_y)


    def get_z_score(self, median, mad, data, col, threshold=3):
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


    @logger.catch
    def fit(self, data, cols=None):
        """
        Fit CleanOutliers.

        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
            cols list() features: the input data
        Returns:
            self
        """
        if cols is not None:
            chek_columns = cols
        else:
            chek_columns = data._get_numeric_data().columns

        self.weight = {}

        for column in chek_columns:
            if self.method == 'IQR':
                lower_bound, upper_bound = self.IQR(data,col=column,threshold=self.threshold)
                self.weight[column] = [lower_bound, upper_bound]
                #logger.info(self.weight)
                if self.verbose:
                    total_outliers = len(data[column][(data[column] < lower_bound) | (data[column] > upper_bound)])

            elif self.method == 'z_score':
                median, mad = self.fit_z_score(data, col=column)
                self.weight[column] = [median, mad]
                if self.verbose:
                    filtered_entries = self.get_z_score(median, mad, data, column, threshold=self.threshold)
                    total_outliers = filtered_entries.sum()
            else:
                raise ValueError('Wrong method')

            if self.verbose:
                if total_outliers > 0:
                    logger.info(f'Num of outlier detected: {total_outliers} in Feature {column}')
                    logger.info(f'Proportion of outlier detected: {round((100/(len(data)/total_outliers)),1)} %')

        return self


    @logger.catch
    def transform(self, data,) -> pd.DataFrame:
        """Transforms the dataset.
        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
            cols list() features: the input data
        Returns:
            pandas.Dataframe of shape = (n_train, n_features)
                The dataset.
        """
        #logger.info(self.weight)
        for weight_values in self.weight:
            if self.method == 'IQR':
                data.loc[data[weight_values] < self.weight[weight_values][0], weight_values] = self.weight[weight_values][0]
                data.loc[data[weight_values] > self.weight[weight_values][1], weight_values] = self.weight[weight_values][1]

                feature_name = weight_values+'_Is_Outliers_'+self.method
                data[feature_name] = 0
                data.loc[
                    (data[weight_values] < self.weight[weight_values][0]) | (data[weight_values] > self.weight[weight_values][1]), \
                    feature_name
                    ] = 1

            elif self.method == 'z_score':
                filtered_entries = self.get_z_score(
                    self.weight[weight_values][0], 
                    self.weight[weight_values][1], 
                    data, 
                    weight_values, 
                    threshold=self.threshold
                    )
                data.loc[filtered_entries, weight_values] = data[weight_values].median()

                feature_name = weight_values+'_Is_Outliers_'+self.method
                data[feature_name] = 0
                data.loc[filtered_entries, feature_name] = 1

        return data


    def fit_transform(self, data, cols=None) -> pd.DataFrame:
        """Fit and transforms the dataset.
        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
            cols list() features: the input data
        Returns:
            pandas.Dataframe of shape = (n_train, n_features)
                The dataset.
        """
        self.fit(data, cols)

        return self.transform(data)


class DataPrepare(object):
    """
    Сlass for cleaning, encoding and processing your dataset
    """
    clean_outliers_enc = None
    binary_encoder = None
    clean_nan_encoder = None
    cat_clean_ord_encoder = None
    fit_cat_encoders={}


    def __init__(self, 
                cat_features=None,
                clean_and_encod_data=True,
                cat_encoder_names=['HelmertEncoder','CountEncoder'],
                clean_nan=True,
                clean_outliers=True,
                outliers_method='IQR', # method : ['IQR', 'z_score',]
                outliers_threshold=2,
                drop_invariant=True,
                num_generator_features=True,
                operations_num_generator=['/','*','-',],
                #group_generator_features=False,
                #frequency_enc_num_features=False,
                normalization=True,
                reduce_memory=False,
                random_state=42,
                verbose=3):
        """
        Description of __init__

        Args:
            cat_features=None (list or None): 
            clean_and_encod_data=True (undefined):
            cat_encoder_names=None (list or None):
            clean_nan=True (undefined):
            drop_invariant=True (bool): boolean for whether or not to drop columns with 0 variance.
            num_generator_features=True (undefined):
            random_state=42 (undefined):
        """
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
        self._operations_num_generator = operations_num_generator
        self._normalization = normalization
        self._reduce_memory = reduce_memory
        self.cat_features = cat_features


    def check_data_format(self, data):
        """
        Description of check_data_format:
            Check that data is not pd.DataFrame or empty

        Args:
            data (pd.DataFrame, shape (n_samples, n_features)): the input data
        Return:
            True or Exception
        """
        #if (not isinstance(data, pd.DataFrame)) or data.empty:
        #    raise Exception("data is not pd.DataFrame or empty")


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


    @logger.catch
    def fit_transform(self, data, verbose=None):
        """
        Fit and transforms the dataset.

        Args:
            data (pd.DataFrame, shape = (n_samples, n_features)): 
                the input data
        Returns:
            data (pd.Dataframe, shape = (n_train, n_features)):
                The dataset with clean numerical and encoded categorical features.
        """
        if verbose is not None:
            self.verbose =  verbose
        logger_print_lvl(self.verbose)
        ########### check_data_format ######################
        self.check_data_format(data)

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

            self.binary_encoder = OrdinalEncoder()
            self.binary_encoder = self.binary_encoder.fit(data[self.binary_features])
            data[self.binary_features] = self.binary_encoder.transform(data[self.binary_features]).replace(2,0)
            

        ########### Categorical Features ######################
        #if self.cat_features is not None:
        # Clean Categorical Features
        if self.object_features is not None:
            logger.info('> Clean Categorical Features')
            self.cat_clean_ord_encoder = OrdinalEncoder()
            self.cat_clean_ord_encoder = self.cat_clean_ord_encoder.fit(data[self.object_features])
            data[self.object_features] = self.cat_clean_ord_encoder.transform(data[self.object_features])


        if self.cat_features is not None:
            # Encode Categorical Features
            logger.info('> Transform Categorical Features.')

            for cat_encoder_name in self.cat_encoder_names:

                if cat_encoder_name not in cat_encoders_names.keys():
                    raise Exception(f"{cat_encoder_name} not support!")

                self.fit_cat_encoders[cat_encoder_name] = cat_encoders_names[cat_encoder_name](cols=self.cat_features, drop_invariant=True)
                if cat_encoder_name == 'HashingEncoder':
                    self.fit_cat_encoders[cat_encoder_name] = cat_encoders_names[cat_encoder_name](
                            n_components=int(np.log(len(data.columns))*1000), 
                            drop_invariant=True)
                
                self.fit_cat_encoders[cat_encoder_name] = \
                    self.fit_cat_encoders[cat_encoder_name].fit(data[self.cat_features])

                data_encodet = self.fit_cat_encoders[cat_encoder_name].transform(data[self.cat_features])
                data_encodet = data_encodet.add_prefix(cat_encoder_name + '_')
                if self._reduce_memory: 
                    data_encodet = reduce_mem_usage(data_encodet)
                logger.info(f' - Encoder: {cat_encoder_name} ADD features: {len(data_encodet.columns)}')
                data = data.join(data_encodet.reset_index(drop=True))

        ########### Numerical Features ######################
        # CleanOutliers
        if self._clean_outliers:
            logger.info('> CleanOutliers',)
            self.clean_outliers_enc = CleanOutliers(
                threshold=self._outliers_threshold, 
                method=self._outliers_method,
                verbose=self.verbose)
            self.clean_outliers_enc = self.clean_outliers_enc.fit(data, cols=self.num_features)
            data = self.clean_outliers_enc.transform(data)

        # CleanNans
        if self._clean_nan:
            if self.check_num_nans(data):
                self.clean_nan_encoder = CleanNans(verbose=self.verbose)
                self.clean_nan_encoder = self.clean_nan_encoder.fit(data[self.num_features])
                data = self.clean_nan_encoder.transform(data)
                logger.info(f'> CleanNans, total nans columns: {len(self.clean_nan_encoder.nan_columns)}')
            else:
                logger.info('  No nans features')

        # Generator interaction Num Features
        if self._num_generator_features:
            if len(self.num_features) > 1:
                logger.info('> Generate interaction Num Features')
                self.num_generator = NumericInteractionFeatures(operations=self._operations_num_generator,)
                self.num_generator = self.num_generator.fit(list(self.num_features))
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
        return data


    @logger.catch
    def transform(self, data, verbose=None) -> pd.DataFrame:
        """Transform dataset.
        Args:
            data (pd.DataFrame, shape = (n_samples, n_features)): 
                the input data
        Returns:
            data (pd.Dataframe, shape = (n_train, n_features)):
                The dataset with clean numerical and encoded categorical features.
        """
        if verbose is not None:
            self.verbose = verbose

        logger_print_lvl(self.verbose)

        ########### check_data_format ######################
        self.check_data_format(data)

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
        
        if self.binary_encoder:
            data[self.binary_features] = self.binary_encoder.transform(data[self.binary_features]).replace(2,0)
            logger.info('> Clean Binary Features')

        ########### Categorical Features ######################
        # Clean Categorical Features
        if self.object_features is not None:
            logger.info('> Clean Categorical Features')
            data[self.object_features] = self.cat_clean_ord_encoder.transform(data[self.object_features])
        
        if self.cat_features is not None:
            # Encode Categorical Features
            logger.info('> Transform Categorical Features.')
            for cat_encoder_name in self.cat_encoder_names:
                data_encodet = self.fit_cat_encoders[cat_encoder_name].transform(data[self.cat_features])
                data_encodet = data_encodet.add_prefix(cat_encoder_name + '_')
                if self._reduce_memory:
                    data_encodet = reduce_mem_usage(data_encodet)
                if self.verbose > 0:
                    logger.info(f' - Encoder: {cat_encoder_name} ADD features: {len(data_encodet.columns)}')
                data = data.join(data_encodet.reset_index(drop=True))

        ########### Numerical Features ######################
        # CleanOutliers
        if self._clean_outliers:
            data = self.clean_outliers_enc.transform(data)

        # CleanNans
        if self.clean_nan_encoder:
            data = self.clean_nan_encoder.transform(data)
            logger.info('> Clean Nans')

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


    @logger.catch
    def save(self, name='DataPrepare_dump', folder='./'):
        dir_tmp = "./DataPrepare_tmp/"
        Path(dir_tmp).mkdir(parents=True, exist_ok=True)
        for cat_encoder_name in self.cat_encoder_names:
            joblib.dump(self.fit_cat_encoders[cat_encoder_name], \
                dir_tmp+cat_encoder_name+'.pkl')

        joblib.dump(self, dir_tmp+'DataPrepare'+'.pkl')

        shutil.make_archive(folder+name, 'zip', dir_tmp)

        shutil.rmtree(dir_tmp)
        logger.info('Save DataPrepare')


    @logger.catch
    def load(self, name='DataPrepare_dump', folder='./'):
        dir_tmp = "./DataPrepare_tmp/"
        Path(dir_tmp).mkdir(parents=True, exist_ok=True)

        shutil.unpack_archive(folder+name+'.zip', dir_tmp)

        de = joblib.load(dir_tmp+'DataPrepare'+'.pkl')

        for cat_encoder_name in de.cat_encoder_names:
            de.fit_cat_encoders[cat_encoder_name] = joblib.load(dir_tmp+cat_encoder_name+'.pkl')

        shutil.rmtree(dir_tmp)
        logger.info('Load DataPrepare')
        return(de)


@logger.catch
def reduce_mem_usage(df, verbose=0):
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