import sklearn
import pandas as pd
import pytest
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from automl_alex.data_prepare import *

RANDOM_SEED = 42
TMP_FOLDER = '.automl-alex_tmp/'

@pytest.fixture
def get_data():
    dataset = fetch_openml(name='adult', version=1, as_frame=True)
    # convert target to binary
    dataset.target = dataset.target.astype('category').cat.codes
    dataset.data.head(5)
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, 
                                                        dataset.target,
                                                        test_size=0.2, 
                                                        random_state=RANDOM_SEED,)
    return(X_train, X_test)


#### Test Cleaning ######

def test_default(get_data):
    X_train, X_test = get_data

    de = DataPrepare()
    X_train = de.fit_transform(X_train)
    assert isinstance(X_train, pd.DataFrame)
    assert not X_train.empty

    c_test = de.transform(X_test)
    assert isinstance(c_test, pd.DataFrame)
    assert not c_test.empty

    de.save('test_de', folder=TMP_FOLDER)

    de = DataPrepare()
    de = de.load('test_de', folder=TMP_FOLDER)
    c_test = de.transform(X_test)
    assert isinstance(c_test, pd.DataFrame)
    assert not c_test.empty


def test_default_datasets(get_data):
    for data_id in [179,1461,31,1471,151,1067,1046,1489,1494]:
        dataset = fetch_openml(data_id=data_id, as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, 
                                                            dataset.target,
                                                            test_size=0.2, 
                                                            random_state=RANDOM_SEED,)

        de = DataPrepare(num_denoising_autoencoder=False,)
        X_train = de.fit_transform(X_train)
        assert isinstance(X_train, pd.DataFrame)
        assert not X_train.empty
        assert not (X_train.isnull().any().any())

        c_test = de.transform(X_test)
        assert isinstance(c_test, pd.DataFrame)
        assert not c_test.empty
        assert not (c_test.isnull().any().any())

        de.save('de', folder=TMP_FOLDER)
        de = None
        de_new = DataPrepare()
        de_new = de_new.load('de', folder=TMP_FOLDER)

        c_test_new = de_new.transform(X_test)
        assert isinstance(c_test_new, pd.DataFrame)
        assert not c_test_new.empty
        assert not (c_test_new.isnull().any().any())


def test_encoders(get_data):
    X_train, X_test = get_data

    for cat_encoder_name in cat_encoders_names.keys():
        X_train, X_test = get_data
        de = DataPrepare(cat_encoder_names=[cat_encoder_name,],
                        num_denoising_autoencoder=False,)
        c_train = de.fit_transform(X_train)
        assert isinstance(c_train, pd.DataFrame)
        assert not c_train.empty

        c_test = de.transform(X_test)
        assert isinstance(c_test, pd.DataFrame)
        assert not c_test.empty

def test_clean_nans():
    for method in ['median', 'mean',]:
        df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                            [3, 4, np.nan, 1],
                            [np.nan, np.nan, 7, 5],
                            [np.nan, 3, np.nan, 4]],
            columns=list('ABCD'))
        assert (df.isnull().sum().sum() > 0)

        clean_nan_encoder = CleanNans(method)
        clean_nan_encoder.fit(df)
        df_clean = clean_nan_encoder.transform(df)

        assert isinstance(df_clean, pd.DataFrame)
        assert not df_clean.empty
        assert not (df_clean.isnull().any().any())

        df_test = pd.DataFrame([[7, 2, np.nan, 0],
                            [3, np.nan, np.nan, 1],
                            [9, np.nan, 7, 5],
                            [np.nan, 3, 10, 4]],
            columns=list('ABCD'))
        df_test_clean = clean_nan_encoder.transform(df_test)

        assert isinstance(df_test_clean, pd.DataFrame)
        assert not df_test_clean.empty
        assert not (df_test_clean.isnull().any().any())

def test_denoising_autoencoder():
    for data_id in [179,1461,31,1471,151,1067,1046,1489,1494]:
        dataset = fetch_openml(data_id=data_id, as_frame=True)
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, 
                                                            dataset.target,
                                                            test_size=0.2, 
                                                            random_state=RANDOM_SEED,)

        da_encoder = DenoisingAutoencoder()
        da_encoder.fit(X_train)
        new_features = da_encoder.transform(X_train)

        assert isinstance(new_features, pd.DataFrame)
        assert not new_features.empty
        assert not (new_features.isnull().any().any())

        test_new_features = da_encoder.transform(X_test)

        assert isinstance(test_new_features, pd.DataFrame)
        assert not test_new_features.empty
        assert not (test_new_features.isnull().any().any())
