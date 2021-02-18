import sklearn
import pandas as pd
import pytest
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from automl_alex.data_prepare import *

RANDOM_SEED = 42


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
    de = de.fit(X_train)
    X_train = de.transform(X_train)
    assert isinstance(X_train, pd.DataFrame)
    assert not X_train.empty

    c_test = de.transform(X_test)
    assert isinstance(c_test, pd.DataFrame)
    assert not c_test.empty

    de.save('test_de')

    de = DataPrepare()
    de = de.load('test_de')
    c_test = de.transform(X_test)
    assert isinstance(c_test, pd.DataFrame)
    assert not c_test.empty


def test_encoders(get_data):
    X_train, X_test = get_data

    for cat_encoder_name in cat_encoders_names.keys():
        X_train, X_test = get_data
        de = DataPrepare(cat_encoder_names=[cat_encoder_name,])
        de = de.fit(X_train)
        c_train = de.transform(X_train)
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

        clean_nan_encoder = CleanNans(method).fit(df)
        df_clean = clean_nan_encoder.transform(df)

        assert isinstance(df_clean, pd.DataFrame)
        assert not df_clean.empty
        assert not (df_clean.isnull().any().any())
