"""Setup Utils"""
import joblib
import pandas as pd

from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def preprocess_one_file(df):
    """Pre-Process 1 csv file"""
    df.columns = df.columns.str.lstrip()
    return df


def normalize(df):
    """Normalize features values"""
    return pd.DataFrame(MinMaxScaler().fit_transform(df.values))


def encode_classes(arr):
    """Encode classes"""
    le = LabelEncoder()
    return le.fit_transform([[x] for x in arr])


def save_train_data(data, save_location):
    """Save train data"""
    joblib.dump(data, save_location)


def save_test_data(data, save_location):
    """Save test data"""
    joblib.dump(data, save_location)


def split_out_target(df):
    """Split out Labels from DataFrame & remove useless columns ['Flow Packets/s','Flow Bytes/s']"""
    target = df["Label"]
    df = df.drop(["Flow Packets/s", "Flow Bytes/s", "Label"], axis=1)
    assert "Label" not in df.columns
    return df, target


def drop_normals(df, target):
    indices = target[target == "BENIGN"].index
    return df.drop(indices, axis=0), target.drop(indices, axis=0)


def replace_nan(df):
    """Replace NaN"""

    nan_count = df.isnull().sum().sum()

    if nan_count > 0:
        df.fillna(df.mean(), inplace=True)

    df = df.astype(float).apply(pd.to_numeric)

    assert df.isnull().sum().sum() == 0, "There should not be any NaN"

    return df
