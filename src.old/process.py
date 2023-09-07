import joblib
import pandas as pd
from prefect import flow, task
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from config import Location


# @task
def get_preprocessed_data(save_location: str):
    """Get Preprocessed Dataset"""
    return pd.read_csv(save_location)


@task
def normalize(df):
    """Normalize features values"""
    return pd.DataFrame(MinMaxScaler().fit_transform(df.values))


# @task
def encode_classes(arr):
    """Encode classes"""
    le = LabelEncoder()
    return le.fit_transform([[x] for x in arr])


# @task
def save_train_data(data, save_location):
    """Save train data"""
    joblib.dump(data, save_location)


# @task
def save_test_data(data, save_location):
    """Save test data"""
    joblib.dump(data, save_location)


@task
def save_labels(data, save_location):
    """Save Labels set"""
    joblib.dump(data, save_location)


# @task
def split_out_target(df):
    """Split out Labels from DataFrame & remove useless columns ['Flow Packets/s','Flow Bytes/s']"""
    target = df["Label"]
    df = df.drop(["Flow Packets/s", "Flow Bytes/s", "Label"], axis=1)
    assert "Label" not in df.columns
    return df, target


@task
def split(df, target):
    """split training and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        df, target, train_size=0.95, stratify=target, random_state=75
    )

    return {"X": X_train, "y": y_train}, {"X": X_test, "y": y_test}


# @task
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


# @flow
def process(location: Location = Location()):
    """Process data, divide dataset into train & test"""
    data = get_preprocessed_data(location.processed_data)
    data, target = split_out_target(data)
    data, target = drop_normals(data, target)
    data = replace_nan(data)
    assert target[target == "BENIGN"].shape[0] == 0

    target = encode_classes(target)
    X_train, X_test, y_train, y_test = train_test_split(
        data.values, target, train_size=0.95, stratify=target, random_state=75
    )
    del data, target  # , labels
    save_train_data({"X": X_train, "y": y_train}, location.train_data)
    save_test_data({"X": X_test, "y": y_test}, location.test_data)


if __name__ == "__main__":
    process()
