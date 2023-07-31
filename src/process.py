import joblib
import pandas as pd
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from config import Location


@task
def get_preprocessed_data(save_location: str):
    """Get Preprocessed Dataset"""
    return pd.read_csv(save_location)


@task
def normalize(df):
    """Normalize features values"""
    return pd.DataFrame(MinMaxScaler().fit_transform(df.values))


@task
def encode_classes(arr):
    """Encode classes"""
    le = LabelEncoder()
    le.fit(arr)
    return le.classes_, le.transform(arr)


@task
def save_train_data(data, save_location):
    """Save train data"""
    joblib.dump(data, save_location)


@task
def save_test_data(data, save_location):
    """Save test data"""
    joblib.dump(data, save_location)


@task
def save_labels(data, save_location):
    """Save Labels set"""
    joblib.dump(data, save_location)


@task
def split_out_target(df):
    """Split out Labels from DataFrame"""
    target = df["Label"]
    df = df.drop(["Label"], axis=1)
    assert "Label" not in df.columns
    return df, target


@task
def split(df, target):
    """split training and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        df, target, train_size=0.8, stratify=target, random_state=75
    )

    return {"X": X_train, "y": y_train}, {"X": X_test, "y": y_test}


@flow
def process(location: Location = Location()):
    """Process data, divide dataset into train & test"""
    data = get_preprocessed_data(location.processed_data)
    data, target = split_out_target(data)
    labels, target = encode_classes(target)
    save_labels(labels, location.labels)
    data = normalize(data)
    train, test = split(data, target)
    del data, target, labels
    save_train_data(train, location.train_data)
    save_test_data(test, location.test_data)


if __name__ == "__main__":
    process()
