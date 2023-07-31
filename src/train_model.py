"""Python script to train the model"""
import sys

import joblib
import numpy as np
import pandas as pd
from prefect import flow, task

from architectures import get_architecture
from config import Location
from utils import Model


@task
def get_processed_data(train_data_location: str, test_data_location: str):
    """Get processed data from a specified location

    Parameters
    ----------
    train_data_location : str
        Location to get the train data
    test_data_location : str
        Location to get the test data
    """
    return joblib.load(train_data_location), joblib.load(test_data_location)


@task
def predict(model: Model, X_test: pd.DataFrame):
    """_summary_

    Parameters
    ----------
    model : one of the architectures Models
    X_test : pd.DataFrame
        Features for testing
    """
    return model.predict(X_test)


@task
def save_model(model: Model, save_path: str):
    """Save model to a specified location

    Parameters
    ----------
    model : one of the architectures Models
    save_path : str
    """
    joblib.dump(model, save_path)


@task
def save_predictions(predictions: np.array, save_path: str):
    """Save predictions to a specified location

    Parameters
    ----------
    predictions : np.array
    save_path : str
    """
    joblib.dump(predictions, save_path)


@flow
def train(
    model,
    location: Location = Location(),
):
    """Flow to train the model

    Parameters
    ----------
    location : Location instance,
        Locations of inputs and outputs
    model : Model instance,
        one of CentralizedModel, FedratedModel, SplitModel
    """
    train, test = get_processed_data(location.train_data, location.test_data)
    model.train(train["X"], train["y"])
    predictions = predict(model, test["X"])
    save_model(model, save_path=location.model)
    save_predictions(predictions, save_path=location.data_final)


if __name__ == "__main__":
    train(**get_architecture(sys.argv[1]))
