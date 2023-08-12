"""Experiement: test if clients can detect intrusion using a binary classifier"""
import sys

import numpy as np
import pandas as pd
from prefect import flow, task

from config import Location
from utils import chunks_generator

sys.path.append("src")

experimentLocation = Location(
    model="models/binary_classifier.py",
    predictions="data/final/predictions.py",
    results_notebook="notebooks/binary_classification_results.ipynb",
    results_template_notebook="notebooks/templates/binary_classification_template.ipynb",
)


@task
def get_data(saved_path: str):
    """Import clients data"""
    df = pd.read_csv(saved_path)
    return df.drop(["Label"], axis=1), df.Label


@task
def get_model():
    """Get Binary classifier"""
    pass


@flow
def experiment(location=experimentLocation):
    """Experiment flow"""
    X, y = get_data(location.processed_data)

    for splits in chunks_generator(X, y, 3):
        print(f"# of clients: {len(splits)}")
        for index, (split_features, split_labels) in enumerate(splits):
            print(
                f"    split #{index} shape: {split_features.shape} {split_labels.shape}"
            )
            print(f"        # of classes: {len(np.unique(split_labels))}")


if __name__ == "__main__":
    experiment()
