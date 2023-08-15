"""Experiement: test if clients can detect intrusion using a binary classifier"""
import sys
import joblib

import numpy as np
import pandas as pd
from prefect import flow, task
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

sys.path.append("src")

from config import Location
from run_notebook import run_notebook
from utils import chunks_generator
from architectures.centralized.model import CentralizedModel


experimentLocation = Location(
    model="models/binary_classifier.py",
    predictions="data/final/experiements/Binary Classification Experiment/observation.pkl",
    results_notebook="notebooks/binary_classification_results.ipynb",
    results_template_notebook="notebooks/templates/binary_classification_template.ipynb",
    test_data="data/final/experiements/Binary Classification Experiment/y_true.pkl",
)


@task
def predict(model, x):
    """Predict labels"""
    return model.predict(x)


@flow
def experiment(location=experimentLocation):
    # """Experiment flow"""
    df = pd.read_csv(location.processed_data)

    encoder = LabelEncoder()

    X, y = df.drop(["Label"], axis=1).values, encoder.fit_transform(
        df.Label.values
    )

    del df

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=75
    )

    del X

    number_classes = len(np.unique(y))

    del y

    observations = {}

    # Reduce classes to 2
    get_binary_labels = np.vectorize(lambda x: x != 0)
    y_test = get_binary_labels(y_test)

    for splits, split_name in chunks_generator(X_train, y_train, 3):
        split_observation = []
        for split_features, split_labels in splits:
            values, counts = np.unique(split_labels, return_counts=True)

            over = SMOTE()
            under = RandomUnderSampler(sampling_strategy=0.5)
            _X, _y = under.fit_resample(
                split_features, get_binary_labels(split_labels)
            )

            _X, _y = over.fit_resample(_X, _y)

            clf = CentralizedModel()
            clf.train(_X, _y)

            y_pred = predict(clf, X_test)

            split_observation.append((values, counts, y_pred))
            del (
                split_features,
                split_labels,
                y_pred,
                values,
                counts,
                clf,
                _X,
                _y,
            )

        observations[split_name] = split_observation
        del split_observation, splits, split_name

    joblib.dump(observations, location.predictions)
    joblib.dump(y_test, location.test_data)

    run_notebook(
        location=location,
        params={
            "observations": location.predictions,
            "y_true": location.test_data,
            "number_classes": number_classes,
        },
    )


if __name__ == "__main__":
    experiment()
