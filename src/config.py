"""
create Pydantic models
"""
from typing import List

from pydantic import BaseModel, validator
from scipy.stats import randint, uniform


def must_be_non_negative(v: float) -> float:
    """Check if the v is non-negative

    Parameters
    ----------
    v : float
        value

    Returns
    -------
    float
        v

    Raises
    ------
    ValueError
        Raises error when v is negative
    """
    if v < 0:
        raise ValueError(f"{v} must be non-negative")
    return v


class Location(BaseModel):
    """Specify the locations of inputs and outputs"""

    model: str = ""
    predictions: str = ""
    results_notebook: str
    labels: str = "data/processed/labels.pkl"
    processed_data: str = "data/processed/dataset.csv"
    train_data: str = "data/processed/train.pkl"
    test_data: str = "data/processed/test.pkl"
    results_template_notebook = "notebooks/templates/analyze_results.ipynb"


CentralizedLocation = Location(
    train_data="data/processed/train.pkl",
    model="models/centralized_model.pkl",
    predictions="data/final/centralized_predictions.pkl",
    results_notebook="notebooks/centralized_results.ipynb",
)

CentralizedModelParams = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.5),
    "learning_rate": uniform(0.03, 0.3),
    "max_depth": randint(2, 6),
    "n_estimators": randint(100, 150),
    "subsample": uniform(0.6, 0.4),
}

FederatedLocation = Location(
    train_data="data/processed/train.pkl",
    model="models/federated_model.pkl",
    predictions="data/final/fedrated_predictions.pkl",
    results_notebook="notebooks/fedrated_results.ipynb",
)

SplitLocation = Location(
    train_data="data/processed/train.pkl",
    model="models/split_model.pkl",
    predictions="data/final/split_predictions.pkl",
    results_notebook="notebooks/split_results.ipynb",
)


class ProcessConfig(BaseModel):
    """Specify the parameters of the `process` flow"""

    drop_columns: List[str] = ["Id"]
    label: str = "Species"
    test_size: float = 0.3

    _validated_test_size = validator("test_size", allow_reuse=True)(
        must_be_non_negative
    )


class ModelParams(BaseModel):
    C: List[float] = [0.1, 1, 10, 100, 1000]
    gamma: List[float] = [1, 0.1, 0.01, 0.001, 0.0001]
