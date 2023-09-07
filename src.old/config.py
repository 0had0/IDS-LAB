"""
create Pydantic models
"""
from typing import List

from pydantic import BaseModel, ConfigDict
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
    results_notebook: str = ""
    labels: str = "data/processed/labels.pkl"
    processed_data: str = "data/processed/dataset.csv"
    train_data: str = "data/processed/train.pkl"
    test_data: str = "data/processed/test.pkl"
    results_template_notebook = "notebooks/templates/analyze_results.ipynb"

    model_config = ConfigDict(extra="allow")


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


class FederatedLocationClass(Location):
    # Fedrated Learning Architecture Client number

    clients_number: int
    train_data: str = "data/processed/train.pkl"
    model: str = "models/federated_model.pkl"
    predictions: str = "data/final/fedrated_predictions.pkl"
    results_notebook: str = "notebooks/fedrated_results.ipynb"

    def __init__(self, clients_number=3, **data) -> None:
        """My custom init!"""
        super().__init__(clients_number=clients_number, **data)
        self.clients_number = clients_number

    def get_client(self, client_index: int = -1) -> str:
        """Federated Learning Architecture Client data getter"""
        if client_index > self.clients_number or client_index == -1:
            raise ValueError(f"Client #{client_index} doesn't Exist")
        return f"data/processed/federated_clients/client_{client_index}.pkl"


FederatedLocation = FederatedLocationClass(clients_number=3)

SplitLocation = Location(
    train_data="data/processed/train.pkl",
    model="models/split_model.pkl",
    predictions="data/final/split_predictions.pkl",
    results_notebook="notebooks/split_results.ipynb",
)


class ModelParams(BaseModel):
    C: List[float] = [0.1, 1, 10, 100, 1000]
    gamma: List[float] = [1, 0.1, 0.01, 0.001, 0.0001]
