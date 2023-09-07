"""Testing Centralized Model on Fedrated Clients data"""
import sys
import joblib

from prefect import flow, task


sys.path.append("src")

from run_notebook import run_notebook
from config import FederatedLocationClass, Location
from train_model import train

ExperimentLocation = FederatedLocationClass(
    train_data="data/processed/train.pkl",
    model="models/centralized_model.pkl",
    predictions="data/final/centralized_model_on_federated_clients_data_predictions.pkl",
    results_notebook="notebooks/centralized_model_on_federated_clients_data_results.ipynb",
    clients_number=3,
)


@task
def get_model(save_path: str):
    """Get Model..."""
    return joblib.load(save_path)


@flow
def main(location):
    model = get_model(save_path=location.model)
    for client_id in range(location.clients_number):
        client_location = Location(
            train_data=location.get_client(client_id),
            model=f"models/centralized_client_{client_id}_model.pkl",
            predictions=f"data/final/centralized_client_{client_id}_predictions.pkl",
            results_notebook=f"notebooks/centralized_client_{client_id}_results.ipynb",
        )
        train(model=model, location=client_location, tune_model=True)
        run_notebook(location=client_location)


if __name__ == "__main__":
    main(location=ExperimentLocation)
