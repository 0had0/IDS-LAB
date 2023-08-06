"""Prepare Clients for Federated Learning Server"""
import joblib
from prefect import flow, task

from utils import FederatedLocation


@task
def get_train_data(saved_path: str):
    """Get Train Data"""
    return joblib.load(saved_path)


@task
def split_data(data, clients_number: int):
    """Split train data into clients sets"""
    pass


@task
def save_clients_data(splits, get_path):
    """Save clients splits to storage"""
    for client_index, split in enumerate(splits):
        joblib.dump(split, get_path(client_index))


@flow
def prepare_clients(location=FederatedLocation()):
    """Prepare Federated Learning Clients"""
    data = get_train_data(saved_path=location.train_data)
    splits = split_data(data=data, clients_number=location.clients_number)
    save_clients_data(splits=splits, get_path=location.get_client)


if __name__ == "__main___":
    prepare_clients(location=FederatedLocation())
