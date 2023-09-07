"""Prepare Clients for Federated Learning Server"""
import sys
import joblib
import numpy as np
from typing import Dict, List
from pandas import DataFrame, Series
from prefect import flow, task

sys.path.append("src")

from config import FederatedLocation


@task
def get_train_data(saved_path: str):
    """Get Train Data"""
    return joblib.load(saved_path)


def split_arr(arr: List[int] | Series, n: int):
    return np.array_split(arr, n)


@task
def stratified_split_data(data: Dict[str, any], clients_number: int):
    """Split train data into clients sets"""
    _data, _target = DataFrame(data["X"]), Series(data["y"].argmax(1))
    data_indexes_map: Dict[str, List[any]] = {
        f"client_{index}": [] for index in range(clients_number)
    }

    for label in _target.unique():
        related_records_indexes = list(
            _data[_data.index.isin(_target[_target == label].index)].index
        )
        splits = split_arr(
            related_records_indexes,
            clients_number,
        )
        assert len(splits) == clients_number
        for index, _indexes in enumerate(splits):
            key = f"client_{index}"
            data_indexes_map[key] = np.concatenate(
                (data_indexes_map[key], _indexes)
            )

    assert len(data_indexes_map.keys()) == clients_number

    del _target

    labels = DataFrame(data["y"])

    del data

    return list(
        map(
            lambda d: {"X": np.array(d[0]), "y": d[1]},
            [
                (
                    _data[_data.index.isin(idx)],
                    labels[labels.index.isin(idx)],
                )
                for idx in data_indexes_map.values()
            ],
        )
    )


@task
def save_clients_data(splits, get_path):
    """Save clients splits to storage"""
    for client_index, split in enumerate(splits):
        joblib.dump(split, get_path(client_index))


@flow
def prepare_clients(location=FederatedLocation):
    """Prepare Federated Learning Clients"""
    data = get_train_data(saved_path=location.train_data)
    splits = stratified_split_data(
        data=data, clients_number=location.clients_number
    )
    del data
    save_clients_data(splits=splits, get_path=location.get_client)


if __name__ == "__main__":
    # TODO: handle custom splits options
    split_option = sys.argv[1]  # STRATIFIED | RANDOM | ...
    prepare_clients(location=FederatedLocation)
