"""Federated Learning Model"""
import sys
from collections import OrderedDict
from typing import List

import flwr as fl
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from config import FederatedLocation
from utils import Model

sys.path.append("src")


DEVICE = torch.device("cpu")


def load_datasets(
    location=FederatedLocation,
):
    """Load datasets"""
    trainloaders = []
    valloaders = []

    for client_id in range(location.clients_number):
        data = joblib.load(location.get_client(client_id))
        X, y = torch.Tensor(data["X"]), torch.Tensor(data["y"].values)
        ds = TensorDataset(X, y)
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(
            ds, lengths, torch.Generator().manual_seed(42)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))

    testset = joblib.load(location.test_data)
    print(f"type of X {type(testset['X'])}, type of y {type(testset['y'])}")
    X, y = torch.Tensor(testset["X"].values), torch.Tensor(testset["y"])
    testloader = DataLoader(TensorDataset(X, y), batch_size=32)

    return trainloaders, valloaders, testloader


class Net(nn.Module):
    """Model Class"""

    def __init__(self, input_dim=78, output_units=15) -> None:
        super(Net, self).__init__()
        units = [64, 100]
        self.layer1 = nn.Linear(input_dim, units[0])
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(units[0], units[1])
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(units[1], output_units)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        return x


def get_parameters(net) -> List[np.ndarray]:
    """Get params"""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    """Set params"""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(
            f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}"
        )


class FlowerClient(fl.client.NumPyClient):
    """Fedrated Client"""

    def __init__(self, cid, net, trainloader, valloader):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        """Get params"""
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        """fit client model"""
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        """evalute"""
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data_points, labels in testloader:
            data_points, labels = data_points.to(DEVICE), torch.max(
                labels, 1
            ).to(DEVICE)
            outputs = net(data_points)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


class FedratedModel(Model):
    """Federated Architecture Model"""

    def __init__(self) -> None:
        super().__init__()
        self.trainloaders, self.valloaders, self.testloader = load_datasets()

    def client_fn(self, cid) -> FlowerClient:
        net = Net().to(DEVICE)
        trainloader = self.trainloaders[int(cid)]
        valloader = self.valloaders[int(cid)]
        return FlowerClient(cid, net, trainloader, valloader)

    def train(self):
        """train fedrated global model"""
        # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
        client_resources = None
        if DEVICE.type == "cuda":
            client_resources = {"num_gpus": 1}

        fl.simulation.start_simulation(
            client_fn=self.client_fn,
            num_clients=3,
            config=fl.server.ServerConfig(num_rounds=3),
            client_resources=client_resources,
        )


if __name__ == "__main__":
    f = FedratedModel()
    f.train()
