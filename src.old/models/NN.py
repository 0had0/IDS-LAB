"""Federated Learning Model"""
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


DEVICE = torch.device("cuda")


class Net(nn.Module):
    """Model Class"""

    def __init__(self, input_dim=76, num_classes=14) -> None:
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        layers = []
        layers.append(nn.Linear(input_dim, 128))

        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(128, 256))

        layers.append(nn.BatchNorm1d(256))
        layers.append(nn.Dropout(p=0.3))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(256, 256))

        layers.append(nn.BatchNorm1d(256))
        layers.append(nn.Dropout(p=0.4))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(256, 128))

        layers.append(nn.BatchNorm1d(128))
        layers.append(nn.Dropout(p=0.5))
        layers.append(nn.ReLU(True))
        layers.append(nn.Linear(128, num_classes))
        layers.append(nn.LogSoftmax())

        self.model = nn.Sequential(*layers).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def train(
    net: nn.Module, trainloader: torch.utils.data.DataLoader, epochs: int
) -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for features, labels in tqdm(trainloader, desc=f"Epock {epoch+1}: "):
            features, labels = features.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(features)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(net(features), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        print(
            f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}"
        )


def test(
    net: nn.Module, testloader: torch.utils.data.DataLoader
) -> [float, float, np.ndarray, np.ndarray]:
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss, predictions, true = 0, 0, 0.0, [], []
    net.eval()
    with torch.no_grad():
        for data_points, labels in tqdm(testloader):
            true.append(labels)
            data_points, labels = data_points.to(DEVICE), labels.to(DEVICE)
            outputs = net(data_points)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted.cpu().tolist())
            loss += criterion(outputs, labels).item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy, true, predictions
