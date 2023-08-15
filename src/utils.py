"""Utils functions"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class Model:
    """Architecture Model Template"""

    def train(self, x, y):
        """training phase, overwrite to add custom logic"""
        self.model.fit(x, y)

    def predict(self, x):
        """predict"""
        return self.model.predict(x)

    def evaluate(self, y_true, y_pred):
        """Evalute output"""
        return confusion_matrix(y_true, y_pred)


def straitified_split(x: np.ndarray, y: np.ndarray, n: int):
    """
    straitified_split: split data while conserving same disctribution.
    """
    split_size = len(x) // n
    chunks = []
    x_next, y_next = x, y
    for _ in range(n - 1):
        x_chunk, x_next, y_chunk, y_next = train_test_split(
            x_next,
            y_next,
            stratify=y_next,
            train_size=split_size,
            random_state=75,
        )
        chunks.append((x_chunk, y_chunk))
    if len(x_next):
        chunks.append((x_next, y_next))

    return chunks


def random_split(x: np.ndarray, y: np.ndarray, n: int):
    """
    random_split: split data randomly to same size.
    """
    split_size = len(x) // n
    chunks = []
    x_next, y_next = x, y
    for _ in range(n - 1):
        x_chunk, x_next, y_chunk, y_next = train_test_split(
            x_next, y_next, train_size=split_size, random_state=75
        )
        chunks.append((x_chunk, y_chunk))
    if len(x_next):
        chunks.append((x_next, y_next))

    return chunks


def random_sized_split(x: np.ndarray, y: np.ndarray, n: int):
    """
    random_sized_split: split data to random sizes.
    """

    def split(_x, _train_size):
        indices = np.arange(len(_x))
        np.random.shuffle(indices)

        split_idx = int(_train_size * len(_x))

        return _x[indices[:split_idx]], _x[indices[split_idx:]]

    split_percentage = 1 / n
    chunks = []
    x_next, y_next = x, y
    for _ in range(n - 1):
        x_chunk, x_next = split(x_next, split_percentage)
        y_chunk, y_next = split(y_next, split_percentage)
        chunks.append((x_chunk, y_chunk))
    if len(x_next):
        chunks.append((x_next, y_next))

    return chunks


def chunks_generator(x, y, n):
    """Data Chunks Generator"""
    yield straitified_split(x, y, n), "Straitified Split"
    yield random_split(x, y, n), "Random Split (== size)"
    yield random_sized_split(x, y, n), "Random Split (!= size)"
