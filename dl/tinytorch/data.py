from __future__ import annotations

from typing import Iterator, Tuple
import numpy as np

from .tensor import Tensor


class TensorDataset:
    def __init__(self, X, y):
        X = X.data if isinstance(X, Tensor) else np.asarray(X)
        y = y.data if isinstance(y, Tensor) else np.asarray(y)

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")

        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class DataLoader:
    def __init__(
        self,
        dataset: TensorDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        indices = np.arange(len(self.dataset))

        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]

            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            X_batch = self.dataset.X[batch_indices]
            y_batch = self.dataset.y[batch_indices]

            yield Tensor(X_batch), Tensor(y_batch)

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
