from __future__ import annotations

from typing import Iterable, Sequence
import numpy as np

from ..tensor import Tensor
from .module import Module, Parameter


class Linear(Module):
    """
    y = x W^T + b

    Weight shape:
        (out_features, in_features)
    Bias shape:
        (out_features,)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")

        limit = np.sqrt(6.0 / (in_features + out_features))
        weight = np.random.uniform(
            -limit,
            limit,
            size=(out_features, in_features),
        )

        self.weight = Parameter(weight)

        if bias:
            self.bias = Parameter(np.zeros(out_features, dtype=np.float64))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Linear currently expects shape (batch, features), got {x.shape}"
            )

        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        out_data = np.maximum(0.0, x.data)
        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="relu",
        )

        def _backward():
            if out.grad is not None and x.requires_grad:
                local_grad = (x.data > 0).astype(out.grad.dtype)
                x._accumulate_grad(out.grad * local_grad)

        out._backward = _backward
        return out


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        positive = x.data >= 0
        neg_x = np.exp(x.data[~positive])
        pos_x = np.exp(-x.data[positive])

        result = np.empty_like(x.data, dtype=np.result_type(x.data, np.float64))
        result[positive] = 1.0 / (1.0 + pos_x)
        result[~positive] = neg_x / (1.0 + neg_x)

        out = Tensor(
            result,
            requires_grad=x.requires_grad,
            _children=(x,),
            _op="sigmoid",
        )

        def _backward():
            if out.grad is not None and x.requires_grad:
                x._accumulate_grad(out.grad * result * (1.0 - result))

        out._backward = _backward
        return out


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sequential(Module):
    def __init__(self, *modules: Module):
        super().__init__()

        for index, module in enumerate(modules):
            self._modules[str(index)] = module
            object.__setattr__(self, str(index), module)

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            x = module(x)
        return x


class MLP(Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: Module | None = None,
    ):
        super().__init__()

        if activation is None:
            activation_factory = ReLU
        else:
            activation_factory = lambda: activation

        dimensions = [input_dim, *hidden_dims, output_dim]
        layers = []

        for i in range(len(dimensions) - 1):
            layers.append(Linear(dimensions[i], dimensions[i + 1]))

            # No activation after the final linear layer.
            if i < len(dimensions) - 2:
                layers.append(activation_factory())

        self.network = Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
