from __future__ import annotations

from typing import Iterable
import numpy as np

from .nn.module import Parameter


class Optimizer:
    def __init__(self, parameters: Iterable[Parameter]):
        self.parameters = list(parameters)

        for parameter in self.parameters:
            if not isinstance(parameter, Parameter):
                raise TypeError("Optimizer expects Parameter objects")

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-3,
        momentum: float = 0.0,
    ):
        super().__init__(parameters)

        if lr <= 0:
            raise ValueError("lr must be positive")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must satisfy 0 <= momentum < 1")

        self.lr = lr
        self.momentum = momentum
        self._velocity = {
            id(parameter): np.zeros_like(parameter.data, dtype=np.float64)
            for parameter in self.parameters
        }

    def step(self) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue

            gradient = parameter.grad

            if self.momentum:
                velocity = self._velocity[id(parameter)]
                velocity *= self.momentum
                velocity += gradient
                update = velocity
            else:
                update = gradient

            parameter.data[...] -= self.lr * update


class Adam(Optimizer):
    def __init__(
        self,
        parameters: Iterable[Parameter],
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(parameters)

        if lr <= 0:
            raise ValueError("lr must be positive")

        beta1, beta2 = betas
        if not (0 <= beta1 < 1 and 0 <= beta2 < 1):
            raise ValueError("betas must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._step_count = 0

        self._m = {
            id(parameter): np.zeros_like(parameter.data, dtype=np.float64)
            for parameter in self.parameters
        }
        self._v = {
            id(parameter): np.zeros_like(parameter.data, dtype=np.float64)
            for parameter in self.parameters
        }

    def step(self) -> None:
        self._step_count += 1

        for parameter in self.parameters:
            if parameter.grad is None:
                continue

            key = id(parameter)
            gradient = parameter.grad

            m = self._m[key]
            v = self._v[key]

            m *= self.beta1
            m += (1.0 - self.beta1) * gradient

            v *= self.beta2
            v += (1.0 - self.beta2) * (gradient ** 2)

            m_hat = m / (1.0 - self.beta1 ** self._step_count)
            v_hat = v / (1.0 - self.beta2 ** self._step_count)

            parameter.data[...] -= self.lr * m_hat / (
                np.sqrt(v_hat) + self.eps
            )
