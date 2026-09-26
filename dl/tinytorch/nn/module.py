from __future__ import annotations

from typing import Dict, Iterator, Tuple
import numpy as np

from ..tensor import Tensor


class Parameter(Tensor):
    """A Tensor that is intended to be learned by an optimizer."""

    def __init__(self, data):
        super().__init__(data, requires_grad=True)


class Module:
    def __init__(self):
        object.__setattr__(self, "_parameters", {})
        object.__setattr__(self, "_modules", {})
        self.training = True

    def __setattr__(self, name, value):
        if name in {"_parameters", "_modules", "training"}:
            object.__setattr__(self, name, value)
            return

        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value

        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self) -> Iterator[Parameter]:
        for parameter in self._parameters.values():
            yield parameter

        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        for name, parameter in self._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, parameter

        for name, module in self._modules.items():
            module_prefix = f"{prefix}.{name}" if prefix else name
            yield from module.named_parameters(module_prefix)

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._modules.values():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def state_dict(self) -> Dict[str, np.ndarray]:
        return {
            name: parameter.data.copy()
            for name, parameter in self.named_parameters()
        }

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        current = dict(self.named_parameters())

        missing = set(current) - set(state_dict)
        unexpected = set(state_dict) - set(current)

        if missing:
            raise ValueError(f"missing parameters: {sorted(missing)}")
        if unexpected:
            raise ValueError(f"unexpected parameters: {sorted(unexpected)}")

        for name, parameter in current.items():
            value = np.asarray(state_dict[name])
            if value.shape != parameter.shape:
                raise ValueError(
                    f"shape mismatch for {name}: expected {parameter.shape}, got {value.shape}"
                )
            parameter.data[...] = value
