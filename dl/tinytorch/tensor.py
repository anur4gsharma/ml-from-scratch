from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union
import numpy as np

ArrayLike = Union["Tensor", np.ndarray, Sequence[float], float, int]


def _ensure_array(data: ArrayLike, dtype=None) -> np.ndarray:
    if isinstance(data, Tensor):
        data = data.data
    if dtype is None:
        return np.asarray(data)
    return np.asarray(data, dtype=dtype)


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """
    Reduce a broadcasted gradient back to the original tensor shape.

    Example:
        A.shape == (4, 3)
        b.shape == (3,)

        dL/db initially has shape (4, 3), so we sum over axis 0.
    """
    grad = np.asarray(grad)

    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)

    return grad.reshape(shape)


class Tensor:
    __array_priority__ = 1000

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        *,
        _children: Tuple["Tensor", ...] = (),
        _op: str = "",
    ):
        self.data = _ensure_array(data)
        self.requires_grad = requires_grad
        self.grad: Optional[np.ndarray] = None

        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def __repr__(self) -> str:
        return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad})"

    def numpy(self) -> np.ndarray:
        return self.data.copy()

    def item(self):
        return self.data.item()

    def zero_grad(self) -> None:
        self.grad = None

    def _accumulate_grad(self, grad: np.ndarray) -> None:
        grad = np.asarray(grad)

        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad += grad

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy(), requires_grad=False)

    def backward(self, grad: Optional[ArrayLike] = None) -> None:
        if not self.requires_grad:
            raise RuntimeError("cannot call backward() on a tensor that does not require gradients")

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "grad must be supplied when backward() is called on a non-scalar tensor"
                )
            grad_array = np.ones_like(self.data, dtype=np.result_type(self.data, np.float64))
        else:
            grad_array = _ensure_array(grad, dtype=np.result_type(self.data, np.float64))
            if grad_array.shape != self.shape:
                raise ValueError(
                    f"backward gradient shape {grad_array.shape} does not match tensor shape {self.shape}"
                )

        # Build a topological ordering of the graph.
        topo = []
        visited = set()

        def build(v: "Tensor") -> None:
            if v in visited:
                return
            visited.add(v)
            for child in v._prev:
                build(child)
            topo.append(v)

        build(self)

        # Seed d(self)/d(self) = 1 and walk backward.
        self.grad = grad_array.copy()

        for node in reversed(topo):
            node._backward()

    # ---------- arithmetic ----------

    @staticmethod
    def _coerce(other: ArrayLike) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="+",
        )

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(out.grad, self.shape))

            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike) -> "Tensor":
        return self + other

    def __neg__(self) -> "Tensor":
        out = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="neg",
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate_grad(-out.grad)

        out._backward = _backward
        return out

    def __sub__(self, other: ArrayLike) -> "Tensor":
        return self + (-self._coerce(other))

    def __rsub__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) - self

    def __mul__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="*",
        )

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                grad_self = out.grad * other.data
                self._accumulate_grad(_unbroadcast(grad_self, self.shape))

            if other.requires_grad:
                grad_other = out.grad * self.data
                other._accumulate_grad(_unbroadcast(grad_other, other.shape))

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike) -> "Tensor":
        return self * other

    def __truediv__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)
        return self * (other ** -1)

    def __rtruediv__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) / self

    def __pow__(self, exponent: Union[int, float]) -> "Tensor":
        if isinstance(exponent, Tensor):
            raise TypeError("Tensor exponents are not supported; use a scalar exponent")

        out = Tensor(
            self.data ** exponent,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"pow({exponent})",
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                grad_self = out.grad * exponent * (self.data ** (exponent - 1))
                self._accumulate_grad(grad_self)

        out._backward = _backward
        return out

    # ---------- matrix operations ----------

    def __matmul__(self, other: ArrayLike) -> "Tensor":
        other = self._coerce(other)

        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("@ currently supports only 2-D tensors")

        if self.shape[1] != other.shape[0]:
            raise ValueError(
                f"matmul shape mismatch: {self.shape} @ {other.shape}"
            )

        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="@",
        )

        def _backward():
            if out.grad is None:
                return

            if self.requires_grad:
                self._accumulate_grad(out.grad @ other.data.T)

            if other.requires_grad:
                other._accumulate_grad(self.data.T @ out.grad)

        out._backward = _backward
        return out

    def __rmatmul__(self, other: ArrayLike) -> "Tensor":
        return self._coerce(other) @ self

    # ---------- elementary functions ----------

    def exp(self) -> "Tensor":
        out_data = np.exp(self.data)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="exp",
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate_grad(out.grad * out_data)

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(
            np.log(self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="log",
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate_grad(out.grad / self.data)

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        out_data = np.tanh(self.data)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="tanh",
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate_grad(out.grad * (1.0 - out_data ** 2))

        out._backward = _backward
        return out

    # ---------- reductions / shape ----------

    def sum(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> "Tensor":
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum",
        )

        if axis is None:
            axes = tuple(range(self.ndim))
        elif isinstance(axis, int):
            axes = (axis if axis >= 0 else self.ndim + axis,)
        else:
            axes = tuple(a if a >= 0 else self.ndim + a for a in axis)

        def _backward():
            if out.grad is None or not self.requires_grad:
                return

            grad = out.grad

            if not keepdims:
                for a in sorted(axes):
                    grad = np.expand_dims(grad, axis=a)

            grad = np.broadcast_to(grad, self.shape)
            self._accumulate_grad(grad)

        out._backward = _backward
        return out

    def mean(
        self,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> "Tensor":
        if axis is None:
            divisor = self.data.size
        elif isinstance(axis, int):
            divisor = self.shape[axis]
        else:
            divisor = int(np.prod([self.shape[a] for a in axis]))

        return self.sum(axis=axis, keepdims=keepdims) / divisor

    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])

        out = Tensor(
            self.data.reshape(*shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape",
        )

        def _backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate_grad(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, *axes: int) -> "Tensor":
        if not axes:
            axes = tuple(reversed(range(self.ndim)))
        else:
            axes = tuple(axes)

        out = Tensor(
            self.data.transpose(axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose",
        )

        inverse = np.argsort(axes)

        def _backward():
            if out.grad is not None and self.requires_grad:
                self._accumulate_grad(out.grad.transpose(inverse))

        out._backward = _backward
        return out
