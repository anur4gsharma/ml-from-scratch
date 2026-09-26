from __future__ import annotations

import numpy as np

from .tensor import Tensor
from .nn.module import Module


class MSELoss(Module):
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError(
                f"MSE shape mismatch: prediction {prediction.shape}, target {target.shape}"
            )

        difference = prediction - target
        return (difference * difference).mean()


class CrossEntropyLoss(Module):
    """
    Mean multiclass cross-entropy loss.

    Inputs:
        logits: Tensor of shape (batch, classes)
        target: integer labels of shape (batch,)

    The operation is implemented as a fused differentiable operation:
        dL/dlogits = (softmax(logits) - one_hot(target)) / batch_size

    This avoids introducing an indexing operation solely for the loss.
    """

    def forward(self, logits: Tensor, target) -> Tensor:
        if logits.ndim != 2:
            raise ValueError(
                f"CrossEntropyLoss expects logits with shape (batch, classes), got {logits.shape}"
            )

        if isinstance(target, Tensor):
            target = target.data
        target = np.asarray(target, dtype=np.int64)

        if target.shape != (logits.shape[0],):
            raise ValueError(
                f"target must have shape ({logits.shape[0]},), got {target.shape}"
            )

        if np.any(target < 0) or np.any(target >= logits.shape[1]):
            raise ValueError("target contains a class index outside the logits range")

        shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        probabilities = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)

        logsumexp = np.log(np.sum(exp_shifted, axis=1)) + np.max(
            logits.data, axis=1
        )
        losses = -logits.data[np.arange(logits.shape[0]), target] + logsumexp
        loss_value = np.mean(losses)

        out = Tensor(
            np.asarray(loss_value, dtype=np.float64),
            requires_grad=logits.requires_grad,
            _children=(logits,),
            _op="cross_entropy",
        )

        def _backward():
            if out.grad is None or not logits.requires_grad:
                return

            grad_logits = probabilities.copy()
            grad_logits[np.arange(logits.shape[0]), target] -= 1.0
            grad_logits /= logits.shape[0]

            upstream = np.asarray(out.grad).item()
            logits._accumulate_grad(grad_logits * upstream)

        out._backward = _backward
        return out
