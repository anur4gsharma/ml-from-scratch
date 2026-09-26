import numpy as np

from tinytorch import Tensor


def numerical_gradient(f, x, eps=1e-6):
    x = x.astype(np.float64)
    gradient = np.zeros_like(x)

    for index in np.ndindex(x.shape):
        original = x[index]

        x[index] = original + eps
        plus = f(x)

        x[index] = original - eps
        minus = f(x)

        x[index] = original
        gradient[index] = (plus - minus) / (2.0 * eps)

    return gradient


def test_scalar_chain_rule():
    x = Tensor(2.0, requires_grad=True)
    y = x * x + 3.0 * x
    y.backward()

    # dy/dx = 2x + 3 = 7 at x=2
    np.testing.assert_allclose(x.grad, 7.0)


def test_matmul_gradient():
    x_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    w_data = np.array([[2.0, -1.0], [0.5, 3.0]])

    x = Tensor(x_data, requires_grad=True)
    w = Tensor(w_data, requires_grad=True)

    loss = ((x @ w) ** 2).sum()
    loss.backward()

    def f_w(w_value):
        return float(((x_data @ w_value) ** 2).sum())

    expected = numerical_gradient(f_w, w_data.copy())

    np.testing.assert_allclose(w.grad, expected, rtol=1e-5, atol=1e-6)
