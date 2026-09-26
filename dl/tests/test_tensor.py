import numpy as np

from tinytorch import Tensor


def test_add():
    a = Tensor([1.0, 2.0])
    b = Tensor([3.0, 4.0])
    c = a + b

    np.testing.assert_allclose(c.data, [4.0, 6.0])
    assert c.shape == (2,)


def test_matmul():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])

    np.testing.assert_allclose(
        (a @ b).data,
        [[19.0, 22.0], [43.0, 50.0]],
    )


def test_broadcast_backward():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([10.0, 20.0], requires_grad=True)

    loss = (x + b).sum()
    loss.backward()

    np.testing.assert_allclose(x.grad, np.ones((2, 2)))
    np.testing.assert_allclose(b.grad, [2.0, 2.0])


def test_gradient_accumulation():
    x = Tensor(3.0, requires_grad=True)

    y1 = x * 2.0
    y2 = x * 5.0
    loss = y1 + y2
    loss.backward()

    np.testing.assert_allclose(x.grad, 7.0)
