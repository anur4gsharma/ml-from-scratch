import numpy as np

from tinytorch import CrossEntropyLoss, Linear, MLP, Tensor


def test_linear_shapes():
    layer = Linear(3, 4)
    x = Tensor(np.random.randn(5, 3))

    y = layer(x)

    assert y.shape == (5, 4)


def test_parameter_discovery():
    model = MLP(2, [4, 4], 2)
    parameters = list(model.parameters())

    # Two weights + two biases for each of the three Linear layers.
    assert len(parameters) == 6


def test_cross_entropy_gradient():
    logits = Tensor(
        [[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]],
        requires_grad=True,
    )
    target = np.array([2, 0])

    loss_fn = CrossEntropyLoss()
    loss = loss_fn(logits, target)
    loss.backward()

    assert logits.grad.shape == logits.shape
    np.testing.assert_allclose(logits.grad.sum(axis=1), 0.0, atol=1e-12)
