import numpy as np

from tinytorch import Parameter, SGD, Adam


def test_sgd_moves_parameter_down_gradient():
    parameter = Parameter([2.0])
    parameter.grad = np.array([4.0])

    optimizer = SGD([parameter], lr=0.1)
    optimizer.step()

    np.testing.assert_allclose(parameter.data, [1.6])


def test_adam_updates_parameter():
    parameter = Parameter([1.0])
    parameter.grad = np.array([1.0])

    optimizer = Adam([parameter], lr=0.1)
    optimizer.step()

    assert parameter.data[0] < 1.0
