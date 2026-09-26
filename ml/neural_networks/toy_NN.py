import numpy as np
import pandas as pd

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
    [0.2, 0.8],
    [0.8, 0.2],
    [0.1, 0.1],
    [0.9, 0.9],
])

y = np.array([
    [0],
    [1],
    [1],
    [0],
    [1],
    [1],
    [0],
    [0],
])

def derivative_linear(w, x, b):
    return w

def derivative_sigmoid(t):
    pass

def forward(x, w1, b1, w2, b2):
    z = w1 @ x + b1

    u = np.maximum(0, x)

    t = w2 @ u + b2

    y_pred = np.sigmoid(t)

def backward(x, w1, b1, w2, b2):
    derivative_y_t = derivative_sigmoid()
      

b1, b2 = np.zeros, np.zeros

w1, w2 = [np.ones(8), np.ones(8)], [np.ones(8), np.ones(8)]

print(w1)

forward(X, w1, b1, w2, b2)
rme_loss = ((y - y_pred) ** 2) ** 0.5

print(rme_loss)