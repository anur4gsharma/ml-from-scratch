import numpy as np

def predict(X, w, b):
    return X @ w + b

def compute_loss(X, y, w, b):
    residuals = y - predict(X, w, b)
    squared_error = residuals ** 2
    mse = np.mean(squared_error)

    dict = {
        "residuals" : residuals,
        "sq_error": squared_error,
        "mse": mse
    }

    return dict

def compute_gradients(X, y, w, b):
    error = predict(X, w, b) - y

    dw = (2 / len(y)) * (X.T @ error)
    db = (2 / len(y)) * np.sum(error)

    return dw, db

def train(X, y, w, lr, epochs, b):
    loss_history = []

    for epoch in range(epochs):
        loss = compute_loss(X, y, w, b)["mse"]
        loss_history.append(loss)
        dw, db = compute_gradients(X, y, w, b)
        w = w - lr * dw
        b = b - lr * db

    return w, loss_history, b

X = np.array([
    [1, 2],
    [2, 3],
    [3, 4]
])

w = np.array([2, 1]).astype(float)

y = np.array([5, 7, 9])

initial_w = np.array([2.0, 1.0])

final_w, loss_history, bias = train(
    X,
    y,
    initial_w,
    lr=0.01,
    epochs=10000,
    b = 0
)

print("Final w:", final_w)
print("Final loss:", loss_history[-1])
print("Final b:", bias)
print("Predictions:", predict(X, final_w, bias))
print("Actual:", y)

import matplotlib.pyplot as plt

plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Gradient Descent Convergence")
plt.show()