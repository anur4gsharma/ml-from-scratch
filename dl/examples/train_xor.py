import numpy as np

from tinytorch import (
    Adam,
    CrossEntropyLoss,
    DataLoader,
    MLP,
    Tensor,
    TensorDataset,
)


def make_xor():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    y = np.array([0, 1, 1, 0])
    return X, y


X, y = make_xor()

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = MLP(input_dim=2, hidden_dims=[8, 8], output_dim=2)
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.03)

for epoch in range(1000):
    epoch_loss = 0.0

    for X_batch, y_batch in loader:
        optimizer.zero_grad()

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if epoch % 100 == 0:
        print(f"epoch={epoch:4d} loss={epoch_loss:.6f}")

predictions = []
for X_batch, _ in DataLoader(dataset, batch_size=4, shuffle=False):
    logits = model(X_batch)
    predictions.extend(np.argmax(logits.data, axis=1))

print("predictions:", predictions)
print("targets:   ", y.tolist())
