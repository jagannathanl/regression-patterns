import torch
import torch.nn as nn


if __name__ == "__main__":
    n_features = 2
    n_samples = 500
    lr = 1e-2
    
    n_epochs = 500

    # shape of X is (NxM)
    X = torch.rand(n_samples, n_features)
    # shape of y is (1, N)
    y = torch.rand(n_samples).unsqueeze(1)

    model = nn.Linear(n_features, 1)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for i in range(n_epochs):
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch: {i}, loss: {loss.item():.2f}")
