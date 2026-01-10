import torch
import torch.nn as nn
import numpy as np


if __name__ == "__main__":
    n_features = 2
    n_samples = 500
    lr = 0.5e-2
    
    # X shape: (n_samples, n_features)
    X = torch.rand(n_samples, n_features)
    # y shape : (n_samples,)
    y = 3*X[:, 0] + 5*X[:, 1] + 0.5 * torch.rand(n_samples)

    print(X.shape, y.shape)

    # Xb shape: (n_samples, n_features + 1)
    Xb = torch.hstack([X, torch.ones(n_samples, 1)])
    print(Xb.shape)

    # Ab shape: (n_features + 1,)
    Ab = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
    print(Ab.shape)

    print(Ab)

