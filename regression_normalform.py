import numpy as np 
import torch


def get_data(n_samples):
    #we are creating a bivariable regression to test 
    #this pattern
    X = torch.rand(n_samples, 2)
    #broadcating rule (m,) is treadted as (1, m) when doing arithmetic operations
    y = 5*X[:, 0]+3*X[:, 1] + 0.5*torch.rand(n_samples)
    y = y.unsqueeze(1)
    return X, y

def return_theta(X, y):
    Xb = torch.hstack([X, torch.ones(X.shape[0], 1)])
    print(Xb.shape)
    theta = torch.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
    print(theta.shape)
    return theta



if __name__ == "__main__":
    X, y = get_data(10)
    print(X.shape, y.shape)
    theta = return_theta(X, y)
    print(theta)
    X = torch.tensor([[0.5, 0.5, 1.0]])
    y_pred = X @ theta
    y_eval = 5*0.5 + 3*0.5
    print(f"Prediction: {y_pred}, Eval: {y_eval}")
    