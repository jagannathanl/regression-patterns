import torch

def generate_data(n_samples, n_features, sr=0.8):
    X = torch.rand(n_samples, n_features)
    y = 3.0 * X[:, 0] + 2.0 * X[:, 1] + 0.5 * torch.rand(n_samples)
    y = y.view(-1, 1)

    print(X.shape, y.shape, torch.rand(n_samples).shape)

    n = int(n_samples*sr)
    X_eval, y_eval = X[n:, :], y[n:]
    X, y = X[:n, :], y[:n]
    return X, X_eval, y, y_eval



if __name__ == "__main__":
    n_samples = 100
    n_features = 2

    X, X_eval, y, y_eval = generate_data(n_samples, n_features)
    
    n_epochs = 1500
    lr = 0.001

    Wx = torch.rand(n_features, 1)
    By = torch.rand(1)

    total_loss = 0.0
    for i in range(n_epochs):
        #X shape (n_samples, n_features), Wx shape (n_features, 1)
        #By shape (1), y_pred shape (n_samples, 1)
        y_pred =  X @ Wx + By

        #y shape (n_samples)
        loss = (y_pred - y).pow(2).mean()
        total_loss += loss

        #grad_y shape (n_samples, 1)
        # y_pred shape (n_samples, 1), y shape (n_samples)
        grad_y = 2*(y_pred - y)/y.shape[0]

        #shape grad_y (n_samples, 1), X shape (n_samples, n_features)
        grad_wx = grad_y.T @ X

        # grad_By shape (1)
        grad_By = grad_y.sum()

        #shape Wx (n_features, 1), grad_wx (n_features, 1)
        Wx -= lr * grad_wx.T
        By -= lr * grad_By
        if i %100 == 0:
            print("loss: ", loss)
    print("Wx: ", Wx)
    print("By: ", By)