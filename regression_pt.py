import numpy as numpy
import torch 
import math

def get_data(n_samples):
    X = torch.rand(n_samples, 2)
    y = 3*X[:,0]+ 5*X[:,1] + 0.5*torch.rand(n_samples)
    y = y.unsqueeze(1)
    return X, y

def train(X, y, n_epochs=1000,debug=False):
    model = torch.nn.Linear(2, 1)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
    orig_forward = model.forward 

    if debug:
        def my_forward(X):
            print("X shape: ", X.shape)
            print("", model.weight)
            print("", model.bias)
            return orig_forward(X)

        model.forward = my_forward
        for i in range(2):
            print("Debug Forward Pass ", i)
            pred = model.forward(X)
            print("X: ", X[:2])
            print("y: ", y[:2])
            print("pred: ", pred[:2])
            print("Weights: ", model.weight)
            print("Bias: ", model.bias)
            print("error:", (pred - y)[:])
            error = loss_fn(pred, y)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
    else:
        model.train()
        n = int(X.shape[0]*0.8)          
        X_eval = X[n:, :]
        y_eval = y[n:, :]

        X = X[:n, :]
        y = y[:n, :]
        print("n_samples for training: ", X.shape[0], n)
        for i in range(n_epochs):
            y_pred = model.forward(X)
            error = loss_fn(y_pred, y)
            optimizer.zero_grad()
            error.backward()
            optimizer.step()
            if i % 100 == 0:
                print("Epoch: ", i, "Loss: ", error.item())
            
        model.eval()
        with torch.no_grad():
            eval_pred = model.forward(X_eval)
            squared_error = sum((eval_pred-y_eval)*(eval_pred-y_eval)/len(y_eval))
            rmse = math.sqrt(squared_error)
            print("Evaluation MSE: ", rmse, squared_error)
            print("Eval pred: ", eval_pred[:5], " y_eval: ", y_eval[:5])
            eval_error = torch.sqrt(loss_fn(eval_pred, y_eval))
            print("Evaluation RMSE: ", eval_error, rmse)

    pred = model.forward(X)
    return pred



if __name__ == "__main__":
    X, y = get_data(500)
    train(X, y, debug=False)



    
