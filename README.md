# Regression Coding Patterns
## Introduction
We will explore several patterns for implementing a simple linear regression model. As an example, consider the target function

$$ 
\begin{aligned}
y = 3 \cdot x_1 + 2 \cdot x_2
\end{aligned}
$$


We will first solve this model from scratch using the Normal Equation, then re‑implement it using gradient descent, and finally reproduce the same workflow using PyTorch’s built‑in optimizers, loss functions, and autograd for a more idiomatic approach.

## Theory

To express the model in vectorized form, we augment the input with a bias term and define



$$ 
\begin{aligned}
X = (x_1, x_2, 1.0),
\end{aligned}
$$

and the learnable parameter vector


$$ 
\begin{aligned}
W = (w_1, w_2, w_3),
\end{aligned}
$$


where $w_3$ represents the bias.
The linear model can then be written compactly as

$$ 
\begin{aligned}
\hat{y} =  X \cdot W
\end{aligned}
$$


where $\hat{y}$ denotes the predicted value of the target. 

If $L$ as the loss and $\nabla L$ is the gradient with respect to W, then for a single training example:


$$ 
\begin{aligned}
L = (\hat{y} - X \cdot W)^2  \\
\nabla L = \frac{\partial L}{\partial W} = -2 \cdot X^T \cdot (\hat{y} - X \cdot W) \\
\end{aligned}
$$


Setting the gradient to zero gives

$$
\begin{aligned}
X^T \cdot \hat{y}  = X^T \cdot X \cdot W 
\end{aligned}
$$

which leads to the closed-form solution

$$
\begin{aligned}
W = (X^T \cdot X)^{-1}  \cdot X^T \cdot \hat{y}.            \quad (1)
\end{aligned}
$$
Equation (1) is the Normal Equation, which computes the optimal parameter vector as the product of three matrices.