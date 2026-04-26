# Regression Coding Patterns
## Introduction
We will explore several patterns for implementing a simple linear regression model. As an example, consider the target function

$$ 
\begin{aligned}
y = 3 \cdot x_1 + 2 \cdot x_2
\end{aligned}
$$


We will first solve this model from scratch using the Normal Equation, then re‑implement it using gradient descent, and finally reproduce the same workflow using PyTorch’s built‑in optimizers, loss functions, and autograd for a more idiomatic approach.

## Normal Form

To express the model in vectorized form, we augment the input with a bias term and define $ \begin{aligned} X = (x_1, x_2, 1.0) \end{aligned} $ and the learnable parameter vector $ \begin{aligned} W = (w_1, w_2, w_3), \end{aligned} $ where $w_3$ represents the bias. The linear model can then be written compactly as

$$ 
\begin{aligned}
\hat{y} =  X \cdot W
\end{aligned}
$$


Here, $\hat{y}$ represents the predicted value of the target. It can be confusing whether to write the loss as $(\hat{y} - y)^2$ or $(y - \hat{y})^2$, since both expressions produce the same value. However, this distinction becomes important when computing gradients, as it affects the sign of the derivative. To avoid confusion during backpropagation, it is best to consistently define the loss as $(\hat{y}  - y)^2$, since its derivative natually leads to the standard gradient descent update rule.

If $L$ as the loss and $\nabla L$ is the gradient with respect to W, then for a single training example:


$$ 
\begin{aligned}
L = (X \cdot W - y)^2  \\
\nabla L = \frac{\partial L}{\partial W} = 2 \cdot X^T \cdot (X \cdot W - y) \\
\end{aligned}
$$


Setting the gradient to zero gives

$$
\begin{aligned}
X^T \cdot y  = X^T \cdot X \cdot W 
\end{aligned}
$$

which leads to the closed-form solution

$$
\begin{aligned}
W = (X^T \cdot X)^{-1} \cdot X^T \cdot y       \quad (1)
\end{aligned}
$$

Equation (1) is the Normal Equation, which computes the optimal parameter vector as the product of three matrices.


## Gradient Method

For a single training example, The prediction is:

$$ 
\begin{aligned}
\hat{y_i} =  X_i \cdot W + b
\end{aligned}
$$

where $X = (x_1, x_2 )$ is defined without the bias term.  Then the loss term is define as follows:

$$ 
\begin{aligned}
L = {\frac {1}{N}} . \sum_{i=1}^{N}  ( X_i \cdot W + b - y_i)^2   \quad (2)
\end{aligned}
$$

where $N$ is the number of samples. And the gradients with respect to W and b $\nabla W$ and $\nabla b$ respectively can be written as follows:


$$ 
\begin{aligned}
\nabla W =  \frac {2}{N}.X^T.( X \cdot W + b - y)      \quad (3)  \\
\nabla b = \frac {2}{N}.\sum_{i=1}^{N}  ( X_i \cdot W + b - y_i)       \quad (4)
\end{aligned}
$$

In equation (4) above, it is worth noting the summation sign which is often times confusing. The gradient with respect to $b$ does not involve X in equation (4) as it did in equation (3). The derivative of $b$ term vanishes to a unit vector. Therefore, the gradient reduces to the sum of residuals in equation (4).  Now the gradient descent update becomes:

$$
\begin{aligned}
W =  W - \eta . \nabla W \\
b = b - \eta . \nabla b 
\end{aligned}
$$

where $\eta$ is the learning rate of the gradient descent.
