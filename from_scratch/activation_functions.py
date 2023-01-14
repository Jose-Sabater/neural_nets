"""
This module contains activation functions to be used for forward and backward propagation
"""
import numpy as np


def tanh(x: np.array) -> np.array:
    """
    Applies the hyperbolic tangent function element-wise to the input array.
    Args:
        x (np.array): The input array.
    Returns:
        np.array: The output array with the hyperbolic tangent applied element-wise.
    """
    return np.tanh(x)


def tanh_prime(x: np.array) -> np.array:
    """
    Calculates the derivative of the hyperbolic tangent function element-wise for the input array.
    Args:
        x (np.array): The input array.
    Returns:
        np.array: The output array with the derivative of the hyperbolic tangent applied element-wise.
    """
    return 1 - np.tanh(x) ** 2


def relu(x: np.array) -> np.array:
    """
    Applies the rectified linear unit (ReLU) function element-wise to the input array.
    Args:
        x (np.array): The input array.
    Returns:
        np.array: The output array with the ReLU applied element-wise.
    """
    return np.maximum(0, x)


def relu_prime(x: np.array) -> np.array:
    """
    Calculates the derivative of the rectified linear unit (ReLU) function element-wise for the input array.
    Args:
        x (np.array): The input array.
    Returns:
        np.array: The output array with the derivative of the ReLU applied element-wise.
    """
    return (x > 0).astype(int)


def softmax(X: np.array) -> np.array:
    """
    Applies the softmax function to the input array.
    Args:
        X (np.array): The input array.
    Returns:
        np.array: The output array with the softmax applied.
    """
    # Get the shape of the input array
    shape = X.shape
    # Check the shape of the array
    if len(shape) == 2:
        axis = 1
    else:
        axis = -1

    exps = np.exp(X - np.max(X, axis=axis))
    return exps / np.sum(exps, axis=axis)


def softmax_prime(x: np.array) -> np.array:
    """
    Returns the derivative of the softmax. Filters depending on if the input is vectorized or not
    """
    if x.shape[0] == 1:
        # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
        s = softmax(x)
        s = s.reshape(-1, 1)
        # Use the diagonal to return the values in the shape we need them for backprop
        return np.diagonal(np.diagflat(s) - np.dot(s, s.T))

    else:
        # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
        # input s is softmax value of the original input x.
        # s.shape = (1, n)
        # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])

        s = softmax(x)
        # initialize the 2-D jacobian matrix.
        jacobian_m = np.diag(s)
        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = s[i] * (1 - s[i])
                else:
                    jacobian_m[i, j] = -s[i] * s[j]
        return jacobian_m


# Softmax alternative
# def softmax_prime(pred):
#     s = softmax(pred)
#     return s * (1 - (1 * s).sum(axis=1)[:, None])
