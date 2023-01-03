import numpy as np


def tanh(x: np.ndarray[float]) -> np.ndarray[float]:
    return np.tanh(x)


def tanh_prime(x: np.ndarray[float]) -> np.ndarray[float]:
    return 1 - np.tanh(x) ** 2


def relu(x: np.ndarray[float]) -> np.ndarray[float]:
    return np.maximum(0, x)


def relu_prime(x: np.ndarray[float]) -> np.ndarray[float]:
    return (x > 0).astype(int)


def softmax(x: np.ndarray[float]) -> np.ndarray[float]:
    e = np.exp(x)
    return e / np.sum(e, axis=1)


"""
def softmax_prime(x):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = x.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)"""


def softmax_prime(x):
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x.
    # s.shape = (1, n)
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])

    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(x)

    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = x[i] * (1 - x[i])
            else:
                jacobian_m[i, j] = -x[i] * x[j]
    return jacobian_m
