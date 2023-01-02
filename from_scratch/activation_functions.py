import numpy as np


def tanh(x: float) -> float:
    return np.tanh(x)


def tanh_prime(x: float) -> float:
    return 1 - np.tanh(x) ** 2
