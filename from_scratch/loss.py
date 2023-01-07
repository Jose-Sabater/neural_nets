import numpy as np
from activation_functions import softmax


def mse(y_true: np.ndarray[int, float], y_pred: np.ndarray[float]) -> float:
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true: float, y_pred: float) -> float:
    return 2 * (y_pred - y_true) / y_true.size


# def cross_entropy_loss(y_true, y_pred):
#     loss = []
#     y_pred.clip(min=1e-8, max=None)
#     for i in range(len(y_pred[0])):
#         if y_true[i] == 1:
#             loss.append(-np.log(y_pred[0][i]))
#         else:
#             loss.append(-np.log(1 - y_pred[0][i]))
#     return np.sum(loss)


def cross_entropy_loss(y_true, y_pred):
    y_pred = y_pred.clip(min=1e-8, max=None)
    result = -np.sum((np.where(y_true == 1, np.log(y_pred), 0)))
    return result / y_true.size


# def cross_entropy_prime(y_true, y_pred):
#     print("y_pred", y_pred)
#     m = y_true.shape[0]
#     print("m", m)
#     grad = softmax(y_pred)
#     print(grad)
#     y_true = np.argmax(y_true)
#     grad[range(m), y_true] -= 1
# grad = grad / m
# return grad
# def cross_entropy_prime(y_true, y_pred):
#     # y_pred = y_pred.clip(min=1e-8, max=None)
#     # print('\n\nCED: ', np.where(y==1,-1/X, 0))
#     print("prime", np.where(y_true == 1, -1 / y_pred, 0))
#     return np.where(y_true == 1, -1 / y_pred, 0)


def cross_entropy_prime(y_true, y_pred):
    # result = [-(y / x) for x, y in zip(y_pred, y_true)]
    return (np.where(y_true == 1, -1 / y_pred, 0)) / y_true.size
    # return result
