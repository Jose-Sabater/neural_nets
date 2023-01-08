import numpy as np
from activation_functions import softmax


def mse(y_true: np.ndarray[int, float], y_pred: np.ndarray[int, float]) -> float:
    """
    Calculate the mean squared error (MSE) loss for a set of predicted and true labels.

    Parameters:
    y_true (ndarray): An array of true labels, of shape (num_examples, num_classes).
    y_pred (ndarray): An array of predicted labels, of shape (num_examples, num_classes).

    Returns:
    float: The average MSE loss across all examples.
    """

    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true: np.ndarray[int, float], y_pred: np.ndarray[int, float]) -> float:
    """
    Calculate the derivative of the mean squared error (MSE) loss function with respect to the predicted label.

    Parameters:
    y_true (float): The true label.
    y_pred (float): The predicted label.

    Returns:
    float: The derivative of the MSE loss with respect to the predicted label.
    """
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


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = y_pred.clip(min=1e-8, max=None)
    result = -np.sum((np.where(y_true == 1, np.log(y_pred), 0)))
    return result / y_true.size


def cross_entropy_prime(y_true, y_pred):
    # result = [-(y / x) for x, y in zip(y_pred, y_true)]
    # return (np.where(y_true == 1, -1 / y_pred, 0)) / y_true.size (I believe this is the correct one)
    return ((y_pred - y_true) / (y_pred * (1 - y_pred))) / y_true.shape[0]
