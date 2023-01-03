"""
Build a neural network from scratch:
1. Feed data into the network
2. Data moves through the layers, until we reach the output (forward propagation)
3. Calculate the error
4. Backward propagation. We adjust given parameters of our network.
    Usually it will be the derivative of our error in respect to the parameter
5. Iterate through that process.

The goal is to minimize the error through the iterations
"""
import numpy as np
from scipy import signal


class relu:
    def __init__(self):
        self.mask = None

    def forward_propagation(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward_propagation(self, x):
        return self.mask


# Base class
class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    # compute the output Y of a layer based on X
    def forward_propagation(self, input):
        raise NotImplementedError

    # compute dE/dX for a dE/dY and update parameters.
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError


# Fully connected layer (every input neuron connected to every output neuron)
class FCLayer(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # output for a given input
    def forward_propagation(self, input_data: np.ndarray[float]) -> np.ndarray[float]:
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # compute dE/dW, dE/dB for an error dE/dY
    def backward_propagation(
        self, output_error: float, learning_rate: float
    ) -> np.ndarray[float]:
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return input_error


class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime) -> None:
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(
        self, output_error: float, learning_rate: float
    ) -> np.ndarray[float]:
        return self.activation_prime(self.input) * output_error


class ConvolutionLayer(Layer):
    """
    A layer that uses convolution, based on math from
    https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e

    Attributes:
    - input_shape = (i,j,d)
    - kernel_shape = (m,n) size of filter
    - layer_depth = output_depth

    Methods:
    - forward_propagation
    - backward_propagation
    """

    def __init__(
        self,
        input_shape: tuple[tuple[int], tuple[int], int],
        kernel_shape: tuple[int],
        layer_depth: int,
    ) -> None:
        self.input_shape = input_shape
        self.input_depth = input_shape[2]
        self.kernel_shape = kernel_shape
        self.layer_depth = layer_depth
        self.output_shape = (
            input_shape[0] - kernel_shape[0] + 1,
            input_shape[1] - kernel_shape[1] + 1,
            layer_depth,
        )
        self.weights = (
            np.random.rand(
                kernel_shape[0], kernel_shape[1], self.input_depth, layer_depth
            )
            - 0.5
        )
        self.bias = np.random.rand(layer_depth) - 0.5

    def forward_propagation(self, input: np.ndarray[float]) -> np.ndarray[float]:
        self.input = input
        # initialize output to zero
        self.output = np.zeros(self.output_shape)
        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                # use np convolve with valid to not have the zero padding
                self.output[:, :, k] += (
                    signal.correlate2d(
                        self.input[:, :, d], self.weights[:, :, d, k], "valid"
                    )
                    + self.bias[k]
                )
        return self.output

    #
    def backward_propagation(
        self, output_error: np.ndarray[float], learning_rate: float
    ) -> np.ndarray[float]:
        """
        Calculates errors and backprop dE/dW, dE/dB, dE/dY, input error=dE/dX
        """
        in_error = np.zeros(self.input_shape)
        dWeights = np.zeros(
            (
                self.kernel_shape[0],
                self.kernel_shape[1],
                self.input_depth,
                self.layer_depth,
            )
        )
        dBias = np.zeros(self.layer_depth)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                in_error[:, :, d] += signal.convolve2d(
                    output_error[:, :, k], self.weights[:, :, d, k], "full"
                )
                dWeights[:, :, d, k] = signal.correlate2d(
                    self.input[:, :, d], output_error[:, :, k], "valid"
                )
                dBias[k] = self.layer_depth * np.sum(output_error[:, :, k])

                self.weights -= learning_rate * dWeights
                self.bias -= learning_rate * dBias
        return in_error


class FlattenLayer(Layer):
    """
    Class that returns the flattened input, necessary to combine convolutional layers and FClayers
    Methods
    - forward_propagation
    - backward_propagation
    """

    def forward_propagation(self, input_data: np.ndarray[float]) -> np.ndarray[float]:
        self.input = input_data
        self.output = input_data.flatten().reshape((1, -1))
        return self.output

    # Learning rate not used because  there are no learnable parameters
    def backward_propagation(
        self, output_error: np.ndarray[float], learning_rate: float
    ) -> np.ndarray[float]:
        return output_error.reshape(self.input.shape)
