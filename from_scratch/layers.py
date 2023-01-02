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
