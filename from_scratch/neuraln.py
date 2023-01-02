import numpy as np


class Network:
    def __init__(self) -> None:
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add_layer(self, layer) -> None:
        """Add a layer to your network"""
        self.layers.append(layer)

    def set_loss(self, loss, loss_prime) -> None:
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data) -> np.ndarray[float]:
        samples = len(input_data)
        result = []

        # loop over samples
        for i in range(samples):
            # forward prop
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)

        # training loop

        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # calculate the loss
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # error on all samples
            err /= samples
            print(f"epoch{i+1} / {epochs}    error={err}")
