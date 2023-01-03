import numpy as np

from neuraln import Network
from layers import ActivationLayer, ConvolutionLayer, FCLayer, FlattenLayer
from activation_functions import (
    tanh,
    tanh_prime,
    relu,
    relu_prime,
    softmax,
    softmax_prime,
)
from loss import mse, mse_prime
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten
x_train, x_test = x_train.reshape(x_train.shape[0], 28, 28, 1), x_test.reshape(
    x_test.shape[0], 28, 28, 1
)
x_train, x_test = x_train / 255, x_test / 255
print(x_train.shape, x_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create CNN
cnn = Network()
cnn.add_layer(
    ConvolutionLayer((28, 28, 1), (3, 3), 1)
)  # input (28,28,1) output (26,26,1)
cnn.add_layer(ActivationLayer(tanh, tanh_prime))
cnn.add_layer(
    ConvolutionLayer((26, 26, 1), (3, 3), 1)
)  # input (26,26,1) output (24,24,1)
cnn.add_layer(ActivationLayer(tanh, tanh_prime))
cnn.add_layer(FlattenLayer())  # input (26,26,1) output (1,26*26*1)
cnn.add_layer(FCLayer(24 * 24 * 1, 100))  # input (1, 24*24*1) output (1,100)
cnn.add_layer(ActivationLayer(tanh, tanh_prime))
cnn.add_layer(FCLayer(100, 10))  # input (1,100) output (1,10)
cnn.add_layer(ActivationLayer(softmax, softmax_prime))

cnn.set_loss(mse, mse_prime)
cnn.fit(x_train[0:2000], y_train[0:2000], epochs=100, learning_rate=0.1)

# test some samples
out = cnn.predict(x_test[0:3])
print("Result: ")
print(out)
print("true values: ", y_test[0:3])
