import numpy as np

from neuraln import Network
from layers import FCLayer, ActivationLayer
from activation_functions import tanh, tanh_prime, relu, relu_prime
from loss import mse, mse_prime

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

np.random.seed(0)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Flatten
x_train, x_test = x_train.reshape(-1, 1, 28 * 28), x_test.reshape(-1, 1, 28 * 28)
x_train, x_test = x_train / 255, x_test / 255
print(x_train.shape, x_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

nn = Network()
nn.add_layer(FCLayer(28 * 28, 100))  # input (1, 28*28) output (100)
nn.add_layer(ActivationLayer(tanh, tanh_prime))
nn.add_layer(FCLayer(100, 50))  # input (1, 100) output (1,50)
nn.add_layer(ActivationLayer(tanh, tanh_prime))
nn.add_layer(FCLayer(50, 10))  # input (1, 50) output (1,10)
nn.add_layer(ActivationLayer(tanh, tanh_prime))

nn.set_loss(mse, mse_prime)
nn.fit(x_train[0:2000], y_train[0:2000], epochs=40, learning_rate=0.1)

# test some samples
out = nn.predict(x_test[0:3])
print("Result: ")
print(out)
print("true values: ", y_test[0:3])
