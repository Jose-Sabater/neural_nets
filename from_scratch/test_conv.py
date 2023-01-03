import numpy as np

from neuraln import Network
from layers import ActivationLayer, ConvolutionLayer
from activation_functions import tanh, tanh_prime, relu, relu_prime
from loss import mse, mse_prime

x_train = [np.random.rand(10, 10, 1)]
y_train = [np.random.rand(4, 4, 2)]

# Create CNN
cnn = Network()
cnn.add_layer(ConvolutionLayer((10, 10, 1), (3, 3), 1))
cnn.add_layer(ActivationLayer(tanh, tanh_prime))
cnn.add_layer(ConvolutionLayer((8, 8, 1), (3, 3), 1))
cnn.add_layer(ActivationLayer(tanh, tanh_prime))
cnn.add_layer(ConvolutionLayer((6, 6, 1), (3, 3), 2))
cnn.add_layer(ActivationLayer(tanh, tanh_prime))

cnn.set_loss(mse, mse_prime)
cnn.fit(x_train, y_train, epochs=500, learning_rate=0.3)

out = cnn.predict(x_train)
print("predicted result = ", out)
print("expected result = ", y_train)
