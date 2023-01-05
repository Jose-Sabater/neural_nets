import numpy as np

from neuraln import Network
from layers import ActivationLayer, ConvolutionLayer
from activation_functions import tanh, tanh_prime
from utils import conf

np.random.seed(0)

# Read config file
settings = conf()

x_train = [np.random.rand(10, 10, 1)]
y_train = [np.random.rand(4, 4, 2)]

# Create CNN
cnn = Network()
cnn.add_layer(ConvolutionLayer((10, 10, 1), (3, 3), 1))
cnn.add_layer(ActivationLayer(*settings.network_activation))
cnn.add_layer(ConvolutionLayer((8, 8, 1), (3, 3), 1))
cnn.add_layer(ActivationLayer(*settings.network_activation))
cnn.add_layer(ConvolutionLayer((6, 6, 1), (3, 3), 2))
cnn.add_layer(ActivationLayer(tanh, tanh_prime))

cnn.set_loss(*conf.loss_functions)
cnn.fit(x_train, y_train, settings.epochs, settings.learning_rate)

out = cnn.predict(x_train)
print("predicted result = ", out)
print("expected result = ", y_train)
