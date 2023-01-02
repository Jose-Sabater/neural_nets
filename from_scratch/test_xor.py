import numpy as np

from neuraln import Network
from layers import FCLayer, ActivationLayer
from activation_functions import tanh, tanh_prime
from loss import mse, mse_prime

# training data
x_train = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# create network
nn = Network()
nn.add_layer(FCLayer(2, 3))
nn.add_layer(ActivationLayer(tanh, tanh_prime))
nn.add_layer(FCLayer(3, 1))
nn.add_layer(ActivationLayer(tanh, tanh_prime))

# train
nn.set_loss(mse, mse_prime)
nn.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

# test
out = nn.predict(x_train)
print(out)
