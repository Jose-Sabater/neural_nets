from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np


# Load,and set train and test data
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
