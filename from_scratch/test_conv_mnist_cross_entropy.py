import numpy as np
import matplotlib.pyplot as plt

from neuraln import Network
from layers import ActivationLayer, ConvolutionLayer, FCLayer, FlattenLayer
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from utils import conf
from loss import cross_entropy_loss, cross_entropy_prime

# Read config file
settings = conf()

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

# Create CNN
cnn = Network()
cnn.add_layer(
    ConvolutionLayer((28, 28, 1), (3, 3), 1)
)  # input (28,28,1) output (26,26,1)
cnn.add_layer(ActivationLayer(*settings.network_activation))
cnn.add_layer(
    ConvolutionLayer((26, 26, 1), (3, 3), 1)
)  # input (26,26,1) output (24,24,1)
cnn.add_layer(ActivationLayer(*settings.network_activation))
cnn.add_layer(FlattenLayer())  # input (24,24,1) output (1,24*24*1)
cnn.add_layer(FCLayer(24 * 24 * 1, 100))  # input (1, 24*24*1) output (1,100)
cnn.add_layer(ActivationLayer(*settings.network_activation))
cnn.add_layer(FCLayer(100, 10))  # input (1,100) output (1,10)
cnn.add_layer(ActivationLayer(*settings.output_activation))

cnn.set_loss(cross_entropy_loss, cross_entropy_prime)
cnn.fit(x_train[0:5000], y_train[0:5000], settings.epochs, 0.01)

# # test some samples
# out = cnn.predict(x_test[0:3])
# print("Result: ")
# print(out)
# print("true values: ", y_test[0:3])

predictions = cnn.predict(x_test)

# Calculate the accuracy of the predictions
accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1))

# Print the accuracy
print(f"Test accuracy: {accuracy:.2f}")

# Plot the first 25 test images with their predicted and true labels
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i][:, :, 0], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = np.argmax(y_test[i])
    if predicted_label == true_label:
        color = "blue"
    else:
        color = "red"
    plt.xlabel(f"Predicted: {predicted_label}\nTrue: {true_label}", color=color)
plt.show()
