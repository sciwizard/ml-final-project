import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from svm import svm_classifier

# Train/test data is stored in a tuple of (data, labels)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Scale data to range [0, 1)
x_train = x_train / 255.0
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

print(x_train.shape)

# Translations from numeric label (index in 'labels' array) to English item name
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


svm_classifier((x_train, y_train), (x_test, y_test), labels)