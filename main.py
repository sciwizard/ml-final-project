import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np

from svm import svm_classifier
from cnn import cnn_classifier

# Train/test data is stored in a tuple of (data, labels)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Scale data to range [0, 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

print(x_train.shape)

# Translations from numeric label (index in 'labels' array) to English item name
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

svm = True
full_svm_test = True
accuracy = []
x_values = []

if svm:
    if full_svm_test:
        for kernel in ('linear', 'rbf', 'sigmoid'):
            for i in range(6):
                num_samples = (i+1) * 10000
                accuracy += svm_classifier((x_train, y_train), (x_test, y_test), labels, kernel=kernel, num_samples=num_samples)
                x_values += [num_samples]
            plt.plot(x_values, accuracy, label=kernel)
        accuracy = []
        x_values = []
    else:
        for i in range(6):
            num_samples = (i+1) * 10000
            accuracy += svm_classifier((x_train, y_train), (x_test, y_test), labels, kernel='rbf', num_samples=num_samples)
            x_values += [num_samples]
        plt.plot(x_values, accuracy)

    plt.title('SVM Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('# training samples')
    plt.show()

else:
    for i in range(4):
        cnn_classifier((x_train, y_train), (x_test, y_test), labels, batch_size=10**i)




