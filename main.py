import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import sys

from svm import svm_classifier
from cnn import cnn_classifier

def main():
    is_svm = True
    kernel = 'rbf'
    if len(sys.argv) < 3:
        print("usage: python main.py <CNN/SVM> [kernel]")
        exit(-1)
    
    print(len(sys.argv))

    is_svm = True if sys.argv[1].upper() == "SVM" else False
    kernel = sys.argv[2] if is_svm else kernel        

    # Train/test data is stored in a tuple of (data, labels)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Scale data to range [0, 1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print(x_train.shape)

    # Translations from numeric label (index in 'labels' array) to English item name
    labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    if is_svm:
        for i in range(6):
            svm_classifier((x_train, y_train), (x_test, y_test), labels, kernel, num_samples=(i+1) * 10000)
    else:
        for i in range(4):
            cnn_classifier((x_train, y_train), (x_test, y_test), labels, batch_size=10**i)


if __name__ == "__main__":
    main()