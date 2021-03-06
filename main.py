import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import sys

from svm import svm_classifier
from cnn import cnn_classifier


def main():
    if len(sys.argv) < 3:
        print("usage: python main.py <CNN/SVM/COMP> <SINGLE/FULL>")
        exit(-1)

    is_svm = False
    is_cnn = False
    compare = False
    full_svm_test = False

    arg1 = sys.argv[1].upper()
    arg2 = sys.argv[2].upper()
    if arg1 == "SVM":
        is_svm = True
    elif arg1 == "COMP":
        compare = True
    elif arg1 == "CNN":
        is_cnn = True
    if arg2 == "FULL":
        full_svm_test = True

    # Train/test data is stored in a tuple of (data, labels)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # Scale data to range [0, 1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print(x_train.shape)

    # Translations from numeric label (index in 'labels' array) to English item name
    labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    accuracy = []
    x_values = []

    if is_svm:
        if full_svm_test:
            for kernel in ('linear', 'sigmoid', 'rbf', 'poly'):
                for i in range(6):
                    num_samples = (i + 1) * 10000
                    accuracy += svm_classifier((x_train, y_train), (x_test, y_test), labels, kernel=kernel,
                                               num_samples=num_samples)
                    x_values += [num_samples]
                plt.plot(x_values, accuracy, label=kernel)
                accuracy.clear()
                x_values.clear()
        else:
            for i in range(6):
                num_samples = (i + 1) * 10000
                accuracy += svm_classifier((x_train, y_train), (x_test, y_test), labels, kernel='rbf',
                                           num_samples=num_samples)
                x_values += [num_samples]
            plt.plot(x_values, accuracy)

        plt.title('SVM Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('# training samples')
        plt.legend()
        plt.show()

    if is_cnn:
        # for i in range(1):
        cnn_classifier((x_train, y_train), (x_test, y_test), labels)

    if compare:
        svm_accuracy = []
        cnn_accuracy = []
        x_values = [6000, 12000, 24000, 36000, 48000]
        for split_size in (.9, .8, .6, .4, .2):
            cnn_accuracy += [cnn_classifier((x_train, y_train), (x_test, y_test), labels, split=split_size)]

        for sample_size in x_values:
            svm_accuracy += [svm_classifier((x_train, y_train), (x_test, y_test), labels, kernel='rbf',
                                           num_samples=sample_size)]


        plt.title('Best CNN vs SVN Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('# training samples')
        plt.plot(x_values, svm_accuracy, label='SVN')
        plt.plot(x_values, cnn_accuracy, label='CNN')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
