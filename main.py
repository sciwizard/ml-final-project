import tensorflow as tf
import matplotlib.pyplot as plt

# Train/test data is stored in a tuple of (data, labels)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Translations from numeric label (index in 'labels' array) to English item name
labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Print the label for the first training example and show it with matplotlib
print(labels[y_train[0]])
plt.imshow(x_train[0])
plt.show()