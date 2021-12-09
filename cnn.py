import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import TensorBoard

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def cnn_classifier(train_data, test_data, labels, batch_size=100):
    (x_train, y_train) = train_data
    (x_test, y_test) = test_data

    #Building a CNN

    image_rows = 28
    image_cols = 28
    image_shape = (image_rows, image_cols, 1)

    x_train = x_train.reshape(x_train.shape[0], *image_shape)
    x_test = x_test.reshape(x_test.shape[0], *image_shape)

    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    x_validate = x_validate.reshape(x_validate.shape[0], *image_shape)

    # print('x_train shape: {}'.format(x_train.shape))
    # print('x_test shape: {}'.format(x_test.shape))
    # print('x_validate shape: {}'.format(x_validate.shape))


    model= Sequential([
    Conv2D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling2D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')])

    tensor_board = TensorBoard(
        log_dir = r'logs\{}'.format('layers_2_lr_0_0001'),
        write_graph=True,
        write_grads = True,
        histogram_freq = 1,
        write_images = True
    )

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=SGD(learning_rate=0.1, momentum = 0.2),
        metrics=['accuracy']
    )

    # model.compile(
    #     loss='sparse_categorical_crossentropy',
    #     optimizer=Adam(learning_rate=0.001),
    #     metrics=['accuracy']
    # )

    model.fit(x_train, y_train, batch_size = batch_size, epochs = 10, 
                verbose = 1, validation_data=(x_validate, y_validate),
                callbacks = tensor_board)

    evalulation_score = model.evaluate(x_test, y_test, verbose = 0)

    print('    test loss : {:.4f}'.format(evalulation_score[0]))
    print('test accuracy : {:.4f}'.format(evalulation_score[1]))