import sklearn as sk
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt

hot_confusion = np.identity(10)
confusion = np.zeros((10, 10))

def svm_classifier(train_tuple, test_tuple, labels, kernel='rbf', num_samples=100):
    (x_train, y_train) = train_tuple
    (x_test, y_test) = test_tuple
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

    print('training... ', end='')
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel=kernel))
    clf.fit(x_train[:num_samples], y_train[:num_samples])

    ## Sanity check, picks a random test datum and classifies it ##
    # sample = np.random.randint(y_test.shape[0])
    # prediction = clf.predict([x_test[sample]])[0]
    # plt.imshow(x_test[sample].reshape(28,28))
    # plt.show()
    # print("predicted {}, actual is {}".format(labels[prediction], labels[y_test[sample]]))

    # Calculate test accuracy
    print("calculating accuracy...", end='')
    correct = 0
    for row in range(y_test.shape[0]):
        prediction = clf.predict([x_test[row]])[0]
        confusion[y_test[row]] += hot_confusion[prediction]
        label = y_test[row]
        if prediction == label:
            correct += 1
    accuracy = correct / y_test.shape[0]
    plt.matshow(confusion)
    plt.xlabel('Actual Label')
    plt.ylabel('Predicted Label')
    plt.title('SVM Confusion Matrix')
    plt.xticks(np.arange(10))
    plt.yticks(np.arange(10))

    print("\nModel with kernel={}, num training data={}, accuracy={}".format(kernel, num_samples, 100 * accuracy))
    return [accuracy]
