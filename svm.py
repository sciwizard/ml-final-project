import sklearn as sk
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import numpy as np
import matplotlib.pyplot as plt

def svm_classifier(train_tuple, test_tuple, labels_list):
    (x_train, y_train) = train_tuple
    (x_test, y_test) = test_tuple
    labels = labels_list
    
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    print("Training model...")
    clf.fit(x_train[:100], y_train[:100])

    ## Sanity check, picks a random test datum and classifies it ##
    #sample = np.random.randint(y_test.shape[0])
    #prediction = clf.predict([x_test[sample]])[0]
    #plt.imshow(x_test[sample].reshape(28,28))
    #plt.show()
    #print("predicted {}, actual is {}".format(labels[prediction], labels[y_test[sample]]))

    # Calculate test accuracy
    print("Calculating model accuracy...")
    correct = 0
    for row in range(y_test.shape[0]):
        prediction = clf.predict([x_test[row]])[0]
        label = y_test[row]
        if prediction == label:
            correct += 1
    accuracy = correct / y_test.shape[0]
    print("\nAccuracy: {}%".format(100 * accuracy))

