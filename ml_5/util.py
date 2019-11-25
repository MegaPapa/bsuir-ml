from sklearn.svm import SVC
import numpy as np


def find_best_params(x, y, features_cv, result_cv):
    possible_c = [0.5, 1, 3, 10, 30]
    possible_c_size = len(possible_c)
    possible_gamma = [0.5, 1, 3, 10, 30]
    possible_gamma_size = len(possible_gamma)

    best_c = None
    best_gamma = None
    maximal_accuracy = None
    for i in range(0, possible_c_size):
        for j in range(0, possible_gamma_size):
            c_test = possible_c[i]
            gamma_test = possible_gamma[j]
            model = rbf_svc(x, y, c_test, gamma_test)
            prediction_accuracy = np.mean(np.double(model.predict(features_cv) == result_cv))
            if maximal_accuracy is None or maximal_accuracy < prediction_accuracy:
                maximal_accuracy = prediction_accuracy
                best_c = c_test
                best_gamma = gamma_test

    return best_c, best_gamma


def gauss_kernel(x, gamma=0.5):
    x0, x1 = x[:, 0], x[:, 1]
    return np.exp(-gamma * max(x0 - x1) ** 2)


def linear_svc(x, y, c):
    model = SVC(c, kernel="linear").fit(x, y)
    return model


def rbf_svc(x, y, c, gamma=0.1):
    model = SVC(c, kernel="rbf", gamma=gamma).fit(x, y)
    return model


def find_best_c(x, y, x_test, y_test):
    possible_c = [1, 0.5, 0.1]
    possible_c_size = len(possible_c)

    best_c = None
    maximal_accuracy = None
    for i in range(0, possible_c_size):
        c_test = possible_c[i]
        model = rbf_svc(x, y, c_test)
        prediction_accuracy = np.mean(np.double(model.predict(x_test) == y_test.T)) * 100
        if maximal_accuracy is None or maximal_accuracy < prediction_accuracy:
            if best_c is not None:
                maximal_accuracy = prediction_accuracy
                best_c = c_test

    return best_c


def convert_to_features(word_indices, vocabulary_size):
    features = np.zeros((1, vocabulary_size))
    features[0, word_indices] = 1
    return features
