import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random

from util.algorithms.logistic_regression import sigmoid

EPSILON = 1e-5

def compute_with_tnc(features, result, lambda_, number_count):
    m, n = features.shape
    initial_theta = np.zeros((n, 1)).reshape(-1)
    all_theta = np.zeros((number_count, n))
    for i in range(10):
        i_class = i if i else 10
        logic_result = np.array([1 if x == i_class else 0 for x in result]).reshape((features.shape[0], 1))
        optimal = op.minimize(fun=compute_cost_function_with_regularization,
                              x0=initial_theta,
                              args=(features, logic_result, lambda_),
                              method='TNC',
                              jac=True)
        all_theta[i, :] = optimal.x

    predictions_by_numbers = features @ all_theta.T
    predicted_rows = np.where(predictions_by_numbers.T == np.amax(predictions_by_numbers.T, axis=0))[0]
    predicted_rows[predicted_rows == 0] = 10

    return all_theta, np.array([predicted_rows]).T


def compute_with_slsqp(features, result, lambda_, number_count):
    m, n = features.shape
    initial_theta = np.zeros((n, 1)).reshape(-1)
    all_theta = np.zeros((number_count, n))
    for i in range(10):
        i_class = i if i else 10
        logic_result = np.array([1 if x == i_class else 0 for x in result]).reshape((features.shape[0], 1))
        optimal = op.minimize(fun=compute_cost_function_with_regularization,
                              x0=initial_theta,
                              args=(features, logic_result, lambda_),
                              method='SLSQP',
                              jac=True)
        all_theta[i, :] = optimal.x

    predictions_by_numbers = features @ all_theta.T
    predicted_rows = np.where(predictions_by_numbers.T == np.amax(predictions_by_numbers.T, axis=0))[0]
    predicted_rows[predicted_rows == 0] = 10

    return all_theta, np.array([predicted_rows]).T


def compute_cost_function_with_regularization(theta, features, results, lambda_value):
    m = results.shape[0]
    theta = np.reshape(theta, (features.shape[1], 1))
    h = sigmoid(features @ theta)

    one_case = np.matmul(-results.T, np.log(h + EPSILON))
    zero_case = np.matmul(-(1 - results).T, np.log(1. - h + EPSILON))
    j = (1 / m) * (one_case + zero_case) + (lambda_value / (2 * m)) * np.sum(theta[1:] ** 2)

    gradient = np.zeros(theta.shape)
    gradient[0] = (1 / m) * ((h - results).T @ features[:, 0]).T
    gradient[1:] = (1 / m) * ((h - results).T @ features[:, 1:]).T + (lambda_value / m) * theta[1:]

    return j, gradient
