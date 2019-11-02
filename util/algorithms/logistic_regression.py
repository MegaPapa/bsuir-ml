import numpy as np
import pandas
import scipy.optimize as op

from util.logger import LoggerBuilder
from sklearn.preprocessing import PolynomialFeatures
from util.timed import timed


EPSYLON = 1e-5

logger = LoggerBuilder().with_name("logistic_regression").build()

def sigmoid(z):
    return 1. / (1 + np.e ** (-z))


def calc_cost_function(x, y, theta, learning_rate, m):
    z = x @ theta
    h = sigmoid(z)
    loss = calc_loss(h, y)
    cost = np.sum(loss) / m
    gradient = np.dot(x.T, (h - y)) / m
    theta = theta - learning_rate * gradient
    return cost, gradient, theta


def calc_loss(h, y):
    return (-y * np.log(h + EPSYLON) - (1 - y) * np.log(1 - h + EPSYLON)).mean()


def predict(x, thetas):
    x_with_ones = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    z = x_with_ones @ thetas
    return sigmoid(z)


@timed
def logistic_gradient(x, y, theta, thetas_container, costs_container, learning_rate, iterations=1000):
    x_with_ones = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    m = len(x_with_ones)
    for i in range(iterations):
        cost, gradient, theta = calc_cost_function(x_with_ones, y, theta, learning_rate, m)
        thetas_container.append(theta)
        costs_container.append(cost)
    return theta

###
def compute_cost_function(theta, features, results):
    m = results.shape[0]
    theta = np.reshape(theta, (features.shape[1], 1))
    h = sigmoid(features @ theta)

    one_case = np.matmul(-results.T, np.log(h + EPSYLON))
    zero_case = np.matmul(-(1 - results).T, np.log(1. - h + EPSYLON))
    j = (1 / m) * (one_case + zero_case)
    gradient = (1 / m) * ((h - results).T @ features).T
    return j, gradient


def compute_cost_function_with_regularization(theta, features, results, lambda_):
    m = results.shape[0]
    j, gradient = compute_cost_function(theta, features, results)
    reg_j = (lambda_ / (2.0 * m)) * np.sum(theta[1:] ** 2)
    reg_gradient = (lambda_ / m) * theta
    reg_gradient[0] = 0
    return j + reg_j, gradient + reg_gradient


def compute_with_bfgs(x, theta, y):
    x_with_ones = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    j, gradient = compute_cost_function(theta, x_with_ones, y)

    optimal = op.minimize(fun=compute_cost_function,
                          x0=theta,
                          args=(x_with_ones, y),
                          method='L-BFGS-B',
                          jac=True)

    theta = np.array([optimal.x]).reshape(theta.shape)

    j, gradient = compute_cost_function(theta, x_with_ones, y)

    return theta


def compute_with_nelder_mead(x, theta, y):
    x_with_ones = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    j, gradient = compute_cost_function(theta, x_with_ones, y)

    optimal = op.minimize(fun=compute_cost_function,
                          x0=theta,
                          args=(x_with_ones, y),
                          method='Nelder-Mead',
                          jac=True)

    theta = np.array([optimal.x]).reshape(theta.shape)

    j, gradient = compute_cost_function(theta, x_with_ones, y)

    return theta

def compute_with_tnc(x, theta, y):
    x_with_ones = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    j, gradient = compute_cost_function(theta, x_with_ones, y)

    optimal = op.minimize(fun=compute_cost_function,
                          x0=theta,
                          args=(x_with_ones, y),
                          method='TNC',
                          jac=True)

    theta = np.array([optimal.x]).reshape(theta.shape)

    j, gradient = compute_cost_function(theta, x_with_ones, y)

    return theta

def print_polynomial_with_two_features(n):
    data = pandas.DataFrame.from_dict({
        'x': np.random.randint(low=1, high=10, size=5),
        'y': np.random.randint(low=-1, high=1, size=5),
    })
    polynomial = PolynomialFeatures(degree=n).fit(data)
    logger.info(polynomial.get_feature_names(data.columns))