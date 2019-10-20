import numpy as np
import scipy as sc

from util.timed import timed


def sigmoid(z):
    return 1. / (1. + np.e ** (-z))


def calc_loss(h, y):
    return (-y * np.log(h + 1e-5) - (1 - y) * np.log(1 - h + 1e-5)).mean()


@timed
def optimization_nelder_mead():
    pass

@timed
def optimization_bfgs():
    pass


@timed
def logistic_gradient(x, y, theta, thetas_container, costs_container, learning_rate, iterations=1000):
    x_with_ones = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    m = len(x)
    for i in range(iterations):
        z = x_with_ones @ theta
        h = sigmoid(z)
        loss = calc_loss(h, y)
        cost = np.sum(loss) / m
        gradient = np.dot(x_with_ones.T, (h - y)) / m
        theta = theta - learning_rate * gradient
        thetas_container.append(theta)
        costs_container.append(cost)
    return theta

