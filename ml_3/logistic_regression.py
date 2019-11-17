import numpy as np

from ml_3.gradient_container import GradientContainer

EPSILON = 1e-5


def calc_cost_function(x, y, theta, m, loss_function, hypotesis_function, lambda_=0):
    theta = np.reshape(theta, (x.shape[1], 1))
    h = hypotesis_function(x, theta)
    loss = loss_function(h, y)
    lambda_component = (lambda_ / (2 * m)) * np.sum(theta[1:] ** 2)
    cost = (np.sum(loss ** 2) / m) + lambda_component
    # cost = (np.sum(loss) + lambda_component) / m
    gradient = np.zeros(theta.shape)
    gradient[0] = ((x.T @ (h - y))[0] / m)
    gradient[1:] = ((x.T @ (h - y)) / m)[1:] + ((lambda_ * theta[1:]) / m)
    return cost, gradient, loss


def linear_hypotesis(x, thetas):
    return x @ thetas


def sigmoid(x, thetas):
    z = x @ thetas
    return 1. / (1 + np.e ** (-z))


def linear_loss(h, y):
    return h - y


def logistic_loss(h, y):
    return (-y * np.log(h + EPSILON) - (1 - y) * np.log(1 - h + EPSILON)).mean()



def gradient_descent_with_reg(x, y, learning_rate=0.0224, iterations=10000, is_linear=True, lambda_=0, save_results=False, add_ones=True):
    if add_ones:
        x_with_ones = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    else:
        x_with_ones = x
    thetas = np.ones((x_with_ones.shape[1], 1))
    m = len(x_with_ones)
    container = GradientContainer()
    for i in range(iterations):
        if is_linear:
            cost, gradient, loss = calc_cost_function(x_with_ones, y, thetas, m,
                                                       linear_loss, linear_hypotesis, lambda_)
        else:
            cost, gradient, loss = calc_cost_function(x_with_ones, y, thetas, m, logistic_loss,
                                                       sigmoid, lambda_)
        thetas = thetas - learning_rate * gradient
    if save_results:
        container.save_snapshot(cost, gradient, loss, thetas)
    if save_results:
        return thetas, container, cost
    else:
        return thetas, cost
