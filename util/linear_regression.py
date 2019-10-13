import numpy as np


# x here is
# [ 1 x1
#   1 x2
#   ...
#   1 xn ]
from util.timed import timed


@timed
def mean_feature_normalization(x, features_count):
    x_normalized = x.copy()
    x_normalized_i = np.hsplit(x_normalized, features_count)
    # traverse feature's data
    i = 0
    j = 0
    for x_normalized_i_features in x_normalized_i:
        i = 0
        x_mean = np.mean(x_normalized_i_features)
        x_max = np.max(x_normalized_i_features)
        x_min = np.min(x_normalized_i_features)
        for x_normalized_element in x_normalized_i_features:
            x_normalized[j][i] = (x_normalized_element - x_mean) / (x_max - x_min)
            j += 1
        i += 1
    return x_normalized


def calc_cost_function(x, y, m, thetas):
    h = np.dot(x, thetas)
    loss = (h - y)
    cost = np.sum(loss ** 2) / (2 * m)
    return cost, loss


@timed
def calc_gradient(x, y, learning_rate, theta, cost_function_container, thetas_container, iterations=2000):
    x_with_ones = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    m = len(x)
    for i in range(iterations):
        cost, loss = calc_cost_function(x_with_ones, y, m, theta)
        cost_function_container.append(cost)
        gradient = np.dot(x_with_ones.transpose(), loss) / m
        theta = theta - learning_rate * gradient
        thetas_container.append(theta)
    return theta

