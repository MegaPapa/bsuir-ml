import numpy as np


# x here is
# [ 1 x1
#   1 x2
#   ...
#   1 xn ]
def calc_gradient(x, y, learning_rate, theta, cost_function_container, thetas_container, iterations=2000):
    x_with_ones = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    m = len(x)
    for i in range(iterations):
        h = np.dot(x_with_ones, theta)
        loss = (h - y)
        cost = np.sum(loss ** 2) / (2 * m)
        cost_function_container.append(cost)
        gradient = np.dot(x_with_ones.transpose(), loss) / m
        theta = theta - learning_rate * gradient
        thetas_container.append(theta)
    return theta

