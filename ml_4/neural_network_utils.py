import numpy as np

from ml_4.math import sigmoid, sigmoid_derivative

EPSILON = 1e-5


def random_weights(l_in_size, l_out_size):
    epsilon_init = np.sqrt(6) / np.sqrt(l_in_size + l_out_size)
    return np.random.rand(l_out_size, l_in_size + 1) * 2 * epsilon_init - epsilon_init


def debug_weights(l_in_size, l_out_size):
    weights = np.zeros((l_out_size, l_in_size + 1))
    return np.reshape(np.sin(np.arange(0, weights.size)), weights.shape) / 10


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


def logistic_loss(h, y):
    return (-y * np.log(h + EPSILON) - (1 - y) * np.log(1 - h + EPSILON)).mean()


def customized_cost_function(nn_theta, x, y, lambda_value, input_layer_size, hidden_layer_size, num_labels):
    x = np.insert(x, 0, 1, axis=1)
    m = x.shape[0]
    theta_1 = np.reshape(nn_theta[:hidden_layer_size * (input_layer_size + 1)],
                         (hidden_layer_size, (input_layer_size + 1)))
    theta_2 = np.reshape(nn_theta[(hidden_layer_size * (input_layer_size + 1)):],
                         (num_labels, (hidden_layer_size + 1)))

    cost = 0
    gradient_1 = np.zeros(theta_1.shape)
    gradient_2 = np.zeros(theta_2.shape)

    for i in range(0, m):
        logistic_y = np.zeros((num_labels, 1))
        logistic_y[y[i, 0] - 1] = 1

        a_1 = x[i, :].reshape((x.shape[1], 1))

        z_2 = theta_1 @ a_1
        a_2 = np.insert(sigmoid(z_2), 0, 1, axis=0)

        z_3 = theta_2 @ a_2
        a_3 = sigmoid(z_3)

        cost_i = -logistic_y.T @ np.log(a_3 + EPSILON) - (1 - logistic_y).T @ np.log(1 - a_3 + EPSILON)
        cost = cost + cost_i

        d_3 = a_3 - logistic_y
        d_2 = (theta_2.T @ d_3) * np.insert(sigmoid_derivative(z_2), 0, 1, axis=0)
        d_2 = d_2[1:]

        gradient_2 = gradient_2 + d_3 @ a_2.T
        gradient_1 = gradient_1 + d_2 @ a_1.T

    cost = (1 / m) * cost + (lambda_value / (2 * m)) * (np.sum(theta_1[:, 1:] ** 2) + np.sum(theta_2[:, 1:] ** 2))
    gradient_2 = (1 / m) * gradient_2
    gradient_2[:, 1:] = gradient_2[:, 1:] + (lambda_value / m) * theta_2[:, 1:]
    gradient_1 = (1 / m) * gradient_1
    gradient_1[:, 1:] = gradient_1[:, 1:] + (lambda_value / m) * theta_1[:, 1:]

    gradient = np.concatenate([gradient_1.ravel(), gradient_2.ravel()])

    # print('Cost ', cost, end='\r')
    return cost, gradient