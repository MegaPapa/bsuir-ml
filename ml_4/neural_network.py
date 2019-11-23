from ml_4.math import sigmoid, sigmoid_derivative
from ml_4.neural_network_layer import NeuralNetworkLayer
import numpy as np

from ml_4.neural_network_utils import logistic_loss, EPSILON


class NeuralNetwork:

    def __init__(self, layers_sizes, thetas):
        self.layers = []
        self.thetas = thetas
        for index, size in enumerate(layers_sizes):
            self.layers.append(
                NeuralNetworkLayer(activation_function=sigmoid, thetas=thetas[index], size=size)
            )

    def forward_propagation(self, x, add_bias=True):
        input_value = x
        for layer in self.layers:
            # add biases
            if add_bias:
                input_value = np.insert(input_value, 0, 1, axis=1)
            input_value = layer.forward_propagation(input_value)
        return input_value

    def back_propagation(self, delta):
        transitive_delta = delta
        for layer in reversed(self.layers):
            transitive_delta = layer.back_propagation(transitive_delta)

    def cost_function(self, x, y, lambda_, theta_1=None, theta_2=None, num_labels=10, big_thetas=None):
        if theta_1 is None:
            theta_1 = self.thetas[0]
        if theta_2 is None:
            theta_2 = self.thetas[1]
        if theta_1 is 1:
            theta_1 = self.thetas[0]
        if big_thetas is not None:
            theta_1 = np.reshape(big_thetas[:20],
                                 (5, (4)))
            theta_2 = np.reshape(big_thetas[(20):],
                                 (3, (6)))
        x = np.insert(x, 0, 1, axis=1)
        m = x.shape[0]
        cost = 0
        gradient_1 = np.zeros(theta_1.shape)
        gradient_2 = np.zeros(theta_2.shape)
        for i in range(m):
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

        cost = (1 / m) * cost + (lambda_ / (2 * m)) * (np.sum(theta_1[:, 1:] ** 2) + np.sum(theta_2[:, 1:] ** 2))
        gradient_2 = (1 / m) * gradient_2
        gradient_2[:, 1:] = gradient_2[:, 1:] + (lambda_ / m) * theta_2[:, 1:]
        gradient_1 = (1 / m) * gradient_1
        gradient_1[:, 1:] = gradient_1[:, 1:] + (lambda_ / m) * theta_1[:, 1:]

        gradient = np.concatenate([gradient_1.ravel(), gradient_2.ravel()])

        return cost, gradient
