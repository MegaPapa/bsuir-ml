import numpy as np


def sigmoid_derivative(x):
    return x * (1 - x)


def sigmoid(z):
    return 1. / (1 + np.e ** (-z))


def sigmoid_backward(dA, z):
    S = sigmoid(z)
    dS = S * (1 - S)
    return dA * dS

