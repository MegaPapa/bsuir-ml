from ml_4.math import sigmoid
from ml_4.neural_network_layer import NeuralNetworkLayer
import numpy as np

class NeuralNetwork:

    def __init__(self, layers_sizes, thetas):
        self.layers = []
        for index, size in enumerate(layers_sizes):
            self.layers.append(
                NeuralNetworkLayer(activation_function=sigmoid, thetas=thetas[index], size=size)
            )

    def forward_propagation(self, x):
        input_value = x
        for layer in self.layers:
            # add biases
            input_value = np.insert(input_value, 0, 1, axis=1)
            input_value = layer.forward_propagation(input_value)
        return input_value

    def back_propagation(self, delta):
        pass