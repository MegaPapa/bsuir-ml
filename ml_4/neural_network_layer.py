from ml_4.math import sigmoid_derivative


class NeuralNetworkLayer:

    def __init__(self, activation_function, thetas, size):
        self.activation_function = activation_function
        self.thetas = thetas
        self.size = size
        self.previous_af_result = None

    def forward_propagation(self, input_values):
        self.previous_af_result = self.activation_function(input_values @ self.thetas.T)
        return self.previous_af_result

    def back_propagation(self, delta_from_forward):
        error = delta_from_forward @ self.thetas.T
        own_delta = error * sigmoid_derivative(self.previous_af_result)
        return own_delta

