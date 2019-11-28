from abstract_lab import Lab
from ml_4 import graph, gradient_checker
from ml_4.neural_network import NeuralNetwork
from util.logger import LoggerBuilder
from util.file.matlab_file_reader import read_matlab_file
import ml_4.neural_network_utils as nnu
import numpy as np
import scipy.optimize as op

DATA_PATH = "./ml_4/resources/ex4data1.mat"
WEIGHTS_PATH = "./ml_4/resources/ex4weights.mat"

logger = LoggerBuilder().with_name("lab4").build()
INPUT_LAYER_SIZE = 400
HIDDEN_LAYER_SIZE = 25
OUTPUT_LAYER_SIZE = 10

class FourthLab(Lab):

    def __init__(self):
        self.lab_number = 4

    def run_lab(self):
        # (1)
        dataset = read_matlab_file(DATA_PATH)
        x = dataset.get("X")
        y = dataset.get("y")
        # (2), sturcture is 3 layers, 1 hidden, L_1 = 25 (+ 1 bias), L_2 = 10 (+1 bias)
        weights = read_matlab_file(WEIGHTS_PATH)
        theta_1 = weights.get("Theta1")
        theta_2 = weights.get("Theta2")
        logger.info("Dataset was loaded successful!")
        # None theta value for output layer
        graph.display_random_data(x, 'random_examples', 20, 20)
        # (3) - (5)
        neural_network = NeuralNetwork(layers_sizes=[25, 10], thetas=[theta_1, theta_2])
        prediction_result = neural_network.forward_propagation(x)

        predicted_rows = np.where(prediction_result.T == np.amax(prediction_result.T, axis=0))[0]
        predicted_rows[predicted_rows == 0] = 10
        prediction = predicted_rows.reshape((x.shape[0], 1))
        logger.info("Accuracy = %s", np.mean(np.double(prediction == y)) * 100)

        # (6) - (10)
        cost, gradient = neural_network.cost_function(x, y, 0)
        logger.info("LAMBDA = 0; Cost = %s ; Gradient = %s", cost, gradient)

        cost, gradient = neural_network.cost_function(x, y, 1)
        logger.info("LAMBDA = 1; Cost = %s ; Gradient = %s", cost, gradient)

        # (9)
        random_theta_1 = nnu.random_weights(400, 25)
        random_theta_2 = nnu.random_weights(25, 10)
        random_theta = np.concatenate([random_theta_1.ravel(), random_theta_2.ravel()])

        # (11) - (13)
        without_reg, gradient = gradient_checker.check_gradients(neural_network.cost_function)
        logger.info("Gradient checking result with lambda = 0: %s ; gradient = %s", without_reg, gradient)

        with_reg, gradient = gradient_checker.check_gradients(neural_network.cost_function, 1)
        logger.info("Gradient checking result with lambda = 1: %s ; gradient = %s", with_reg, gradient)

        neural_network.thetas = [random_theta_1, random_theta_2]
        lambda_ = 1
        # (14) - (15)
        optimal = op.minimize(fun=nnu.customized_cost_function,
                              x0=random_theta,
                              args=(x, y, lambda_, 400, 25, 10),
                              method='TNC',
                              options={
                                  'maxiter': 100,
                              },
                              jac=True)
        logger.info("Optimal = %s", optimal.x)

        theta = np.array([optimal.x]).reshape(random_theta.shape)
        theta_1 = np.reshape(theta[:HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1)],
                             (HIDDEN_LAYER_SIZE, (INPUT_LAYER_SIZE + 1)))
        theta_2 = np.reshape(theta[(HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1)):],
                             (OUTPUT_LAYER_SIZE, (HIDDEN_LAYER_SIZE + 1)))

        # predict based on new recalculated theta
        neural_network.thetas = [theta_1, theta_2]
        prediction_result = neural_network.forward_propagation(x)
        predicted_rows = np.where(prediction_result.T == np.amax(prediction_result.T, axis=0))[0]
        predicted_rows[predicted_rows == 0] = 10
        prediction = predicted_rows.reshape((x.shape[0], 1))
        logger.info("Accuracy = %s", np.mean(np.double(prediction == y)) * 100)