from abstract_lab import Lab
from ml_4 import graph
from ml_4.neural_network import NeuralNetwork
from util.logger import LoggerBuilder
from util.file.matlab_file_reader import read_matlab_file
import ml_4.neural_network_utils as nnu
import numpy as np

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
        # (2), sturcture is 4 layers, 1 hidden, L_1 = 25 (+ 1 bias), L_2 = 10 (+1 bias)
        weights = read_matlab_file(WEIGHTS_PATH)
        theta_1 = weights.get("Theta1")
        theta_2 = weights.get("Theta2")
        logger.info("Dataset was loaded successful!")
        # None theta value for output layer
        graph.display_random_data(x, 'random_examples', 20, 20)
        neural_network = NeuralNetwork(layers_sizes=[25, 10], thetas=[theta_1, theta_2])
        prediction_result = neural_network.forward_propagation(x)

        predicted_rows = np.where(prediction_result.T == np.amax(prediction_result.T, axis=0))[0]
        predicted_rows[predicted_rows == 0] = 10
        prediction = predicted_rows.reshape((x.shape[0], 1))
        logger.info("Accuracy = %s", np.mean(np.double(prediction == y)) * 100)
