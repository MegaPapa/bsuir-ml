from abstract_lab import Lab
from ml_3 import logistic_regression, graph, util, lambda_finder
from util.file.matlab_file_reader import read_matlab_file
from util.logger import LoggerBuilder
import numpy as np

DATA_PATH = "./ml_3/resources/ex3data1.mat"

logger = LoggerBuilder().with_name("lab3").build()

LAMBDA = 1
ALPHA = 0.001


class ThirdLab(Lab):

    def __init__(self):
        self.lab_number = 3

    def run_lab(self):
        # (1) - (4)
        dataset = read_matlab_file(DATA_PATH)
        x = dataset.get("X")
        y = dataset.get("y")
        Xtest = dataset.get("Xtest")
        Ytest = dataset.get("ytest")
        Xval = dataset.get("Xval")
        Yval = dataset.get("yval")
        logger.info("Dataset was loaded.")
        graph.show_plot_by_points(x, y)
        # (5)
        thetas_with_lambda_equaled_zero, last_cost  = logistic_regression.gradient_descent_with_reg(x, y, learning_rate=ALPHA,
                                                                                        iterations=20000, lambda_=0)
        logger.info("Thetas with LABMDA = 0.\n %s", thetas_with_lambda_equaled_zero)
        thetas_with_lambda_not_equaled_zero, last_cost = logistic_regression.gradient_descent_with_reg(x, y, learning_rate=ALPHA,
                                                                                        iterations=20000, lambda_=10)
        logger.info("Thetas with LABMDA != 0.\n %s", thetas_with_lambda_not_equaled_zero)
        graph.show_graphic_by_thetas(thetas_with_lambda_equaled_zero, x, y, -40, 40, additional_concatenation=True)
        # (6)
        # with test samples
        learning_curves_test_sample = []
        sample_length = len(Xtest)
        for i in range(sample_length):
            thetas, container, last_cost = logistic_regression.gradient_descent_with_reg(Xtest[0:i+1], Ytest[0:i+1], learning_rate=ALPHA,
                                                      iterations=20000, lambda_=0, save_results=True)
            learning_curves_test_sample.append(last_cost)
        graph.show_learning_curve(sample_length, learning_curves_test_sample)
        # with validation samples
        learning_curves_validation_sample = []
        sample_length = len(Xval)
        for i in range(sample_length):
            thetas, container, last_cost = logistic_regression.gradient_descent_with_reg(Xval[0:i + 1], Yval[0:i + 1],
                                                                              learning_rate=ALPHA,
                                                                              iterations=20000, lambda_=0,
                                                                              save_results=True)
            learning_curves_validation_sample.append(last_cost)
        graph.show_learning_curve(sample_length, learning_curves_validation_sample)

        # (7) - (8)
        poly_features = util.get_poly_feature(x[:,0], 7)
        # poly_features, meta = util.scale_parameters(poly_features)
        # cut first "ones" vector, after normalization we'll return it
        poly_features = util.mean_feature_normalization(poly_features[:,1:], poly_features.shape[1] - 1)
        poly_features = np.insert(poly_features, 0, 1, axis=1)
        # (9) - (10)

        learning_curves_learning_sample = []
        sample_length = len(poly_features)
        for i in range(sample_length):
            thetas, container, last_cost = logistic_regression.gradient_descent_with_reg(poly_features[:,0:i + 1], y[:, 0:i + 1],
                                                                              learning_rate=0.1,
                                                                              iterations=20000, lambda_=0,
                                                                              save_results=True, add_ones=False)
            learning_curves_learning_sample.append(last_cost)
        graph.show_learning_curve(sample_length, learning_curves_learning_sample)

        # (11)
        # Lambda = 1
        learning_curves_learning_sample = []
        sample_length = len(poly_features)
        for i in range(sample_length):
            thetas, container, last_cost = logistic_regression.gradient_descent_with_reg(poly_features[:,0:i + 1], y[:, 0:i + 1],
                                                                              learning_rate=0.1,
                                                                              iterations=20000, lambda_=1,
                                                                              save_results=True, add_ones=False)
            learning_curves_learning_sample.append(last_cost)
        graph.show_learning_curve(sample_length, learning_curves_learning_sample)

        # Lambda = 100
        learning_curves_learning_sample = []
        sample_length = len(poly_features)
        for i in range(sample_length):
            thetas, container, last_cost = logistic_regression.gradient_descent_with_reg(poly_features[:,0:i + 1], y[:, 0:i + 1],
                                                                              learning_rate=0.1,
                                                                              iterations=20000, lambda_=100,
                                                                              save_results=True, add_ones=False)
            learning_curves_learning_sample.append(last_cost)
        graph.show_learning_curve(sample_length, learning_curves_learning_sample)
        # (12)
        optimal_lambda, cost_with_optimal_lambda = lambda_finder.find_lambda(Xval, Yval)
        logger.info("Optimal lambda = %s", optimal_lambda)
        # (13)
        optimal_theta, opt_container, opt_cost = logistic_regression.gradient_descent_with_reg(poly_features, y,
                                                      learning_rate=0.1,
                                                      iterations=20000, lambda_=0,
                                                      save_results=True, add_ones=False)
        graph.show_graphic_by_thetas(optimal_theta, x, y, -50, 40)
        logger.info("Optimal theta = %s", optimal_theta)