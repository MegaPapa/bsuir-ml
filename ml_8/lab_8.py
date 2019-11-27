from abstract_lab import Lab
from util.file.matlab_file_reader import read_matlab_file
from util.logger import LoggerBuilder

import numpy as np
import ml_8.util as util
import ml_8.graph as graph

logger = LoggerBuilder().with_name("lab8").build()

DATA_PATH_1 = "./ml_8/resources/ex8data1.mat"
DATA_PATH_2 = "./ml_8/resources/ex8data2.mat"


class EightLab(Lab):

    def __init__(self):
        pass

    def run_lab(self):
        # (1)
        dataset = read_matlab_file(DATA_PATH_1)
        x = dataset.get("X")
        x_val = dataset.get("Xval")
        y_val = dataset.get("yval")
        # (2) - (7)
        mu, sigma = util.check_gauss(x)
        p = util.gaussian_distribution_density(x, mu, sigma)
        p_cv = util.gaussian_distribution_density(x, mu, sigma)
        epsilon, f_1 = util.select_threshold(y_val, p_cv)
        graph.plot_clusters(x, p, epsilon)
        # (8)
        dataset = read_matlab_file(DATA_PATH_2)
        x = dataset.get("X")
        x_val = dataset.get("Xval")
        y_val = dataset.get("yval")
        # (9)
        mu, sigma_squared = util.check_gauss(x)
        # (10) - (12)
        p = util.gaussian_distribution_density(x, mu, sigma_squared)
        p_cv = util.gaussian_distribution_density(x_val, mu, sigma_squared)
        epsilon, f_1 = util.select_threshold(y_val, p_cv)
        logger.info('Generated epsilon = %s (F1 = %s)', epsilon, f_1)
        outliers_count = np.where(p < epsilon)[0].shape[0]
        logger.info('Found %s outliers', outliers_count)
        graph.plot_anomalies(x, mu, sigma, x_val, y_val)
