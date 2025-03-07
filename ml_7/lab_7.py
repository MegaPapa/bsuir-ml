import numpy as np
import scipy.optimize as op

from abstract_lab import Lab
from ml_7 import graph
import ml_7.pc_util as util
from ml_7.util import scale_parameters, modify_parameters
from util.file.matlab_file_reader import read_matlab_file
from util.logger import LoggerBuilder

logger = LoggerBuilder().with_name("lab7").build()

DATA_PATH_1 = "./ml_7/resources/ex7data1.mat"
DATA_PATH_2 = "./ml_7/resources/ex7faces.mat"
BIRDSMALL_IMAGE_PATH = "./ml_6/images/bird_small.jpg_2.jpg"

VECTORS_AMOUNT = 1


class SeventhLab(Lab):

    def __init__(self):
        pass

    def run_lab(self):
        # (1)
        dataset = read_matlab_file(DATA_PATH_1)
        x = dataset.get("X")
        logger.info("Xs was loaded successfully. %s", x)
        x_scaled, meta = scale_parameters(x, has_bias_unit=False)
        # (2)
        graph.plot_2d(x)
        # (3) - (8)
        u, s = util.pca(x_scaled)
        projected_data = util.project_data(x_scaled, u, VECTORS_AMOUNT)
        logger.info("Data was projected to less-size with PCA successfully. %s", projected_data)
        recovered_xs = util.recover_data(projected_data, u, VECTORS_AMOUNT)
        logger.info("Data was recovered successfully. %s", recovered_xs)
        graph.plot_loaded_data(x, meta, u, s, x_scaled, projected_data, recovered_xs)
        # (9)
        dataset = read_matlab_file(DATA_PATH_2)
        x = dataset.get("X")
        logger.info("Faces was loaded successfully. %s", x)
        # (10)

        graph.display_data(x, randomize=False)
        x_scaled, meta = scale_parameters(x, has_bias_unit=False)
        # (11)
        u, s = util.pca(x_scaled)
        projected_data = util.project_data(x_scaled, u, 36)
        recovered_features = util.recover_data(projected_data, u, 36)
        # (12) - (13)
        graph.display_data(u[:, 0: 36].T, n_cols=6, n_rows=6, randomize=False)
        # (14) - (15)
        graph.display_data(modify_parameters(recovered_features, meta), randomize=False)
