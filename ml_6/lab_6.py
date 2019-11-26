import numpy as np
import scipy.optimize as op

from abstract_lab import Lab
from ml_6 import util, graph
from ml_6.k_mean import k_mean_algorithm, hierarchical_clustering
from ml_6.util import translate_mat_to_compressed_img, read_image
from util.logger import LoggerBuilder
from util.file.matlab_file_reader import read_matlab_file

logger = LoggerBuilder().with_name("lab5").build()

DATA_PATH_1 = "./ml_6/resources/ex6data1.mat"
DATA_PATH_2 = "./ml_6/resources/bird_small.mat"
INIT_CENTERS_COUNT = 3
ITERATIONS_COUNT = 100
BIRDSMALL_CLASSES_COUNT = 16
WOLF_CLASSES_COUNT = 16

BIRDSMALL_IMAGE_PATH = "./ml_6/images/bird_small.jpg"
WOLF_IMAGE_PATH = "./ml_6/images/wolf.jpg"


class SixthLab(Lab):

    def __init__(self):
        pass

    def run_lab(self):
        # (1)
        dataset = read_matlab_file(DATA_PATH_1)
        x = dataset.get("X")
        # (2)
        centers = util.init_centers(x, INIT_CENTERS_COUNT)
        # (3) - (5)
        computed_centers, indices, history = k_mean_algorithm(x, centers, ITERATIONS_COUNT)
        # (6)
        graph.draw_clusters_with_traectory(x, indices, computed_centers, history)
        # (7)
        dataset = read_matlab_file(DATA_PATH_2)
        x = dataset.get("A")
        # (8) - (9)
        translate_mat_to_compressed_img(x, BIRDSMALL_CLASSES_COUNT, ITERATIONS_COUNT, BIRDSMALL_IMAGE_PATH)
        # (10)
        additional_image_data = read_image(WOLF_IMAGE_PATH)
        translate_mat_to_compressed_img(additional_image_data, WOLF_CLASSES_COUNT, ITERATIONS_COUNT, WOLF_IMAGE_PATH)
        # (11)
        hierarchical_clustering(x, 48)

