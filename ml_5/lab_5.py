import numpy as np
import scipy.optimize as op

from abstract_lab import Lab
from ml_5 import graph
from ml_5.text_processing import is_spam
from ml_5.util import linear_svc, gauss_kernel, find_best_params, rbf_svc, find_best_c
from util.file.data_loader import read_vocabulary
from util.logger import LoggerBuilder
from util.file.matlab_file_reader import read_matlab_file

logger = LoggerBuilder().with_name("lab5").build()

DATA_PATH_1 = "./ml_5/resources/ex5data1.mat"
DATA_PATH_2 = "./ml_5/resources/ex5data2.mat"
DATA_PATH_3 = "./ml_5/resources/ex5data3.mat"
SPAM_TRAIN_DATA = "./ml_5/resources/spamTrain.mat"
SPAM_TEST_DATA = "./ml_5/resources/spamTest.mat"
PATH_TO_EMAIL_1_FILE = './ml_5/resources/emailSample1.txt'
PATH_TO_EMAIL_2_FILE = './ml_5/resources/emailSample2.txt'
PATH_TO_SPAM_1_FILE = './ml_5/resources/spamSample1.txt'
PATH_TO_SPAM_2_FILE = './ml_5/resources/spamSample2.txt'
PATH_TO_REAL_SPAM_1_FILE = './ml_5/resources/custom/spam_1.txt'
PATH_TO_REAL_SPAM_2_FILE = './ml_5/resources/custom/spam_2.txt'


GAMMA = 30

class FifthLab(Lab):

    def __init__(self):
        pass

    def run_lab(self):
        # (1) - (4)
        dataset = read_matlab_file(DATA_PATH_1)
        x = dataset.get("X")
        y = dataset.get("y")
        # optimazed_1 = linear_svc(x, y.ravel(), 1)
        # optimazed_100 = linear_svc(x, y.ravel(), 100)
        # graph.draw_plot(x, y, optimazed_1, optimazed_100)
        # (5)
        g_kernel = gauss_kernel(x)
        logger.info("Gauss kernell = %s", g_kernel)
        # (6)
        dataset = read_matlab_file(DATA_PATH_2)
        x = dataset.get("X")
        y = dataset.get("y")
        # (7)
        g_kernel = gauss_kernel(x)
        logger.info("Gauss kernell for x's in ex5data2 = %s", g_kernel)
        # (8)
        # optimazed_1 = rbf_svc(x, y.ravel(), 1, GAMMA)
        # optimazed_100 = rbf_svc(x, y.ravel(), 100, GAMMA)
        # (9)
        # graph.draw_plot(x, y, optimazed_1, optimazed_100)
        # (10)
        dataset = read_matlab_file(DATA_PATH_3)
        x = dataset.get("X")
        y = dataset.get("y")
        x_val = dataset.get("Xval")
        y_val = dataset.get("yval")
        # (11)
        optimal_c, optimal_gamma = find_best_params(x, y, x_val, x_val)
        logger.info("Optimal C = %s ; Optimal gamma = %s", optimal_c, optimal_gamma)
        # (12)
        # optimazed = rbf_svc(x, y.ravel(), optimal_c, optimal_gamma)
        # graph.draw_plot_with_decision_boundary(x, y, lambda v: optimazed.predict(v),
        #                                        optimazed.support_vectors_, 1, draw_kernel_points=True)
        # (13)
        dataset = read_matlab_file(SPAM_TRAIN_DATA)
        x = dataset.get("X")
        y = dataset.get("y")
        # (14)
        # optimazed = rbf_svc(x, y.ravel(), 1, optimal_gamma)
        #
        dataset = read_matlab_file(SPAM_TEST_DATA)
        x_test = dataset.get("Xtest")
        y_test = dataset.get("ytest")
        # (16) - (17)
        optimal_c = find_best_c(x, y, x_test, y_test)
        optimized = rbf_svc(x, y, optimal_c)
        logger.info('Accuracy on training data: %s', np.mean(np.double(optimized.predict(x) == y.T)) * 100)
        logger.info('Accuracy on test data: %s', np.mean(np.double(optimized.predict(x_test) == y_test.T)) * 100)
        # (18) - (25)
        voc, voc_size = read_vocabulary()
        is_spam('Email sample 1', PATH_TO_EMAIL_1_FILE, optimized, voc, voc_size)
        is_spam('Email sample 2', PATH_TO_EMAIL_2_FILE, optimized, voc, voc_size)
        is_spam('Spam sample 1', PATH_TO_SPAM_1_FILE, optimized, voc, voc_size)
        is_spam('Spam sample 2', PATH_TO_SPAM_2_FILE, optimized, voc, voc_size)
        is_spam('Real spam 1', PATH_TO_REAL_SPAM_1_FILE, optimized, voc, voc_size)
        is_spam('Real spam 2', PATH_TO_REAL_SPAM_2_FILE, optimized, voc, voc_size)
