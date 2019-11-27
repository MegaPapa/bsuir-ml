from abstract_lab import Lab
from ml_9 import util
from util.file.matlab_file_reader import read_matlab_file
from util.logger import LoggerBuilder
import scipy.optimize as op

import numpy as np

logger = LoggerBuilder().with_name("lab8").build()

MOVIES_PATH = "./ml_9/resources/ex9_movies.mat"
MOVIES_IDS = "./ml_9/resources/movie_ids.txt"

LAMBDA = 10
XS_COUNT = 10


class NinthLab(Lab):

    def __init__(self):
        pass

    def run_lab(self):
        # (1)
        dataset = read_matlab_file(MOVIES_PATH)
        ratings = dataset.get("Y")
        user_rates = dataset.get("R")
        # (2)
        movies_amount, users_amount = ratings.shape
        users_amount = users_amount + 1


        ratings, rated_matrix = util.rate_movies(ratings, user_rates)
        # (3) - (7)

        ratings_mean, normalized_ratings = util.normalize(ratings, user_rates)

        init_features = util.random_weights(XS_COUNT, movies_amount, include_bias_unit=False)
        init_theta = util.random_weights(XS_COUNT, users_amount, include_bias_unit=False)
        initial_parameters = np.concatenate([init_features.ravel(), init_theta.ravel()])
        logger.info("Starting learning...")
        optimal = op.minimize(fun=util.cost_function,
                              x0=initial_parameters,
                              args=(normalized_ratings, user_rates, users_amount, movies_amount,
                                    XS_COUNT, LAMBDA),
                              method='TNC',
                              options={
                                  'maxiter': 100,
                              },
                              jac=True)

        optimal_theta = optimal.x
        logger.info("Was found optimal theta value = %s", optimal_theta)
        features = optimal_theta[:movies_amount * XS_COUNT].reshape((movies_amount, XS_COUNT))
        theta = optimal_theta[movies_amount * XS_COUNT:].reshape((users_amount, XS_COUNT))

        # (8)
        movies = util.read_file_with_line_processor(MOVIES_IDS, util.movies_ids_line_processor, encoding="ISO-8859-1")
        # (9) - (10)
        predictions = features @ theta.T
        predictions_for_me = predictions[:, 0].reshape((movies_amount, 1)) + ratings_mean
        top_predictions = np.where(predictions_for_me > 4.5)[0]
        top_predictions = top_predictions[0:5]
        logger.info("Next movies are recommending for you:")
        for top_prediction in top_predictions:
            logger.info('\t%s', movies[top_prediction][1])