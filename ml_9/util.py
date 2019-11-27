import numpy as np

from util.logger import LoggerBuilder

logger = LoggerBuilder().with_name("lab8").build()


def movies_ids_line_processor(line):
    return line.replace('\n', '').split(' ', 1)


def read_file_with_line_processor(file_path, line_processor, encoding='UTF-8'):
    full_path = file_path
    logger.debug('Opening file \'%s\'', full_path)
    text_file = open(full_path, "r", encoding=encoding)
    lines = text_file.readlines()
    parsed_lines = []
    for line in lines:
        parsed_lines.append(line_processor(line))
    return parsed_lines


def cost_function(params, ratings, rated_matrix, num_users, num_movies, num_features, lambda_value=0.0):
    x = params[:num_movies * num_features].reshape((num_movies, num_features))
    theta = params[num_movies * num_features:].reshape((num_users, num_features))

    ratings_by_users = x @ theta.T
    loss = (ratings_by_users - ratings) * rated_matrix
    squared_loss = loss ** 2

    theta_regularization_term = (lambda_value / 2) * np.sum(theta ** 2)
    features_regularization_term = (lambda_value / 2) * np.sum(x ** 2)
    cost = (1 / 2) * np.sum(squared_loss) + theta_regularization_term + features_regularization_term

    features_gradient = loss @ theta + x * lambda_value
    theta_gradient = loss.T @ x + theta * lambda_value

    gradient = np.concatenate([features_gradient.ravel(), theta_gradient.ravel()])
    return cost, gradient


def normalize(ratings, user_rates):
    m, n = ratings.shape
    ratings_mean = np.zeros((m, 1))
    normalized_ratings = np.zeros(ratings.shape)
    for i in range(0, m):
        idx = np.where(user_rates[i, :] == 1)
        ratings_mean[i] = np.mean(ratings[i, idx])
        normalized_ratings[i, idx] = ratings[i, idx] - ratings_mean[i]
    return ratings_mean, normalized_ratings


def rate_movies(ratings, rated_matrix):
    my_ratings = np.zeros((ratings.shape[0], 1))
    my_ratings[0] = 5
    my_ratings[93] = 5
    my_ratings[95] = 5
    my_ratings[1436] = 3
    my_ratings[1494] = 3
    my_ratings[1581] = 4
    ratings = np.concatenate([my_ratings, ratings], axis=1)
    rated_matrix = np.concatenate([my_ratings > 0, rated_matrix], axis=1)

    rated_indices = np.where(my_ratings > 0)[0]

    return ratings, rated_matrix


def random_weights(l_in_size, l_out_size, include_bias_unit=True):
    epsilon_init = np.sqrt(6) / np.sqrt(l_in_size + l_out_size)
    return np.random.rand(l_out_size, l_in_size + 1 if include_bias_unit else l_in_size) * 2 * epsilon_init - \
           epsilon_init