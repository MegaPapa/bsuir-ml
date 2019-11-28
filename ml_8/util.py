import numpy as np


def check_gauss(x):
    m, n = x.shape
    mu = (np.sum(x, 0) / m).reshape((n, 1))
    sigma_squared = (np.sum((x - mu.T) ** 2, 0) / m).reshape((n, 1))
    return mu, sigma_squared


def gaussian_distribution_density(x, mu, sigma_squared):
    k = mu.shape[0]

    if sigma_squared.shape[1] == 1 or sigma_squared.shape[0] == 1:
        sigma_squared = np.diag(sigma_squared.reshape(-1))

    x = x - mu.T
    return (2 * np.pi) ** (- k / 2) * np.linalg.det(sigma_squared) ** (-0.5) * \
           np.exp(-0.5 * np.sum(x @ np.linalg.pinv(sigma_squared) * x, 1)).reshape((x.shape[0], 1))


def select_threshold(y, p):
    max_p = np.max(p)
    min_p = np.min(p)
    best_f_1 = None
    best_epsilon = None

    steps_count = 1000
    step_size = (max_p - min_p) / steps_count
    step_index = 0
    epsilon = min_p
    while step_index < steps_count:
        step_index = step_index + 1
        epsilon = epsilon + step_size
        cv_predictions = p < epsilon

        false_positives = np.sum((cv_predictions == 1) & (y == 0))
        true_positives = np.sum((cv_predictions == 1) & (y == 1))
        false_negatives = np.sum((cv_predictions == 0) & (y == 1))

        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f_1 = (2 * precision * recall) / (precision + recall + 1e-10)

        if best_f_1 is None or best_f_1 < f_1:
            best_f_1 = f_1
            best_epsilon = epsilon

    return best_epsilon, best_f_1