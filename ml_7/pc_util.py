import numpy as np


def pca(features):
    m = features.shape[0]
    covariance_matrix = (1 / m) * (features.T @ features)
    u, s, v = np.linalg.svd(covariance_matrix)
    return u, s


def project_data(x, u, e_vectors_amount):
    m = x.shape[0]
    projections = np.zeros((m, e_vectors_amount))
    for i in range(0, m):
        example = x[i, :].T
        projections[i, :] = example.T @ u[:, 0:e_vectors_amount]
    return projections


def recover_data(projections, u, eigenvectors_amount):
    m = projections.shape[0]
    recovered_data = np.zeros((m, u.shape[0]))
    for i in range(0, m):
        projection = projections[i, :].T
        recovered_data[i, :] = projection.T @ u[:, 0:eigenvectors_amount].T
    return recovered_data