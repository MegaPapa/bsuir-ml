import numpy as np

from ml_3 import logistic_regression, util

LAMBDA_END = 100
ALPHA = 0.1

def find_lambda(Xval, Yval):
    results = []
    poly_features = util.get_poly_feature(Xval[:, 0], 8)
    # cut first "ones" vector, after normalization we'll return it
    poly_features = util.mean_feature_normalization(poly_features[:, 1:], poly_features.shape[1] - 1)
    poly_features = np.insert(poly_features, 0, 1, axis=1)
    for currentLambda in range(LAMBDA_END):
        thetas, container, last_cost = logistic_regression.gradient_descent_with_reg(poly_features, Yval,
                                                                                     learning_rate=ALPHA,
                                                                                     iterations=200, lambda_=currentLambda,
                                                                                     save_results=True, add_ones=False)
        results.append(last_cost)
    return np.asarray(results).argmin(), np.asarray(results).min()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]
