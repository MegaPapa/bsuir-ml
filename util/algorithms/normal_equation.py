import numpy as np


# x here is
# [ 1 x1
#   1 x2
#   ...
#   1 xn ]
from util.timed import timed


@timed
def calc_normal_equation(x, y):
    x_with_ones = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    y_with_ones = np.concatenate((np.ones((len(y), 1)), y), axis=1)
    x_transp = x_with_ones.T
    x_transp_x = x_with_ones.T @ x_with_ones
    inv = np.linalg.pinv(x_transp_x)
    thetas = (inv @ x_transp) @ y_with_ones
    return thetas
