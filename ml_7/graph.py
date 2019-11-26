import itertools

import matplotlib.pyplot as plt
import numpy as np


def get_markers():
    return itertools.cycle((".", "+", "o", "x", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*",
                           "h", "H", "X", "D", "d"))


def get_feature_boundaries(feature, h=1.):
    feature_min = np.min(feature) - h
    feature_max = np.max(feature) + h
    return feature_min, feature_max


def plot_loaded_data(x, meta, u, s, x_scaled, projected_data, x_recovered):
    x_1_min, x_1_max = get_feature_boundaries(x[:, 0], 1)
    plt.xlabel('X1')
    x_2_min, x_2_max = get_feature_boundaries(x[:, 1], 1)
    plt.ylabel('X2')
    plt.axis([x_1_min, x_1_max, x_2_min, x_2_max])
    plt.plot(x[:, 0], x[:, 1], 'o', marker=".", c=np.random.rand(3, ))
    xy = np.zeros((4, 2))
    xy[0, :] = meta[:, 1].T
    xy[1, :] = meta[:, 1] + 1.5 * s[0] * u[:, 0]
    plt.plot(xy[0:2, 0], xy[0:2, 1], linestyle='-')
    xy[2, :] = meta[:, 1].T
    xy[3, :] = meta[:, 1] + 1.5 * s[1] * u[:, 1]
    plt.plot(xy[2:4, 0], xy[2:4, 1], linestyle='-')
    plt.show()
    plt.close()

    x_1_min, x_1_max = get_feature_boundaries(x_scaled[:, 0], h=0.5)
    plt.xlabel('X1')
    x_2_min, x_2_max = get_feature_boundaries(x_scaled[:, 1], h=0.5)
    plt.ylabel('X2')
    plt.axis([x_1_min, x_1_max, x_2_min, x_2_max])
    plt.plot(x_scaled[:, 0], x_scaled[:, 1], 'o', marker=".", c=np.random.rand(3, ))
    plt.plot(x_recovered[:, 0], x_recovered[:, 1], 'o', linestyle='-', marker="o", c=np.random.rand(3, ))
    projection_connector_color = np.random.rand(3, )
    for i in range(0, x_scaled.shape[0]):
        xy = np.zeros((2, 2))
        xy[0, 0] = x_scaled[i, 0]
        xy[0, 1] = x_scaled[i, 1]
        xy[1, 0] = x_recovered[i, 0]
        xy[1, 1] = x_recovered[i, 1]
        plt.plot(xy[:, 0], xy[:, 1], linestyle='dashed', c=projection_connector_color)
    plt.show()
    plt.close()


def plot_2d(x):
    x_1 = []
    x_2 = []
    for x_container in x:
        x_1.append(x_container[0])
        x_2.append(x_container[1])
    plt.plot(x_1, x_2, "x")
    plt.show()
    plt.close()