import itertools

import matplotlib.pyplot as plt
import numpy as np

from ml_8 import util


def get_markers():
    return itertools.cycle((".", "+", "o", "x", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*",
                            "h", "H", "X", "D", "d"))


def get_feature_boundaries(x, h=1.):
    feature_min = np.min(x) - h
    feature_max = np.max(x) + h
    return feature_min, feature_max


def plot_clusters(x, p, epsilon):
    x_1_min, x_1_max = get_feature_boundaries(x[:, 0], 1)
    x_2_min, x_2_max = get_feature_boundaries(x[:, 1], 1)
    plt.axis([x_1_min, x_1_max, x_2_min, x_2_max])
    non_anomaly_idx = np.where(p > epsilon)[0]
    anomaly_idx = np.where(p < epsilon)[0]
    plt.plot(x[non_anomaly_idx, 0], x[non_anomaly_idx, 1], 'o', marker=".", c=np.random.rand(3, ))
    plt.plot(x[anomaly_idx, 0], x[anomaly_idx, 1], 'o', marker="*", c=np.random.rand(3, ))
    plt.show()
    plt.close()

def plot_anomalies(x, mu, sigma, x_val, y_val):

    x_1_min, x_1_max = get_feature_boundaries(x[:, 0], 1)
    x_2_min, x_2_max = get_feature_boundaries(x[:, 1], 1)
    x_1_cv_min, x_1_cv_max = get_feature_boundaries(x_val[:, 0], 1)
    x_2_cv_min, x_2_cv_max = get_feature_boundaries(x_val[:, 1], 1)
    plt.axis([x_1_cv_min, x_1_cv_max, x_2_cv_min, x_2_cv_max])
    non_anomaly_idx = np.where(y_val == 0)[0]
    anomaly_idx = np.where(y_val == 1)[0]
    plt.plot(x_val[non_anomaly_idx, 0], x_val[non_anomaly_idx, 1], 'o', marker=".", c=np.random.rand(3, ))
    plt.plot(x_val[anomaly_idx, 0], x_val[anomaly_idx, 1], 'o', marker="*", c=np.random.rand(3, ))
    plt.show()
    plt.close()

    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.axis([x_1_min, x_1_max, x_2_min, x_2_max])
    # xx, yy = np.meshgrid(np.arange(0, 35.5, 0.3), np.arange(0, 35.5, 0.3))
    # z = util.gaussian_distribution_density(np.stack([xx.ravel(), yy.ravel()], axis=1), mu, sigma)
    # z = z.reshape(xx.shape)
    # if np.all(abs(z) != np.inf):
    #     plt.contour(xx, yy, z, levels=10 ** (np.arange(-20., 1, 3)), zorder=100)
    # plt.plot(x[:, 0], x_val[:, 1], 'o', marker=".", c=np.random.rand(3, ))
    # plt.show()
    # plt.close()
