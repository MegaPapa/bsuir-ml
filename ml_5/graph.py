import matplotlib.pyplot as plt
import itertools
import numpy as np


def create_plot_for_features(x, y, h=0.01):
    markers = get_markers()
    unique_results = np.unique(y)
    x_1_min, x_1_max = get_feature_boundaries(x[:, 0], h)
    x_2_min, x_2_max = get_feature_boundaries(x[:, 1], h)
    plt.axis([x_1_min, x_1_max, x_2_min, x_2_max])
    plt.ylabel('X1')
    plt.xlabel('X2')
    plt.xlim(x_1_min, x_1_max)
    plt.ylim(x_2_min, x_2_max)
    for unique_result in unique_results:
        unique_result_indexes = np.where(y == unique_result)
        x_1 = x[unique_result_indexes, 0]
        x_2 = x[unique_result_indexes, 1]
        plt.plot(x_1, x_2, 'o', marker=next(markers), c=np.random.rand(3, ))

    return x_1_min, x_1_max, x_2_min, x_2_max, markers


def draw_plot_without_decision_boundary(x, y, axes_modifier=0.5):
    create_plot_for_features(x, y, axes_modifier)

    plt.show()
    plt.close()


def draw_plot_with_decision_boundary(x, y, prediction_function, support_vectors,
                                      axes_modifier=0.5, draw_kernel_points=False, h=0.01):
    x_1_min, x_1_max, x_2_min, x_2_max, markers = create_plot_for_features(x, y, axes_modifier)

    xx, yy = np.meshgrid(np.arange(x_1_min, x_1_max, h), np.arange(x_2_min, x_2_max, h))
    z = prediction_function(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired)
    if draw_kernel_points:
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], c='k', marker=next(markers), s=100, linewidths='1')

    plt.show()

    plt.close()


def get_markers():
    return itertools.cycle(("+", "o"))


def get_feature_boundaries(x, h=0.01):
    feature_min = np.min(x) - h
    feature_max = np.max(x) + h
    return feature_min, feature_max


def draw_plot(x, y, model_1, model_100):
    draw_plot_without_decision_boundary(x, y)
    draw_plot_with_decision_boundary(x, y, lambda v: model_1.predict(v),
                                     model_1.support_vectors_, 1, draw_kernel_points=True)
    draw_plot_with_decision_boundary(x, y, lambda v: model_100.predict(v),
                                     model_100.support_vectors_, 100, draw_kernel_points=True)