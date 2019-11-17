import matplotlib.pyplot as plt
import numpy as np

from ml_3 import logistic_regression, util


def show_graphic_by_thetas(thetas, x, y, x_start=0, x_end=0, additional_concatenation=False):
    linspaced_values = np.linspace(x_start, x_end, x_end - x_start)
    linspaced_values = linspaced_values.reshape((len(linspaced_values), 1))
    ones = np.ones((len(linspaced_values), 1))
    predictable_points = np.concatenate((ones, linspaced_values), axis=1)
    if additional_concatenation:
        predicted_points = logistic_regression.linear_hypotesis(predictable_points, thetas)
    else:
        poly_features = util.get_poly_feature(linspaced_values[:, 0], 7)
        # cut first "ones" vector, after normalization we'll return it
        poly_features = util.mean_feature_normalization(poly_features[:, 1:], poly_features.shape[1] - 1)
        poly_features = np.insert(poly_features, 0, 1, axis=1)
        predicted_points = logistic_regression.linear_hypotesis(poly_features, thetas)
    plt.ylabel("Water changes")
    plt.xlabel("Count of fluided water")
    plt.plot(x, y, 'bo')
    plt.plot(linspaced_values, predicted_points, 'r-')
    plt.show()



def show_plot_by_points(x, y):
    plt.plot(x, y, 'bo')
    plt.ylabel("Water changes")
    plt.xlabel("Count of fluided water")
    plt.show()


def show_loss_curve(points_count, containers):
    linspaced_values = np.linspace(1, points_count, points_count)
    loss_on_point = []
    for i in range(points_count):
        loss_on_point.append(np.sum(containers[i].get_loss_container()))
    plt.ylabel("Loss")
    plt.xlabel("Count of examples")
    plt.plot(linspaced_values, np.asarray(loss_on_point), 'r-')
    plt.show()


def show_learning_curve(points_count, containers):
    linspaced_values = np.linspace(1, points_count, points_count)
    plt.ylabel("Cost")
    plt.xlabel("Count of examples")
    plt.plot(linspaced_values, np.asarray(containers), 'r-')
    plt.show()