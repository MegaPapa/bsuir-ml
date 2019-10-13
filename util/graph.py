import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import math

from util.linear_regression import calc_cost_function


def show_3d_plot_for_cost_function(thetas, cost_function_results, x, y):
    fig = plt.figure()
    ax = Axes3D(fig)
    m = len(x)
    x_with_ones = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    y_with_ones = np.concatenate((np.ones((len(y), 1)), y), axis=1)
    xs = np.asarray([])
    ys = np.asarray([])
    zs = np.asarray(cost_function_results)
    for theta1, theta2 in thetas:
        xs = np.append(xs, theta1[0])
        ys = np.append(ys, theta2[0])
    num_of_linspaced_points = 100
    linspaced_theta_1 = np.linspace(np.min(xs), np.max(xs))
    linspaced_theta_2 = np.linspace(np.min(ys), np.max(ys))
    theta_x_grid, theta_y_grid = np.meshgrid(linspaced_theta_1, linspaced_theta_2)
    cost_functions_grid = np.zeros((len(linspaced_theta_1), len(linspaced_theta_2)))
    i = 0
    j = 0
    while i < len(linspaced_theta_1):
        while j < len(linspaced_theta_2):
            tmp_thetas = np.array([[theta_x_grid[i, j]], [theta_y_grid[i, j]]])
            cost_functions_grid[j, i], loss = calc_cost_function(x_with_ones, y_with_ones, m, tmp_thetas)
            j += 1
        j = 0
        i += 1

    plt.xlabel("Theta 0")
    plt.ylabel("Theta 1")
    ax.plot_surface(theta_x_grid, theta_y_grid, cost_functions_grid.T, cmap=cm.coolwarm)
    plt.show()


def show_plot_by_points(points):
    plt.plot(*zip(*points), "x")
    plt.ylabel("Profit")
    plt.xlabel("Population")
    plt.show()


def show_plot_by_graph(points):
    plt.plot(*zip(*points), "r-")
    plt.ylabel("Profit")
    plt.xlabel("Population")
    plt.show()


def show_plot_by_graph_and_points(points, graph):
    plt.plot(*zip(*points), "x")
    plt.plot(*zip(*graph), "r-")
    plt.ylabel("Profit")
    plt.xlabel("Population")
    plt.show()


def show_contour_plot_for_cost(thetas, cost_function_results, x, y):
    m = len(x)
    x_with_ones = np.concatenate((np.ones((len(x), 1)), x), axis=1)
    y_with_ones = np.concatenate((np.ones((len(y), 1)), y), axis=1)
    xs = np.asarray([])
    ys = np.asarray([])
    zs = np.asarray(cost_function_results)
    for theta1, theta2 in thetas:
        xs = np.append(xs, theta1[0])
        ys = np.append(ys, theta2[0])
    num_of_linspaced_points = 100
    linspaced_theta_1 = np.linspace(-10, 10, num_of_linspaced_points)
    linspaced_theta_2 = np.linspace(-1, 4, num_of_linspaced_points)
    theta_x_grid, theta_y_grid = np.meshgrid(linspaced_theta_1, linspaced_theta_2)
    cost_functions_grid = np.zeros((len(linspaced_theta_1), len(linspaced_theta_2)))
    i = 0
    j = 0
    while i < len(linspaced_theta_1):
        while j < len(linspaced_theta_2):
            tmp_thetas = np.array([[theta_x_grid[i, j]], [theta_y_grid[i, j]]])
            cost_functions_grid[j, i], loss = calc_cost_function(x_with_ones, y_with_ones, m, tmp_thetas)
            j += 1
        j = 0
        i += 1


    plt.xlabel("Theta 0")
    plt.ylabel("Theta 1")
    plt.xlim(-10, 10)
    plt.ylim(-1, 4)
    plt.contour(linspaced_theta_1, linspaced_theta_2, cost_functions_grid.T, np.logspace(-2, 3, 20))
    plt.show()
