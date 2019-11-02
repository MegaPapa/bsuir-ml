import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm

from util.algorithms import logistic_regression
from util.algorithms.linear_regression import calc_cost_function


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
    plt.plot(*zip(*thetas), "r-")
    plt.show()


def show_gradient_descent_steps_in_3d(thetas, cost_function_results):
    fig = plt.figure()
    ax = Axes3D(fig)
    xs = np.asarray([])
    ys = np.asarray([])
    zs = np.asarray(cost_function_results)
    for theta1, theta2 in thetas:
        xs = np.append(xs, theta1[0])
        ys = np.append(ys, theta2[0])
    ax.plot(xs, ys, zs)
    plt.show()


def show_points_by_classes(points_with_class):
    for i in range(len(points_with_class)):
        data_to_show = np.zeros(shape=(len(points_with_class[i]), 2))
        for j in range(len(points_with_class[i])):
            data_to_show[j][0] = points_with_class[i][j].get_first_exam()
            data_to_show[j][1] = points_with_class[i][j].get_second_exam()
        plt.plot(*zip(*data_to_show), "x")
        plt.xlabel("First exam")
        plt.ylabel("Second exam")
    plt.show()


def create_matrix(x1, x2):
    ones = np.ones((1, 2))
    ones[0][0] = x1
    ones[0][1] = x2
    return ones


def draw_decision_boundary_line(thetas, points_with_classes):
    x1 = 30
    x2 = 100
    x1_points = np.asarray([])
    x2_points = np.asarray([])
    while x1 != 100:
        xs = create_matrix(x1, x2)
        prediction = logistic_regression.predict(xs, thetas)
        if prediction >= 0.5:
            x1_points = np.append(x1_points, x1)
            x2_points = np.append(x2_points, x2)
            while prediction >= 0.5:
                x2 -= 1
                xs = create_matrix(x1, x2)
                prediction = logistic_regression.predict(xs, thetas)
        else:
            x1 += 1
    ###
    data_to_show = np.zeros(shape=(len(x1_points), 2))
    for j in range(len(x1_points)):
        data_to_show[j][0] = x1_points[j]
        data_to_show[j][1] = x2_points[j]
    plt.plot(*zip(*data_to_show), "r-")
    plt.xlabel("First exam")
    plt.ylabel("Second exam")
    show_points_by_classes(points_with_classes)
    plt.show()
    # result = logistic_regression.predict(xs, thetas)
    # print(result)
