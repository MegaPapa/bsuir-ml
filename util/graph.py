import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import math


def show_3d_plot_for_cost_function(thetas, cost_function_results):
    fig = plt.figure()
    ax = Axes3D(fig)
    fig = plt.figure()
    xs = []
    ys = []
    zs = np.asarray(cost_function_results)
    for theta1, theta2 in thetas:
        xs.append(theta1[0])
        ys.append(theta2[0])

    # ax.plot_surface()
    ax.plot_trisurf(ys, xs, zs, cmap=cm.coolwarm, linewidth=0, antialiased=False)


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


def show_contour_plot_for_cost(thetas, cost_function_results):
    xs = []
    ys = []
    zs = np.asarray(cost_function_results)
    for theta1, theta2 in thetas:
        xs.append(theta1[0])
        ys.append(theta2[0])
    plt.contour(xs, ys, zs, cmap='RdGy')
