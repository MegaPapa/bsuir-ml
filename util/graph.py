import matplotlib.pyplot as plt
import math


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