import itertools
import numpy as np
import matplotlib.pyplot as plt


def get_markers():
    return itertools.cycle((".", "+", "o", "x", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*",
                           "h", "H", "X", "D", "d"))


def get_boundaries(feature, h=1):
    feature_min = np.min(feature) - h
    feature_max = np.max(feature) + h
    return feature_min, feature_max


def draw_clusters_with_traectory(x, indices, centers, centroids_history):
    x_1_min, x_1_max = get_boundaries(x[:, 0])
    plt.xlabel('X1')
    x_2_min, x_2_max = get_boundaries(x[:, 1])
    plt.ylabel('X2')
    plt.axis([x_1_min, x_1_max, x_2_min, x_2_max])
    plt.plot(x[:, 0], x[:, 1], 'o', marker=".", c=np.random.rand(3, ))
    plt.show()
    plt.close()

    markers = get_markers()
    x_1_min, x_1_max = get_boundaries(x[:, 0])
    x_2_min, x_2_max = get_boundaries(x[:, 1])
    plt.axis([x_1_min, x_1_max, x_2_min, x_2_max])
    plt.ylabel('X1')
    plt.xlabel('X2')
    plt.xlim(x_1_min, x_1_max)
    plt.ylim(x_2_min, x_2_max)
    centroids_amount = centers.shape[0]
    for centroid_index in range(0, centroids_amount):
        color = np.random.rand(3, )
        marker = next(markers)
        assigned_indices = np.where(np.any(indices == centroid_index, axis=1))[0]
        if assigned_indices.size > 0:
            x_1 = x[assigned_indices, 0]
            x_2 = x[assigned_indices, 1]
            plt.plot(x_1, x_2, 'o', marker=marker, c=color)

        centroid_history = centroids_history[:, centroid_index, :]
        centroid_history_x = centroid_history[:, 0]
        centroid_history_y = centroid_history[:, 1]
        plt.plot(centroid_history_x, centroid_history_y, marker=marker, c=np.random.rand(3, ))

    plt.show()
    plt.close()
