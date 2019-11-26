import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram


def hierarchical_clustering(image_data, max_clusters_amount_to_show):
    s = image_data.shape
    reshaped_custom_image_data = image_data.reshape((s[0] * s[1], s[2]))
    reshaped_custom_image_data = reshaped_custom_image_data / 255
    clusters = linkage(reshaped_custom_image_data, 'centroid')
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('sample index')
    plt.ylabel('distance')
    dendrogram(
        clusters,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=max_clusters_amount_to_show,  # show only the last p merged clusters
        leaf_rotation=90.,  # rotates the x axis labels
        leaf_font_size=12,  # font size for the x axis labels
    )
    plt.show()

    return clusters

def k_mean_algorithm(x, centers, max_iterations):
    indices = np.zeros((x.shape[0], 1))
    centroids_history = np.zeros((max_iterations, centers.shape[0], centers.shape[1]))
    centroids = centers.copy()
    for iteration in range(0, max_iterations):
        indices = find_closest_center(x, centroids)
        centroids_history[iteration] = centroids
        centroids = reassign_centres(x, indices, centroids)
    return centroids, indices, centroids_history


def find_closest_center(x, centers):
    centroids_amount = centers.shape[0]
    distances = np.zeros((centroids_amount, x.shape[0]))
    for centroid_index in range(0, centroids_amount):
        distances[centroid_index, :] = np.sqrt(np.sum((x - centers[centroid_index, :]) ** 2, axis=1))
    return np.argmin(distances, axis=0).reshape((x.shape[0], 1))


def reassign_centres(x, current_indices, centers):
    centroids_amount = centers.shape[0]
    for centroid_index in range(0, centroids_amount):
        assigned_features = x[np.where(np.any(current_indices == centroid_index, axis=1)), :]
        if assigned_features.size > 0:
            centers[centroid_index, :] = np.mean(assigned_features, axis=1)
    return centers