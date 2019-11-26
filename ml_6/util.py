import numpy as np
from PIL import Image
from matplotlib import pyplot

from ml_6.k_mean import k_mean_algorithm


def read_image(file_path):
    image_data = pyplot.imread(file_path)
    return image_data


def translate_mat_to_compressed_img(img_data, colors_amount, iterations_amount, img_path):
    data_shape = img_data.shape
    img_to_save = Image.fromarray(img_data.astype('uint8'), 'RGB')
    img_to_save.save(img_path)
    img_data = img_data.reshape((data_shape[0] * data_shape[1], data_shape[2]))
    img_data = img_data / 255

    initial_centroids = init_centers(img_data, colors_amount)
    centroids, indices, centroids_history = k_mean_algorithm(img_data, initial_centroids, iterations_amount)
    classified_image_data = centroids[indices, :] * 255
    classified_image_data = classified_image_data.reshape(data_shape)

    img_to_save = Image.fromarray(classified_image_data.astype('uint8'), 'RGB')
    img_to_save.save(img_path + "_2.jpg")


def init_centers(x, centroids_count):
    randomly_ordered_indexes = np.random.permutation(x.shape[0])
    return x[randomly_ordered_indexes[0:centroids_count], :]