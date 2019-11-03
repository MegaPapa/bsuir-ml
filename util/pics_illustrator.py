import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.image as mpimg

from util.file.matlab_file_reader import parse_dataset_file

TEMP_PIC_PATH = "/Users/max/Desktop/ex23.png"

def read_dataset(MATLAB_PICS_DATA):
    dataset, entries = parse_dataset_file(MATLAB_PICS_DATA, ['X', 'y'])
    features = entries['X']
    result = entries['y']
    features = np.insert(features, 0, 1, axis=1)
    return features, result


def get_image_from_row(row):
    width, height = 20, 20
    square = row[1:].reshape(width, height)
    return square.T


def display_random_data(features):
    width, height = 20, 20
    n_rows, n_cols = 10, 10
    indices_to_display = random.sample(range(features.shape[0]), n_rows * n_cols)

    big_picture = np.zeros((height * n_rows, width * n_cols))

    i_row, i_col = 0, 0
    for idx in indices_to_display:
        if i_col == n_cols:
            i_row += 1
            i_col = 0
        i_img = get_image_from_row(features[idx])
        big_picture[i_row * height:i_row * height + i_img.shape[0], i_col * width:i_col * width + i_img.shape[1]] = \
            i_img
        i_col += 1
    fig = plt.figure(figsize=(6, 6))
    plt.imsave(fname=TEMP_PIC_PATH, arr=big_picture, cmap=cm.Greys_r)

    img = mpimg.imread(TEMP_PIC_PATH)
    imgplot = plt.imshow(img)
    plt.show()
