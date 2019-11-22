import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


def get_image_from_row(row):
    width, height = 20, 20
    square = row[1:].reshape(width, height)
    return square.T


def display_random_data(x, name, n_rows=10, n_cols=10):
    width, height = 20, 20
    x = np.insert(x, 0, 1, axis=1)
    indices_to_display = random.sample(range(x.shape[0]), n_rows * n_cols)

    big_picture = np.zeros((height * n_rows, width * n_cols))

    i_row, i_col = 0, 0
    for idx in indices_to_display:
        if i_col == n_cols:
            i_row += 1
            i_col = 0
        i_img = get_image_from_row(x[idx])
        big_picture[i_row * height:i_row * height + i_img.shape[0], i_col * width:i_col * width + i_img.shape[1]] = \
            i_img
        i_col += 1
    img = Image.fromarray(np.uint8(big_picture * 255))
    img.show()
