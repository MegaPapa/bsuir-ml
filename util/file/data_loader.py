from os.path import join
import numpy as np


def read_file(file_path):
    text_file = open(file_path, "r")
    lines = text_file.readlines()
    text = ''
    for line in lines:
        text = text + line
    return text


def load_data(path, separator):
    result = []
    file_descriptior = open(path, "r")
    file_lines = file_descriptior.readlines()
    for line in file_lines:
        line_pieces = line.split()
        result.append(line_pieces[:])


def read_vocabulary(file_path):
    vocabulary = np.genfromtxt(join('Data', file_path), dtype=object)
    return list(vocabulary[:, 1].astype(str)), vocabulary[:, 1].shape[0]
