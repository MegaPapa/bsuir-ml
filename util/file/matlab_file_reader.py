import scipy.io
import numpy as np

from util.logger import LoggerBuilder

logger = LoggerBuilder().with_name("file_reader").build()


def read_matlab_file(path):
    dataset = scipy.io.loadmat(path)
    logger.debug("Load matlab file %s", path)

    for (key, value) in dataset.items():
        if '__' not in key:
            logger.debug('\tFound entry \'%s\' with size %s', key,
                         value.shape if hasattr(value, 'shape') else len(value))
    return dataset


def parse_dataset_file(dataset_file_path, expected_labels=None):
    if ".txt" in dataset_file_path:
        dataset = read_dataset_txt_file(dataset_file_path)
        features = np.ones(dataset.shape)
        features[:, 1:] = dataset[:, 0:-1]
        results = dataset[:, -1:]
        return dataset, features, results
    if ".mat" in dataset_file_path:
        dataset = read_matlab_file(dataset_file_path)
        if expected_labels is None:
            return dataset
        dataset_entries = {}
        for label in expected_labels:
            dataset_entries[label] = dataset.get(label)
        return dataset, dataset_entries

    return None


def read_dataset_txt_file(file_path):
    text_file = open(file_path, "r")
    lines = text_file.readlines()
    text_file.close()
    parsed_lines = []
    for line in lines:
        parsed_lines.append(line.replace('\n', '').split(','))
    dataset = np.array(parsed_lines, 'float64')
    return dataset