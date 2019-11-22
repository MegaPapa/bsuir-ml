def read_datasets(path):
    dataset, entries = parse_dataset_file(PATH_TO_DATASET_FILE, ['X', 'y'])
    x = entries['X']
    y = entries['y']