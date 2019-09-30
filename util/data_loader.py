def load_data(path, separator):
    result = []
    file_descriptior = open(path, "r")
    file_lines = file_descriptior.readlines()
    for line in file_lines:
        line_pieces = line.split()
        result.append(line_pieces[:])
