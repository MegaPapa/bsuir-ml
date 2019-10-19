import scipy.io

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