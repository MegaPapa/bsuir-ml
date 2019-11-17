import numpy as np


def mean_feature_normalization(x, features_count):
    x_normalized = x.copy()
    x_normalized_i = np.hsplit(x_normalized, features_count)
    # traverse feature's data
    i = 0
    j = 0
    for x_normalized_i_features in x_normalized_i:
        x_mean = np.mean(x_normalized_i_features)
        x_max = np.max(x_normalized_i_features)
        x_min = np.min(x_normalized_i_features)
        for x_normalized_element in x_normalized_i_features:
            x_normalized[j][i] = (x_normalized_element - x_mean) / (x_max - x_min)
            j += 1
        i += 1
        j = 0
    return x_normalized


def get_poly_feature(feature, degree):
    poly_features = map_poly_feature(feature, degree)
    poly_features = np.insert(poly_features, 0, 1, axis=1)
    return poly_features


def map_poly_feature(feature, degree=6):
    poly_feature = np.zeros((feature.shape[0], degree))
    for i in range(1, degree + 1):
        poly_feature[:, i - 1] = feature ** i
    return poly_feature

def scale_parameters(parameters, has_bias_unit=True, enable_feature_scaling=True):
    scaled_parameters = parameters.copy()
    parameters_count = scaled_parameters.shape[1]
    parameters_meta = np.zeros((parameters_count, 3), float)

    parameter_index = 0
    while parameter_index < parameters_count:
        if (parameter_index == 0 and parameters_count > 1 and has_bias_unit) or enable_feature_scaling is False:
            parameters_meta[parameter_index] = np.array([0, 1, 0])
        else:
            parameters_meta[parameter_index, 0] = 1
            parameters_meta[parameter_index, 1] = np.mean(scaled_parameters[:, parameter_index])
            parameters_meta[parameter_index, 2] = np.std(scaled_parameters[:, parameter_index])
            scaled_parameters[:, parameter_index] = \
                (scaled_parameters[:, parameter_index] - parameters_meta[parameter_index, 1]) / \
                parameters_meta[parameter_index, 2]
        parameter_index = parameter_index + 1
    return scaled_parameters, parameters_meta


def modify_parameters(parameters, parameters_meta, scale_back=1):
    modified = parameters.copy()
    parameters_count = modified.shape[1]
    param_index = 0
    while param_index < parameters_count:
        meta = parameters_meta[param_index, :]
        if meta[0] != 0:
            if scale_back == 1:
                modified[:, param_index] = modified[:, param_index] * meta[2] + meta[1]
            else:
                modified[:, param_index] = (modified[:, param_index] - meta[1]) / meta[2]
        param_index = param_index + 1
    return modified
