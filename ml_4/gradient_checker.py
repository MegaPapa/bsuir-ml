from ml_4.neural_network_utils import debug_weights
import numpy as np

def check_gradients(cost_function, lambda_value=0):
    input_layer_size = 3
    hidden_layer_size = 5
    output_layer_size = 3
    m = 5

    theta_2 = debug_weights(hidden_layer_size, input_layer_size)
    theta_1 = debug_weights(output_layer_size, hidden_layer_size)
    theta = np.concatenate([theta_1.ravel(), theta_2.ravel()])

    x = debug_weights(m - 1, input_layer_size).T
    y = 1 + np.remainder(np.arange(0, m), output_layer_size).reshape((m, 1))

    backprop_cost, backprop_gradient = cost_function(x, y, lambda_value,
                                                     theta_1=theta_1,
                                                     theta_2=theta_2,
                                                     num_labels=3)

    numerical_gradient = np.zeros(theta.shape)
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(0, theta.size):
        perturb[p] = e
        loss1, g1 = cost_function(x, y, lambda_value, big_thetas=(theta - perturb), num_labels=3)
        loss2, g2 = cost_function(x, y, lambda_value, big_thetas=(theta - perturb), num_labels=3)
        numerical_gradient[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    d = np.linalg.norm(numerical_gradient - backprop_gradient) / np.linalg.norm(numerical_gradient + backprop_gradient)

    gradients = np.zeros((numerical_gradient.shape[0], 2))
    gradients[:, 0] = numerical_gradient
    gradients[:, 1] = backprop_gradient
    # print(gradients)

    return d, gradients