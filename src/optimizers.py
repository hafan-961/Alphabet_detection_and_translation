import numpy as np

def initialize_adam(parameters):
    v, s = {}, {}
    for key, val in parameters.items():
        v[key], s[key] = np.zeros_like(val), np.zeros_like(val)
    return v, s

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.0007):
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for key in parameters.keys():
        if "d"+key in grads:
            v[key] = beta1 * v[key] + (1 - beta1) * grads["d"+key]
            s[key] = beta2 * s[key] + (1 - beta2) * (grads["d"+key]**2)
            vc = v[key] / (1 - beta1**t)
            sc = s[key] / (1 - beta2**t)
            parameters[key] -= learning_rate * vc / (np.sqrt(sc) + eps)
    return parameters, v, s