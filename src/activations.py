import numpy as np

def leaky_relu(Z, alpha=0.01):
    return np.where(Z > 0, Z, alpha * Z), Z

def softmax(Z):
    #numerical stability: shift Z by max
    shift_Z = Z - np.max(Z, axis=0, keepdims=True)
    exp_Z = np.exp(shift_Z)
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True), Z