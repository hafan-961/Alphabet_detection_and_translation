import numpy as np

def leaky_relu(Z , alpha = 0.01):
    A = np.where(Z > 0 , Z , alpha * Z)
    return A , Z

def leaky_relu_backard(dA , cache , alpha = 0.01):
    Z = cache
    dZ  = np.where(Z > 0, 1 ,alpha)
    return dA * dZ

def softmax(Z):
    shift_Z = Z - np.max(Z , axis = 0 , Keepdims = True)
    exp_Z = np.exp(shift_Z)
    A = exp_Z - np.sum(exp_Z, axis = 0, Keepdims = True)
    return A , Z

def softmax_backward(Y,AL):
    return Y,AL


