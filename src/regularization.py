import numpy as np

def compute_cost_with_L2(AL, Y, parameters, lambd):
    m = Y.shape[1]
    AL = np.clip(AL, 1e-10, 1.0 - 1e-10)
    cross_entropy = -1/m * np.sum(Y * np.log(AL))
    
    l2_cost = 0
    L = len(parameters) // 4
    for l in range(1, L + 1):
        l2_cost += np.sum(np.square(parameters['W' + str(l)]))
    
    return cross_entropy + (lambd / (2 * m)) * l2_cost

def dropout_forward(A, keep_prob):
    mask = np.random.rand(*A.shape) < keep_prob
    return (A * mask) / keep_prob, mask

def dropout_backward(dA, mask, keep_prob):
    return (dA * mask) / keep_prob