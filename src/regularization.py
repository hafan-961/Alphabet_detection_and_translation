import numpy as np 

def compute_cost_with_L2(AL , Y , parameters , lambd):
    m = Y.shape[1]
    cross_entropy_cost = -1/m * np.sum(Y * np.log(AL + 1e-8))

    # L2 penalty: (lambda / 2m) * sum of squares of all weights
    l2_cost = 0
    L = len(parameters) // 4
    for l in range(1, L+1):
        l2_cost += np.sum(np.square(parameters['W' + str(l)]))

    return cross_entropy_cost + (lambd / (2*m)) * l2_cost

def dropout_forward(A , keep_prob):
    mask = np.random.rand(A.shape[0] , A.shape[1]) < keep_prob
    A = (A * mask) / keep_prob
    return A , mask

def dropout_backward(dA , mask , keep_prob):
    dA = (dA * mask) / keep_prob
    return dA
    