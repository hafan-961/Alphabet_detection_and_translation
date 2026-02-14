import numpy as np
from src.activations import leaky_relu, softmax
from src.layers import Batch_Norm_Forward
from src.regularization import dropout_forward, dropout_backward

def forward_propagation(X, parameters, keep_prob, is_training=True):
    caches = []
    A = X
    L = len(parameters) // 4
    for l in range(1, L):
        Z = np.dot(parameters['W'+str(l)], A) + parameters['b'+str(l)]
        Z_bn, bn_cache = Batch_Norm_Forward(Z, parameters['gamma'+str(l)], parameters['beta'+str(l)])
        A_next, act_cache = leaky_relu(Z_bn)
        mask = None
        if is_training and keep_prob < 1:
            A_next, mask = dropout_forward(A_next, keep_prob)
        caches.append((A, Z_bn, mask, bn_cache, act_cache))
        A = A_next
    
    ZL = np.dot(parameters['W'+str(L)], A) + parameters['b'+str(L)]
    AL, _ = softmax(ZL)
    caches.append((A, ZL, None, None, ZL)) # Final cache
    return AL, caches

def backward_propagation(AL, Y, caches, parameters, keep_prob, lambd):
    grads = {}
    L = len(caches)
    m = Y.shape[1]
    
    #output Layer
    dZ = AL - Y
    A_prev_L = caches[L-1][0]
    grads["dW"+str(L)] = (1/m) * np.dot(dZ, A_prev_L.T) + (lambd/m) * parameters["W"+str(L)]
    grads["db"+str(L)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)

    #hidden layers
    for l in reversed(range(L-1)):
        dA = np.dot(parameters["W"+str(l+2)].T, dZ)
        A_prev, Z_bn, mask, bn_cache, act_cache = caches[l]
        if keep_prob < 1 and mask is not None:
            dA = dropout_backward(dA, mask, keep_prob)
        
        # Leaky ReLU Derivative
        dZ = np.array(dA, copy=True)
        dZ[act_cache <= 0] *= 0.01
        
        grads["dW"+str(l+1)] = (1/m) * np.dot(dZ, A_prev.T) + (lambd/m) * parameters["W"+str(l+1)]
        grads["db"+str(l+1)] = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        grads["dgamma"+str(l+1)] = np.zeros_like(parameters["gamma"+str(l+1)])
        grads["dbeta"+str(l+1)] = np.zeros_like(parameters["beta"+str(l+1)])
        
    return grads