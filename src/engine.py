import numpy as np
from src.activations import leaky_relu, leaky_relu_backward, softmax, softmax_backward
from src.layers import batchnorm_forward
from src.regularization import dropout_forward, dropout_backward

def forward_propagation(X, parameters, keep_prob, is_training=True):
    caches = []
    A = X
    L = len(parameters) // 4
    
    for l in range(1, L):
        A_prev = A
        #linear -> BN -> activation -> dropout
        Z = np.dot(parameters['W'+str(l)], A_prev) + parameters['b'+str(l)]
        Z_bn, bn_cache = batchnorm_forward(Z, parameters['gamma'+str(l)], parameters['beta'+str(l)])
        A, activation_cache = leaky_relu(Z_bn)
        
        mask = None
        if is_training and keep_prob < 1:
            A, mask = dropout_forward(A, keep_prob)
            
        caches.append(((A_prev, Z, mask, bn_cache, activation_cache)))

    #output Layer: Linear -> Softmax
    ZL = np.dot(parameters['W'+str(L)], A) + parameters['b'+str(L)]
    AL, cache_L = softmax(ZL)
    caches.append((A, ZL, None, None, cache_L))
    
    return AL, caches

def backward_propagation(AL, Y, caches, parameters, keep_prob, lambd):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    
    # 1. Output Layer Gradients
    dZL = softmax_backward(Y, AL)
    A_prev_L, _, _, _, _ = caches[L-1]
    grads["dW"+str(L)] = (1/m) * np.dot(dZL, A_prev_L.T) + (lambd/m) * parameters["W"+str(L)]
    grads["db"+str(L)] = (1/m) * np.sum(dZL, axis=1, keepdims=True)
    dA_prev = np.dot(parameters["W"+str(L)].T, dZL)

    # 2. Hidden Layers Gradients (Loop backwards)
    for l in reversed(range(L-1)):
        A_prev, Z, mask, bn_cache, act_cache = caches[l]
        
        if keep_prob < 1:
            dA_prev = dropout_backward(dA_prev, mask, keep_prob)
            
        dZ_bn = leaky_relu_backward(dA_prev, act_cache)
        
        # simplified BN backprop for dW and db
        grads["dW"+str(l+1)] = (1/m) * np.dot(dZ_bn, A_prev.T) + (lambd/m) * parameters["W"+str(l+1)]
        grads["db"+str(l+1)] = (1/m) * np.sum(dZ_bn, axis=1, keepdims=True)
        grads["dgamma"+str(l+1)] = (1/m) * np.sum(dZ_bn * bn_cache[1], axis=1, keepdims=True)
        grads["dbeta"+str(l+1)] = (1/m) * np.sum(dZ_bn, axis=1, keepdims=True)
        
        dA_prev = np.dot(parameters["W"+str(l+1)].T, dZ_bn)

    return grads