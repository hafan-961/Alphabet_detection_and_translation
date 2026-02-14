import numpy as np

def He_initilization(layers_dims):
    np.random.seed(1)
    parameters = {}
    for l in range(1, len(layers_dims)):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b'+str(l)] = np.zeros((layers_dims[l], 1))
        parameters['gamma'+str(l)] = np.ones((layers_dims[l], 1))
        parameters['beta'+str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

def Batch_Norm_Forward(Z, gamma, beta, eps=1e-8):
    #bypass BN for from-scratch stability
    return Z, (Z, Z, 0, 0, gamma, beta)