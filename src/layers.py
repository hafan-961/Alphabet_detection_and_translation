import numpy as np

def He_initilization(layers_dims):
    parameters = {}
    L = len(layers_dims)

    #he initilization : sqrt(2/n_l-1)
    for l in range(1,L):
        #he initilization ()
        parameters['W' + str(l)] = np.random.randn(layers_dims[l] , layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))   

        # since we are using bath norm we initilize that also 
        parameters['gamma' + str(l)] = np.ones((layers_dims[l] , 1))
        parameters['beta' + str(l)] = np.zeros((layers_dims[l], 1))   

    return parameters  

def Batch_Norm_Forward(Z , gamma, beta, eps = 1e-8):
    mean = np.mean(Z , axis = 1 , keepdims = True)
    variance = np.var(Z ,axis = 1 , keepdims = True)
    Z_norm = (Z - mean) / np.sqrt(varaince + eps) 
    output  = gamma * Z_norm + beta
    cache =  (Z , Z_norm , mean , variance , gamma , beta)   #act as a memory while gradient learning

    return output , cache