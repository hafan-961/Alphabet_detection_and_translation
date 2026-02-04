import numpy as np

def initialize_parameters(parameters):
    L =  len(parameters)//4
    v = {}
    s = {}
    for l in range (1, L+1):
        for param in ['W' , 'b' , 'gamma' , 'beta']:
            v[param + str(l)] = np.zeros(parameters[param + str(l)].shape)
            s[param + str(l)] = np.zeros(parameters[param + str(l)].shape)

    return v,s


def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.001,
beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    L = len(parameters) // 4
    for l in range(1, L + 1):
        for p in ['W' , 'b', 'gamma' , 'beta']:
            #Momentum
            v[p+str(l)] = beta1 * v[p+str(l)] + (1-beta1) * grads['d' + p + str(l)]
            v_corrected  = v[p+str(l)] / (1 - beta1**t)

            #Rmsprop
            s[p+str(l)] = beta2 *  s[p+str(l)] + (1-beta2) * (grads['d' + p + str(l)]**2)
            s_corrected = s[p+str(l)] / (1 - beta2**t)

            #update
            parameters[p+str(l)] -= learning_rate * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return parameters , v, s



