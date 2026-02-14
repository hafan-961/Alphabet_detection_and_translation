import numpy as np
from src.preprocessing import load_function 
from src.layers import He_initilization 
from src.optimizers import initialize_adam, update_parameters_with_adam
from src.engine import forward_propagation, backward_propagation 
from src.regularization import compute_cost_with_L2

X_train, Y_train, X_test, Y_test = load_function("data/emnist-letters-train.csv", "data/emnist-letters-test.csv")

layers_dims = [784, 512, 256, 26]
params = He_initilization(layers_dims)
v, s = initialize_adam(params)
t = 0

print("Starting training...")
for i in range(25):
    perm = np.random.permutation(X_train.shape[1])
    X_s, Y_s = X_train[:, perm], Y_train[:, perm]
    
    for j in range(0, X_train.shape[1], 128):
        t += 1
        xb, yb = X_s[:, j:j+128], Y_s[:, j:j+128]
        al, caches = forward_propagation(xb, params, 0.8)
        grads = backward_propagation(al, yb, caches, params, 0.8, 0.01)
        params, v, s = update_parameters_with_adam(params, grads, v, s, t, 0.0007)
    
    al_t, _ = forward_propagation(X_test[:, :2000], params, 1.0, False)
    acc = np.mean(np.argmax(al_t, axis=0) == np.argmax(Y_test[:, :2000], axis=0))
    print(f"Epoch {i} | Accuracy: {acc*100:.2f}%")

np.save("weights/model_params.npy", params)