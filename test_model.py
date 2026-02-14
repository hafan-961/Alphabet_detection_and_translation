import numpy as np
from src.preprocessing import load_function
from src.engine import forward_propagation

#Load data and weights
_, _, X_test, Y_test = load_function("data/emnist-letters-train.csv", "data/emnist-letters-test.csv")
parameters = np.load("weights/model_params.npy", allow_pickle=True).item()

#Predict on 100 samples
AL, _ = forward_propagation(X_test[:, :100], parameters, keep_prob=1.0, is_training=False)
predictions = np.argmax(AL, axis=0)
labels = np.argmax(Y_test[:, :100], axis=0)

accuracy = np.mean(predictions == labels) * 100
print(f"Model Accuracy on CSV Test Data: {accuracy}%")
print(f"Sample Predictions: {predictions[:10]}")
print(f"Actual Labels:      {labels[:10]}")