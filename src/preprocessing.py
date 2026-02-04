import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# df = pd.read_csv("data/emnist-letters-train.csv")
# num_rows = df.shape
# print(num_rows)
def load_function(train_path , test_path):
    train = pd.read_csv(train_path , header = None)
    test = pd.read_csv(test_path, header = None)
    
    #Seprating lables and pixel of both train and test set
    X_train_origin = train.iloc[:,1:].values / 255.0 
    Y_train_origin = train.iloc[:,0].values -1              #the data is from 1-26 , but we want 0-25 as it is good for indexing for ML
            
   
    X_test_origin = test.iloc[:,1:].values / 255.0
    Y_test_origin = test.iloc[:,0].values -1

    X_train = X_train_origin.T
    X_test = X_test_origin.T

    #now applying one-hot encoding 
    Y_train = np.eye(26)[Y_train_origin].T
    Y_test = np.eye(26)[Y_test_origin].T

    return X_train , Y_train , X_test , Y_test

train_path = "data/emnist-letters-train.csv"
test_path = "data/emnist-letters-test.csv"

X_train, Y_train, X_test, Y_test = load_function(train_path, test_path)


print("Min pixel:", X_train.min())
print("Max pixel:", X_train.max())



# data = X_train.flatten()

# Q1 = np.percentile(data, 25)
# Q3 = np.percentile(data, 75)
# IQR = Q3 - Q1

# lower = Q1 - 1.5 * IQR
# upper = Q3 + 1.5 * IQR

# outliers = data[(data < lower) | (data > upper)]

# print("Outlier pixel count:", len(outliers))
# print("Lower bound:", lower)
# print("Upper bound:", upper)

# df = pd.read_csv("data/emnist-letters-train.csv", header=None)
# pixels = df.iloc[:, 1:].values   # remove label column
# plt.boxplot(pixels.flatten())

# plt.title("simple box plot")
# plt.show()








