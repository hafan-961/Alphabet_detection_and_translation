import pandas as pd
import numpy as np

def load_function(train_path, test_path):
    print("Loading and shuffling data...")
    #load and shuffle immediately
    train_df = pd.read_csv(train_path, header=None).sample(frac=1).reset_index(drop=True)
    test_df = pd.read_csv(test_path, header=None).sample(frac=1).reset_index(drop=True)
    
    Y_train_raw = train_df.iloc[:, 0].values - 1
    X_train_raw = train_df.iloc[:, 1:].values / 255.0
    Y_test_raw = test_df.iloc[:, 0].values - 1
    X_test_raw = test_df.iloc[:, 1:].values / 255.0

    def fix_orientation(X):
        #eminist is stored transposed ,  This flips it to be upright.
        X = X.reshape(-1, 28, 28)
        X = np.transpose(X, (0, 2, 1)) 
        return X.reshape(-1, 784)

    X_train = fix_orientation(X_train_raw).T # (784, m)
    X_test = fix_orientation(X_test_raw).T
    
    Y_train = np.eye(26)[Y_train_raw.astype(int)].T # (26, m)
    Y_test = np.eye(26)[Y_test_raw.astype(int)].T
    
    return X_train, Y_train, X_test, Y_test