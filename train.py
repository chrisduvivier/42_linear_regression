import numpy as np
import pandas as pd
import sys
import os.path
from linear_regression import *

def load_dataset(path):
    if not os.path.exists(path):
        print("Error: dataset is missing")
        exit()
    df = pd.read_csv(path)
    return df

def store_normalization_data(path, norm_mean, norm_std):
    """
    store mean and std of original training dataset
    """
    # read the csv file
    df = load_dataset(path)
  
    # updating the column value/data
    df.loc[0, 'normalization_mean'] = norm_mean
    df.loc[0, 'normalization_std'] = norm_std
  
    # writing into the file
    df.to_csv(path, index=False)

def normalize_input(x, norm_mean, norm_std):
    """
    apply same transformation done on training input X, only if params are non 0
    """
    if norm_std != 0:
        return (x - norm_mean) / norm_std
    return x

if __name__ == "__main__":
    
    df = load_dataset("data/data.csv")
    
    x_train = np.array(df["km"])
    y_train = np.array(df["price"])

    norm_mean = x_train.mean()
    norm_std = x_train.std()
    store_normalization_data("data/normalization.csv", norm_mean, norm_std)
    x_train = normalize_input(x_train, norm_mean, norm_std)

    Model = LinearRegressionModel()
    
    # initialize parameters
    thetas = load_dataset("data/parameters.csv")
    w_init = np.array(thetas["theta0"])
    b_init = np.array(thetas["theta1"])

    # parameters for gradient descent
    iterations = 10000
    alpha = 1.0e-2 # 0.01

    # run gradient descent
    w_final, b_final = Model.gradient_descent(x_train ,y_train, w_init, b_init, alpha, 
                                                    iterations, Model.compute_cost, Model.compute_gradient)
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

    
    