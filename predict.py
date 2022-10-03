import numpy as np
import pandas as pd
import sys
import os.path
from linear_regression import *

def check_user_input(input):
    try:
        # Convert it into integer
        val = int(input)
        if val < 0:
            print("Error: Input must be a positive number")
            exit()
    except ValueError:
        try:
            # Convert it into float
            val = float(input)
        except ValueError:
            print("Error: Wrong type of Input")
            exit()
    return val

def load_parameters(path):
    """
    load theta0 and theta1 from csv file. note that w = theta0, b = theta1
    """
    if not os.path.exists(path):
        print("Error: file with parameters is missing")
        exit()
    
    df = pd.read_csv(path)
    thetas = [np.array(df["theta0"][0]), np.array(df["theta1"][0])]
    return thetas

def load_norm_parameters(path):
    """
    Load parameters used for normalization in the training process.
    return the mean and standard deviation of the independent variable X (km)
    """
    if not os.path.exists(path):
        print("Error: file with normalization data is missing")
        exit()
    
    df = pd.read_csv(path)
    norm_mean = np.array(df["normalization_mean"][0])
    norm_std = np.array(df["normalization_std"][0])
    return norm_mean, norm_std

def normalize_input(x, norm_mean, norm_std):
    """
    apply same transformation done on training input X, only if params are non 0
    """
    if norm_std != 0:
        return (x - norm_mean) / norm_std
    return x

if __name__ == "__main__":
    if len(sys.argv) != 1:
        print("Error: wrong number of argument.")
        exit()
    
    thetas = load_parameters("data/parameters.csv")
    norm_mean, norm_std = load_norm_parameters("data/normalization.csv")

    mileage = check_user_input(input("Provide the mileage (in km) to predict the price on: "))
    normalize_mileage = normalize_input(mileage, norm_mean, norm_std)
    Model = LinearRegressionModel()
    prediction = Model.predict(normalize_mileage, thetas[0], thetas[1])

    print(f"Predicted price for the given mileage is: {prediction:8.2f}")

    