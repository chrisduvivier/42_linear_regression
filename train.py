import numpy as np
import pandas as pd
import sys
import os.path
from linear_regression import *
import matplotlib.pyplot as plt
from distutils.util import strtobool

def user_yes_no_query(question):
    sys.stdout.write('%s [y/n]\n' % question)
    while True:
        try:
            return strtobool(input().lower())
        except ValueError:
            sys.stdout.write('Please respond with \'y\' or \'n\'.\n')

def load_dataset(path):
    if not os.path.exists(path):
        print("Error: dataset is missing")
        exit()
    df = pd.read_csv(path)
    return df

def store_parameters(path, theta0, theta1):
    """
    store theta0 and theta1
    """
    # read the csv file
    df = load_dataset(path)
  
    # updating the column value/data
    df.loc[0, 'theta0'] = theta0
    df.loc[0, 'theta1'] = theta1
  
    # writing into the file
    df.to_csv(path, index=False)

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

def plot_evolution(p_history, num_iter, x_train, y_train, x_norm):
    """
    plot the evolution of the line with each iteration checkpoints
    """
    fig, ax1 = plt.subplots()
    ax1.scatter(x_norm, y_train)
    ax1.set_ylabel('price')
    ax1.set_xlabel('km normalized', color='g')

    for idx in range(10):
        i = (idx * num_iter) // 10
        params = p_history[i]
        plt.plot(x_norm, x_norm * params[0] + params[1], label='iter:{i}'.format(i= idx * num_iter//10))
    plt.legend(loc='best')
    plt.title("Evolution of theta parameters")
    plt.show()
    
def plot(x_original, y, x_norm, w, b):
    """
    """
    # plot the model
    y_pred = x_norm * w + b
    x1 = x_norm
    x2 = x_original

    fig, ax1 = plt.subplots()

    ax2 = ax1.twiny()
    ax1.scatter(x1, y)
    ax1.plot(x1, y_pred, "g-")
    ax2.plot(x2, y_pred, 'b-')

    ax1.set_ylabel('price')
    ax1.set_xlabel('km normalized', color='g')
    ax2.set_xlabel('km', color='b')
    plt.title("Final product")
    plt.show()

if __name__ == "__main__":
    
    df = load_dataset("data/data.csv")
    
    x_train = np.array(df["km"])
    y_train = np.array(df["price"])

    norm_mean = x_train.mean()
    norm_std = x_train.std()
    store_normalization_data("data/normalization.csv", norm_mean, norm_std)
    x_norm = normalize_input(x_train, norm_mean, norm_std)

    Model = LinearRegressionModel()
    
    # initialize parameters
    thetas = load_dataset("data/parameters.csv")
    w_init = np.array(thetas["theta0"])[0]
    b_init = np.array(thetas["theta1"])[0]

    # parameters for gradient descent
    iterations = 1000
    alpha = 1.0e-2 # 0.01

    # run gradient descent
    w_final, b_final = Model.gradient_descent(x_norm ,y_train, w_init, b_init, alpha, 
                                                    iterations, Model.compute_cost, Model.compute_gradient)
    print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
    
    prediction = np.dot(x_norm, w_final) + b_final
    r_sq = Model.r_square(prediction, y_train)
    print(f"R square of the model: {r_sq:8.2f}")

    # Prompt user to save parameters or not
    if (user_yes_no_query("Do you wish to save the parameters?")):
        store_parameters("data/parameters.csv", w_final, b_final)

    # plot evolution
    plot_evolution(Model.p_history, iterations, x_train, y_train, x_norm)
    
    # plot results
    plot(x_train, y_train, x_norm, w_final, b_final)

    

    
    