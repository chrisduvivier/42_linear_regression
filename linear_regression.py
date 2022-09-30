import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math, copy

class LinearRegressionModel():
    def __init__(self):
        # An array to store cost J and w's at each iteration primarily for graphing later
        self.J_history = []
        self.p_history = []

    def compute_cost(self, x, y, w, b):
        """
        Computes the cost function for linear regression.
        
        Args:
        x (ndarray (m,)): Data, m examples 
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters  
        
        Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
        to fit the data points in x and y
        """
        # Number of training examples
        m = x.shape[0]
        
        pred = np.dot(x, w) + b
        cost = np.sum((pred - y)**2)/(2*m)
        return cost
    
    def compute_gradient(self, x, y, w, b):
        """
        Computes the gradient for linear regression 
        Args:
        x (ndarray (m,)): Data, m examples 
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters  
        Returns
        dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
        dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
        """
        # Number of training examples
        m = x.shape[0]
        dj_dw = 0
        dj_db = 0
        
        pred = np.dot(x, w) + b
        diff = pred - y
        
        dj_dw = np.sum(np.dot(diff, x))/m
        dj_db = np.sum(diff)/m
        
        return dj_dw, dj_db
    
    def gradient_descent(self, x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
        """
        Performs gradient descent to fit w,b. Updates w,b by taking 
        num_iters gradient steps with learning rate alpha
        
        Args:
        x (ndarray (m,))  : Data, m examples 
        y (ndarray (m,))  : target values
        w_in,b_in (scalar): initial values of model parameters  
        alpha (float):     Learning rate
        num_iters (int):   number of iterations to run gradient descent
        cost_function:     function to call to produce cost
        gradient_function: function to call to produce gradient
        
        Returns:
        w (scalar): Updated value of parameter after running gradient descent
        b (scalar): Updated value of parameter after running gradient descent
        J_history (List): History of cost values
        p_history (list): History of parameters [w,b] 
        """
        
        w = copy.deepcopy(w_in) #deepcopy w to avoid overwriting it
        b = b_in
        w = w_in
        
        for i in range(num_iters):
            dj_dw, dj_db = gradient_function(x, y, w, b)
            # update parameters
            w = w - alpha * dj_dw
            b = b - alpha * dj_db
            # Save cost J at end of each iteration
            if (i < 100000):
                self.J_history.append(cost_function(x, y, w, b))
                self.p_history.append([w, b])
            if (i % math.ceil(num_iters/10) == 0):
                print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                    f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                    f"w: {w: 0.3e}, b:{b: 0.5e}")
                
        return w, b
    
    def predict(self, x, w, b):
        """
        Performs prediction.
        
        Args:
        x (ndarray (m,))  : Data, m examples 
        w, b (scalar): values of model parameters  
        
        Returns:
        y (scalar): predicted value
        """
        y = np.dot(x, w) + b
        return y