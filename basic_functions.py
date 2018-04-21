# Import packages
import math
import numpy as np

def basic_sigmoid(x):
    """
    Computes sigmoid of a scalar x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    
    s = 1/(1 + math.exp(-x))
    
    return s

def sigmoid_derivative(x):
    """
    Computes the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Computed gradient.
    """
    
    s = sigmoid(x)
    ds = s * (1-s)
    
    return ds

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2], 1))
    
    return v

def normalizeRows(x):
    """
    Normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)
    x = x/x_norm

    return x

def softmax(x):
    """
    Calculates the softmax for each row of the input x.

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    
    # Applies exp() element-wise to x.
    x_exp = np.exp(x)

    # Creates a vector x_sum that sums each row of x_exp. 
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    
    # Computes softmax(x) by dividing x_exp by x_sum. 
    s = x_exp/x_sum
    
    return s

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function 
    """
    
    loss = np.sum(abs(yhat - y))
    
    return loss

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function 
    """
    
    loss = np.sum(np.dot(y - yhat, y - yhat))
    
    return loss