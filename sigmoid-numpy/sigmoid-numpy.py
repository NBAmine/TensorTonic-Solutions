import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """

    x_arr = np.array(x)
    
    denominator = 1 + np.exp(-x_arr)
    
    res = 1 / denominator
    
    return res