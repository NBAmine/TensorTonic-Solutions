import numpy as np

def relu(x):
    return np.maximum(0, x)
    
def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """

    w1 = np.array(W1)
    w2 = np.array(W2)
    x = np.array(x)
    
    identity = x

    layer1 = relu(x @ w1.T)
    layer2 = layer1 @ w2.T
    skip_con = layer2 + identity

    y = relu(skip_con)
    
    return y

    
