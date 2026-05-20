import numpy as np

def relu(x):
    """
    Rectified Linear Unit activation function.
    """
    return np.maximum(x, 0)

def conv_block(x, W1, W2, Ws):
    """
    Computes the forward pass of a simplified ResNet Convolutional Block.
    Aligns input dimensions using a projection shortcut before adding to the main path.

    Args:
        x (array_like): Input features or activation matrix.
        W1 (array_like): Weight matrix for the first layer in the main path.
        W2 (array_like): Weight matrix for the second layer in the main path.
        Ws (array_like): Weight matrix for the projection shortcut path.

    Returns:
        np.ndarray: Matrix containing the ReLU-activated sum of the main path and shortcut.
        
    Raises:
        ValueError: If matrix multiplications fail due to incompatible shapes.
        TypeError: If inputs cannot be safely cast into NumPy numeric arrays.
    """

    # ==============================================================
    #               Safe conversion to NumPy arrays
    # ==============================================================
    try:
       
        w1 = np.asarray(W1, dtype=np.float32)
        w2 = np.asarray(W2, dtype=np.float32)
        ws = np.asarray(Ws, dtype=np.float32)
        x_ = np.asarray(x, dtype=np.float32)
        
    except (ValueError, TypeError) as e:
        raise TypeError(f"Input conversion failed. Ensure all inputs are numeric arrays: {e}")

    # ==============================================================
    #           First and Second Linear Transformations
    # ==============================================================
    try:
        layer1 = relu(x_ @ w1)
        layer2 = layer1 @ w2
    except ValueError as e:
        raise ValueError(f"Shape mismatch in the main path. Check dimensions of x {x_.shape}, w1 {w1.shape} and w2 {w2.shape}: {e}")

    # ==============================================================
    #                         Shortcut Path
    # ==============================================================
    try:
        shortcut = x_ @ ws
    except ValueError as e:
        raise ValueError(f"Shape mismatch in the shortcut path. Input x {x_.shape} cannot be multiplied by ws {ws.shape}: {e}")

    # ==============================================================
    #                         Fusion + ReLU
    # ==============================================================
    try:
        output = relu(layer2 + shortcut)
    except ValueError as e:
        raise ValueError(f"Dimension mismatch during residual addition. layer2 {layer2.shape} and shortcut {shortcut.shape} must match: {e}")
        
    return output