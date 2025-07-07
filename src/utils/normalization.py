import numpy as np

def z_score_normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalizes the input data using Z-score normalization.
    
    Parameters:
        data (np.ndarray): Input data of shape (N, 2247), where N is the number of samples.
        
    Returns:
        np.ndarray: Z-score normalized data with the same shape as input.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1
    
    normalized_data = (data - mean) / std
    return normalized_data