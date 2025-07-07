import numpy as np
from scipy.stats import pearsonr

def calculate_pcc_spectrgorams(pred, actual):
    """
        Computes the mean Pearson correlation coefficient (PCC) across spectral bins 
        between predicted and actual spectrograms.


        Parameters:
            pred (np.ndarray): Predicted spectrogram of shape (N, 128), where N is the number 
                            of samples and 23 is the number of spectral bins.
            actual (np.ndarray): Actual spectrogram of the same shape as pred.

        Returns:
            float: Mean Pearson correlation coefficient across spectral bins.
    """
    pcc_values = []
    for spectral_bin in range(pred.shape[1]):
        pred_bin = pred[:, spectral_bin]
        actual_bin = actual[:, spectral_bin]
        r, p = pearsonr(pred_bin, actual_bin)
        pcc_values.append(r)
    print(f'Pred:{pred.shape}, Actual:{actual.shape}, PCC: {np.mean(pcc_values)}')
    return np.mean(pcc_values)