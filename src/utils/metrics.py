import numpy as np
from scipy.stats import pearsonr
import tensorflow as tf

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


def negative_pcc_loss(y_true, y_pred):
    x = y_true - tf.reduce_mean(y_true, axis=-1, keepdims=True)
    y = y_pred - tf.reduce_mean(y_pred, axis=-1, keepdims=True)
    numerator = tf.reduce_sum(x * y, axis=-1)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1) * tf.reduce_sum(tf.square(y), axis=-1))
    pcc = numerator / (denominator + 1e-6)
    return -tf.reduce_mean(pcc)