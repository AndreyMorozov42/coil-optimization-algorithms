import numpy as np


def calc_threshold(array_k):
    """
    Function for calculating the threshold value of a one-dimensional array of coupling coefficients.
        Calculated as:
            abs(k[x+1] - k[x]) / k[x]
    """
    e_min = np.min(
        np.array([np.abs(array_k[idx + 1] - array_k[idx]) / array_k[idx] for idx in range(len(array_k) - 1)]))
    return e_min
