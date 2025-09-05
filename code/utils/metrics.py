__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import numpy as np

def dice(pred, truth, epsilon=1e-8):
    num = 2 * (np.sum((pred * truth), axis=(1,2,3)))
    den = (np.sum(pred,axis=(1,2,3)) + np.sum(truth,axis=(1,2,3)))
    return num / (den + epsilon)