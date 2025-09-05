__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"

import numpy as np
import skimage.exposure as ex
import torch


def normalize_image(image, clip_percentiles=False, pmin=1, pmax=99):
    """
    Function to normalize the images between [0,1]. If percentiles is set to True it clips the intensities at
     percentile 1 and 99
    :param image: numpy array containing the image
    :param clip_percentiles: set to True to clip intensities. (default: False)
    :param pmin: lower percentile to clip
    :param pmax: upper percentile to clip
    :return: normalized image [0,1]
    """
    if clip_percentiles is True:
        pmin = np.percentile(image, pmin)
        pmax = np.percentile(image, pmax)
        v = np.clip(image, pmin, pmax)
    else:
        v = image.copy()

    v_min = v.min(axis=(0, 1, 2), keepdims=True)
    v_max = v.max(axis=(0, 1, 2), keepdims=True)

    return (v - v_min) / (v_max - v_min)