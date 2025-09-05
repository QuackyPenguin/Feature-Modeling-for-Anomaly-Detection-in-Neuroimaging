__author__ = "Alexander Frotscher"
__email__ = "alexander.frotscher@student.uni-tuebingen.de"


import functools
import multiprocessing as mp
import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter
from tqdm import tqdm
import torch.nn.functional as F
import shutil

from utils.metrics import dice

def clean_dir(output_dir):
    if output_dir.exists():
        shutil.rmtree(output_dir)  # Delete entire directory

    output_dir.mkdir(parents=True, exist_ok=True)

def calc_dice_scores(conf, my_volume, my_labels, validation=False):
    thresholds = [
            round(x, 3)
            for x in np.arange(conf.hyperparameters.thr_start, conf.hyperparameters.thr_end, 
                               conf.hyperparameters.thr_step)
        ]
    dice_scores = {}
    my_call = functools.partial(calc_indiv_dice, my_volume, my_labels)
    
    with mp.Pool(conf.general.workers) as pool:
        results = pool.map(my_call, thresholds)
    
    for i, dice_score in enumerate(results):
        dice_scores[f"{thresholds[i]}"] = dice_score
    
    if validation:
        # best threshold and dice score
        best_threshold = max(dice_scores, key=dice_scores.get)
        best_dice = dice_scores[best_threshold]
        return dice_scores, best_threshold, best_dice

    return dice_scores

def calc_indiv_dice(volume, labels, thr):
    segmentation = np.where(volume > thr, 1.0, 0.0)
    segmentation = segmentation.astype(np.float32)
    dices = ([float(x) for x in dice(segmentation, labels)])
    dice_score = np.mean(np.asarray(dices)) # mean over all subjects
    # print(f"Dice score: {dice_score} for threshold: {thr}", flush=True)
    return dice_score


def make_dicts(path,run_name):
    os.makedirs(f"{path}/checkpoints", exist_ok=True)
    os.makedirs(os.path.join(f"{path}/checkpoints", run_name), exist_ok=True)
    

def mahalanobis_distance(features, mu, Sigma, device="cpu"):
    """
    Compute the Mahalanobis distance between the features and the Gaussian distribution.

    Args:
        features (torch.Tensor): Shape (B, C_total)
        mu (torch.Tensor): Shape (C_total,)
        Sigma (torch.Tensor): Shape (C_total, C_total)

    Returns:
        torch.Tensor: Mahalanobis distances of shape (B,)
    """
    delta = features - mu  # (B, C_total)

    # Compute the inverse covariance matrix
    Sigma_inv = torch.inverse(Sigma).to(device)  # (C_total, C_total)

    # Mahalanobis distance formula: sqrt(delta^T * Sigma^-1 * delta)
    distances = torch.sum(delta @ Sigma_inv * delta, dim=1)  # (B,)
    distances = torch.sqrt(torch.clamp(distances, min=0))

    return distances


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)


def normalize_sigmoid(anomaly_maps, percentile_low=5, percentile_high=95, sample_fraction=0.01, use_log=True, weight = 1.0):
    """
    Normalize anomaly maps using robust quantile-based scaling and optional log transformation.

    Args:
        anomaly_maps (torch.Tensor): Anomaly scores from Mahalanobis distance (B, D, H, W).
        percentile_low (float): Lower percentile for robust scaling.
        percentile_high (float): Upper percentile for robust scaling.
        sample_fraction (float): Fraction of values to sample for quantile computation.
        use_log (bool): Whether to apply log1p() transformation for skewed data.

    Returns:
        torch.Tensor: Normalized anomaly scores in the range [0, 1].
    """
    anomaly_maps = anomaly_maps.to(torch.float32)  # Ensure float32 for stability
    all_values = anomaly_maps.view(-1)  # Flatten all batches

    # Subsample values to reduce memory usage
    num_samples = int(len(all_values) * sample_fraction)
    sampled_values = all_values[torch.randperm(len(all_values))[:num_samples]]  # Random sampling

    # Apply log1p() if needed
    if use_log:
        anomaly_maps = torch.log1p(anomaly_maps)
        sampled_values = torch.log1p(sampled_values)

    # Compute robust statistics using sampled values
    median = torch.median(sampled_values)
    q_low = torch.quantile(sampled_values, percentile_low / 100.0)
    q_high = torch.quantile(sampled_values, percentile_high / 100.0)
    iqr = q_high - q_low  # Interquartile range

    # Prevent division by zero in case IQR is too small
    iqr = torch.clamp(iqr, min=1e-6)

    # Standardize based on IQR
    normalized = (anomaly_maps - median) / ( iqr * weight )

    # Apply sigmoid for final normalization or tanh
    # return torch.sigmoid(normalized)
    return torch.sigmoid(normalized), median, iqr, weight


def project_to_target(feature_batch, target_shape):
    if isinstance(feature_batch, np.ndarray):
        feature_batch = torch.tensor(feature_batch, dtype=torch.float32)
    
    feature_tensor = feature_batch.clone().detach().float()
    projected = F.interpolate(
        feature_tensor,
        size=target_shape,
        mode='trilinear',
        align_corners=False
    )
    return projected.cpu().numpy()
