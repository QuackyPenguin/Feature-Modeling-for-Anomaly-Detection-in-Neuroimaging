
import os
import sys
from pathlib import Path
import gc
import inspect

my_path = Path(os.path.dirname(__file__))
sys.path.append(str(my_path))
my_path = os.fspath(my_path.parent)
sys.path.append(str(my_path))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from conf import val_config as config
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from utils.dataloaders import MRI_Volume
from utils.utils import calc_dice_scores, mahalanobis_distance, project_to_target, normalize_sigmoid, clean_dir
from utils.visualize import plot_anomaly_distributions
from safetensors.torch import load_file as load_safetensors


def validation_3d():
    conf = config.get_config()
    print(f'Ouput will be saved in {my_path}/{conf.general.output}', flush=True)
    torch.manual_seed(conf.seed)
    accelerator = Accelerator()
    device = accelerator.device
    
    dataloader = MRI_Volume(conf)
    
    model_kwargs = dict(
        in_channels = conf.model.in_channels,
        out_channels = conf.model.out_channels,
    )

    sig = inspect.signature(conf.model.model)   
    if "use_ckpt" in sig.parameters:
        model_kwargs["use_ckpt"] = False                

    model = conf.model.model(**model_kwargs) 
    model.register_hooks()
    model = torch.compile(model, mode="reduce-overhead")
    model = accelerator.prepare(model)
        
    dataloader = accelerator.prepare(dataloader)
    checkpoint_path = f"{my_path}/checkpoints/{conf.general.run_name}/{conf.feature_extraction.checkpoint}"
    weights = load_safetensors(f"{checkpoint_path}/model.safetensors")
    model.load_state_dict(weights)
    model.eval()

    source_layers = conf.feature_extraction.source_layers
    target_layer = conf.feature_extraction.target_layer
    
    gaussian_dir = Path(f"{my_path}/checkpoints/{conf.general.run_name}/gaussians")
    # gaussian_files = list(gaussian_dir.glob("voxel_*.npz"))  # Get list of files for tqdm
    gaussian_file = gaussian_dir / "gaussians.npz"
    gaussians = {}
    
    data = np.load(gaussian_file)
    
    voxel_count = torch.tensor(data["count"], dtype=torch.float32)
    voxel_mean = torch.tensor(data["mean"], dtype=torch.float32)
    voxel_M2 = torch.tensor(data["M2"], dtype=torch.float32)

    # Compute covariance
    voxel_count_safe = torch.clamp(voxel_count - 1, min=1)  # Avoid division by zero
    voxel_cov = voxel_M2 / voxel_count_safe[..., None, None]  # (D', H', W', C_total, C_total)

    # Add small regularization for numerical stability
    epsilon = 1e-8
    voxel_cov += epsilon * torch.eye(voxel_cov.shape[-1]).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Broadcast across (D', H', W')

    # Store in a dictionary (if needed for compatibility)
    gaussians = {
        "mu": voxel_mean,  # (D', H', W', C_total)
        "Sigma": voxel_cov  # (D', H', W', C_total, C_total)
    }

    print(f"Loaded Gaussians: mean shape {voxel_mean.shape}, covariance shape {voxel_cov.shape}")
    
    
    anomaly_maps = []  # I usally go over the volumes and append the anomaly map for them
    labels = []
    
    with torch.no_grad():
        for i, (volume, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing Batches"):
            volume = volume.to(device, dtype=torch.float32)
            
            anomaly_map = torch.zeros((volume.shape[0], *volume.shape[2:]), dtype=torch.float32, device="cpu") # (B, D, H, W)
            
            _, _ = model(volume)
            features = model.features
            # print(f"{features.keys()}", flush=True)
            target_feature = features[target_layer].to("cpu")
            
            batch_size, _, depth, height, width = target_feature.shape # (B, C', D', H', W')
            
            projected_features = []
            target_shape = (depth, height, width)
            
            for layer_name in source_layers:
                feature = features[layer_name].to("cpu")
                
                if feature.shape[2:] != target_shape: 
                    feature = project_to_target(feature_batch=feature, target_shape=target_shape)
                
                projected_features.append(torch.tensor(feature, dtype=torch.float32))
                
            final_feature_map = torch.cat(projected_features, dim=1) # (B, C_total, D', H', W')
            
            # (B, C_total, D', H', W') -> (B, D'*H'*W', C_total)
            C_total = final_feature_map.shape[1]  # Extract channel dimension 
            final_feature_map = final_feature_map.view(batch_size, C_total, -1).permute(0, 2, 1).to("cpu")
            
            # Compute Mahalanobis distance per voxel
            distances = torch.zeros((batch_size, depth, height, width), device="cpu")
            
            for d in range(depth):
                for h in range(height):
                    for w in range(width):
                        mu = gaussians["mu"][d, h, w]
                        Sigma = gaussians["Sigma"][d, h, w]
                        
                        voxel_feature = final_feature_map[:, d * height * width + h * width + w, :] # (B, C_total)
                        
                        distances[:, d, h, w] = mahalanobis_distance(voxel_feature, mu, Sigma) # (B,)
                
            # project the distances back to the original volume size
            anomaly_map = F.interpolate(
                distances.unsqueeze(1), # add channel dimension
                size=volume.shape[2:], # to (D, H, W)
                mode="trilinear",
                align_corners=False
            ).squeeze(1) # remove channel dimension
        
            anomaly_map, label = accelerator.gather_for_metrics((anomaly_map, label)) # this is needed when you evaluate with multi GPU
            labels.append(label.to("cpu").type(torch.bool))
            anomaly_maps.append(anomaly_map.to("cpu"))
            
            model.features = {}
            
        model.remove_hooks()
        
        del model, gaussians, voxel_count_safe, voxel_count, voxel_cov, voxel_feature, voxel_M2, voxel_mean
        gc.collect()
        
        if accelerator.is_main_process:
            anomaly_maps = torch.cat(anomaly_maps, dim=0)
            labels = torch.cat(labels, dim=0)
                        
            # min_before_norm = anomaly_maps.min().item()
            # print(f"🔍 Min anomaly score before normalization: {min_before_norm}")
            total_anom = int(labels.sum())            # count of voxels == True
            print(f"Dataset-wide anomalous voxels : {total_anom:,}")

            # normalize the distances to [0, 1]
            anomaly_maps, median, iqr, weight = normalize_sigmoid(anomaly_maps, percentile_low=5, percentile_high=95, sample_fraction=0.01,
                                             use_log=True, weight = 1.0)
            
            anomaly_maps = anomaly_maps.numpy()
            labels = labels.numpy()
            
            print("Calculating aupr score...", flush=True)
            aupr = average_precision_score(labels.ravel(), anomaly_maps.ravel()) # calculate AUPRC
            print("Calculating dice scores...", flush=True)
            dice_scores, best_threshold, best_score = calc_dice_scores(conf, anomaly_maps, labels, validation=True)
            dice_scores["median"] = median
            dice_scores["iqr"] = iqr
            dice_scores["weight"] = weight
            dice_scores["AUPRC"] = aupr
            dice_scores["threshold"] = best_threshold
            dice_scores["best_score"] = best_score
            
            df = pd.DataFrame.from_dict(
                dice_scores, orient="index", columns=["value"]
            )
            df.index.name = "thr"
            
            output_dir = Path(f"{my_path}/{conf.general.output}")  # Full path
            output_dir.mkdir(parents=True, exist_ok=True)  # Ensure full path exists
            
            clean_dir(output_dir=output_dir)
            
            print(df)

            df.to_csv(output_dir / "dice_scores.csv")
            print(f"Results saved in: {my_path}/{conf.general.output}")
            
            plot_anomaly_distributions(anomaly_maps, labels, f'{my_path}/{conf.general.output}/anomaly_hist.png', 
                                       bin_size=conf.hyperparameters.thr_step)
            

def validation_3d_pca():
    conf = config.get_config()
    print(f'Ouput will be saved in {my_path}/{conf.general.output}', flush=True)
    torch.manual_seed(conf.seed)
    accelerator = Accelerator()
    device = accelerator.device

    # ---------------- Data ----------------
    dataloader = MRI_Volume(conf)
    dataloader = accelerator.prepare(dataloader)

    # ---------------- Model ----------------
    model_kwargs = dict(
        in_channels=conf.model.in_channels,
        out_channels=conf.model.out_channels,
    )
    sig = inspect.signature(conf.model.model)
    if "use_ckpt" in sig.parameters:
        model_kwargs["use_ckpt"] = False

    model = conf.model.model(**model_kwargs)
    model.register_hooks()
    model = torch.compile(model, mode="reduce-overhead")
    model = accelerator.prepare(model)

    # ---------------- Weights ----------------
    checkpoint_path = f"{my_path}/checkpoints/{conf.general.run_name}/{conf.feature_extraction.checkpoint}"
    weights = load_safetensors(f"{checkpoint_path}/model.safetensors")
    model.load_state_dict(weights)
    model.eval()

    source_layers = conf.feature_extraction.source_layers
    target_layer  = conf.feature_extraction.target_layer

    # ---------------- Load PCA + Gaussians (reduced k) ----------------
    gaussian_dir = Path(f"{my_path}/checkpoints/{conf.general.run_name}/gaussians")

    pca_np   = np.load(gaussian_dir / "pca.npz")
    pca_mean = torch.tensor(pca_np["mean"], dtype=torch.float32)          # (C_total,)
    pca_W    = torch.tensor(pca_np["components"], dtype=torch.float32)    # (C_total, k)
    k        = pca_W.shape[1]

    g_np         = np.load(gaussian_dir / "gaussians_pca.npz")
    voxel_count  = torch.tensor(g_np["count"], dtype=torch.float32)       # (D',H',W')
    voxel_mean   = torch.tensor(g_np["mean"],  dtype=torch.float32)       # (D',H',W', k)
    voxel_M2     = torch.tensor(g_np["M2"],    dtype=torch.float32)       # (D',H',W', k, k)

    voxel_count_safe = torch.clamp(voxel_count - 1, min=1)                # avoid /0
    voxel_cov = voxel_M2 / voxel_count_safe[..., None, None]              # (D',H',W', k, k)
    # small Tikhonov regularization
    eps = 1e-8
    eye = torch.eye(k, dtype=voxel_cov.dtype)
    voxel_cov = voxel_cov + eps * eye.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    gaussians = {"mu": voxel_mean, "Sigma": voxel_cov}
    print(f"Loaded PCA (k={k}) and Gaussians: mu {voxel_mean.shape}, cov {voxel_cov.shape}", flush=True)

    # ---------------- Validate ----------------
    anomaly_maps = []
    labels = []

    with torch.no_grad():
        for _, (volume, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing Batches"):
            volume = volume.to(device, dtype=torch.float32)

            # forward to populate hooks
            _ = model(volume)
            feats = model.features
            tgt = feats[target_layer].to("cpu")                      # (B,Ct,D',H',W')
            B, _, Dp, Hp, Wp = tgt.shape
            tgt_shape = (Dp, Hp, Wp)

            # project all selected layers to target grid and concat channels
            per_layer = []
            for lname in source_layers:
                f = feats[lname].to("cpu")                           # (B,C,D?,H?,W?)
                if f.shape[2:] != tgt_shape:
                    f = project_to_target(feature_batch=f, target_shape=tgt_shape)
                per_layer.append(torch.tensor(f, dtype=torch.float32))
            feat_cat = torch.cat(per_layer, dim=1)                   # (B, C_total, D',H',W')

            # flatten to (B, N, C_total) with N = D'*H'*W'
            C_total = feat_cat.shape[1]
            feat_flat = feat_cat.view(B, C_total, -1).permute(0, 2, 1).to("cpu")  # (B, N, C_total)

            # --- PCA: center + project to k ---
            feat_flat = feat_flat - pca_mean[None, None, :]          # (B, N, C_total)
            feat_flat = torch.matmul(feat_flat, pca_W)               # (B, N, k)

            # Compute Mahalanobis on target grid
            distances = torch.zeros((B, Dp, Hp, Wp), device="cpu")
            for d in range(Dp):
                mu_d, Sig_d = gaussians["mu"][d], gaussians["Sigma"][d]       # (H',W',k), (H',W',k,k)
                for h in range(Hp):
                    mu_dh, Sig_dh = mu_d[h], Sig_d[h]                          # (W',k), (W',k,k)
                    base = d * Hp * Wp + h * Wp
                    for w in range(Wp):
                        mu = mu_dh[w]                                          # (k,)
                        Sig = Sig_dh[w]                                        # (k,k)
                        fv = feat_flat[:, base + w, :]                         # (B,k)
                        distances[:, d, h, w] = mahalanobis_distance(fv, mu, Sig)

            # upsample to original volume size (D,H,W)
            anom = F.interpolate(
                distances.unsqueeze(1),
                size=volume.shape[2:],
                mode="trilinear",
                align_corners=False
            ).squeeze(1)                                                       # (B,D,H,W)

            # multi-GPU safe gather
            anom, label = accelerator.gather_for_metrics((anom, label))
            labels.append(label.to("cpu").type(torch.bool))
            anomaly_maps.append(anom.to("cpu"))

            # clear hooks cache to avoid memory bloat
            model.features = {}

    model.remove_hooks()
    del model, gaussians, voxel_count, voxel_mean, voxel_M2, voxel_cov, pca_mean, pca_W
    gc.collect()

    # ---------------- Metrics + save ----------------
    if accelerator.is_main_process:
        anomaly_maps = torch.cat(anomaly_maps, dim=0)   # (N,D,H,W)
        labels       = torch.cat(labels, dim=0)         # (N,D,H,W)

        total_anom = int(labels.sum())
        print(f"Dataset-wide anomalous voxels : {total_anom:,}")

        # global normalization → [0,1]
        anomaly_maps, median, iqr, weight = normalize_sigmoid(
            anomaly_maps, percentile_low=5, percentile_high=95,
            sample_fraction=0.01, use_log=True, weight=1.0
        )

        anomaly_np = anomaly_maps.numpy()
        labels_np  = labels.numpy()

        print("Calculating aupr score...", flush=True)
        aupr = average_precision_score(labels_np.ravel(), anomaly_np.ravel())

        print("Calculating dice scores...", flush=True)
        dice_scores, best_threshold, best_score = calc_dice_scores(conf, anomaly_np, labels_np, validation=True)
        dice_scores["median"]     = median
        dice_scores["iqr"]        = iqr
        dice_scores["weight"]     = weight
        dice_scores["AUPRC"]      = aupr
        dice_scores["threshold"]  = best_threshold
        dice_scores["best_score"] = best_score

        df = pd.DataFrame.from_dict(dice_scores, orient="index", columns=["value"])
        df.index.name = "thr"

        out_dir = Path(f"{my_path}/{conf.general.output}")
        out_dir.mkdir(parents=True, exist_ok=True)
        clean_dir(output_dir=out_dir)  # optional: wipe dir first

        print(df)
        # Save under a PCA-specific filename to avoid overwriting the non-PCA run
        df.to_csv(out_dir / "dice_scores_pca.csv")
        print(f"Results saved in: {out_dir}")

        plot_anomaly_distributions(anomaly_np, labels_np, str(out_dir / "anomaly_hist_pca.png"),
                                   bin_size=conf.hyperparameters.thr_step)
        

def main():
    validation_3d()

            
if __name__ == "__main__":
    main()
