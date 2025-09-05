import os
import sys
from pathlib import Path

my_path = Path(os.path.dirname(__file__))
sys.path.append(my_path)
my_path = os.fspath(my_path.parent)
sys.path.append(my_path)

import torch
import torch.nn.functional as F
from tqdm import tqdm
from conf import train_config as config
import numpy as np
from utils.utils import project_to_target, clean_dir
import gc  # Garbage collection


def fit_gaussian3d(conf):
    
    print("Starting Gaussian fitting...", flush=True)
    
    features_dir = Path(f"{my_path}/checkpoints/{conf.training.run_name}/{conf.feature_extraction.output}")
    output_dir = Path(f"{my_path}/checkpoints/{conf.training.run_name}/gaussians")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    source_layers = conf.feature_extraction.source_layers
    target_layer = conf.feature_extraction.target_layer
            
    layer_shapes_path = features_dir / "layer_shapes.npz"
    layer_shapes = np.load(layer_shapes_path)
    
    print(layer_shapes.keys())
    print(layer_shapes.items())
    
    target_shape = layer_shapes[target_layer][1:] # D', H', W' of the target layer
    target_shape=tuple(map(int, target_shape))
    
    all_feature_files = sorted(features_dir.glob("batch_*_features.npz"))
    
    initialized = False

    for feature_file in tqdm(all_feature_files, desc="Processing Batches"):
        data = np.load(feature_file, allow_pickle=True)

    # Process combined batch as before...

        projected_features = []

        for layer_name in source_layers:
            feature = data[layer_name]
            C, D, H, W = layer_shapes[layer_name]

            feature = feature.reshape(-1, D, H, W, C).transpose(0, 4, 1, 2, 3)

            if (D, H, W) != tuple(target_shape):
                feature = project_to_target(feature_batch=feature, target_shape=target_shape)

            projected_features.append(feature)

        concatenated_features = np.concatenate(projected_features, axis=1) # (B, C_total, D', H', W')
        
        del projected_features  # Free memory
        gc.collect()  # Force garbage collection

        B, C_total, D_prime, H_prime, W_prime = concatenated_features.shape
        
        print(f"\nAcquiring statistics...", flush=True)
        
        if not initialized:
            # Initialize voxel statistics as np arrays
            voxel_count = np.zeros((D_prime, H_prime, W_prime), dtype=np.int32)
            voxel_mean = np.zeros((D_prime, H_prime, W_prime, C_total), dtype=np.float64)
            voxel_M2 = np.zeros((D_prime, H_prime, W_prime, C_total, C_total), dtype=np.float64)
            initialized = True
            
        voxel_samples = concatenated_features.transpose(2, 3, 4, 0, 1) # (D', H', W', B, C_total)
        
        batch_mean = np.mean(voxel_samples, axis = 3) # (D', H', W', C_total)
        centered = voxel_samples - batch_mean[..., None, :] # broadcast over B
        batch_M2   = np.einsum('dhwbc,dhwbx->dhwcx', centered, centered)   # sum((x-m)(x-m)^T)
        
        del concatenated_features, centered, voxel_samples
        gc.collect()  # Force garbage collection
        
        print("Calculating...", flush=True)
        
        new_count = voxel_count + B
        
        delta = batch_mean - voxel_mean
        voxel_mean += delta * (B / new_count[..., None])
        
        del batch_mean
        gc.collect()
        
        # outer(delta, delta) with broadcasting → (D',H',W',C,C)
        delta_outer = np.einsum('dhwc,dhwx->dhwcx', delta, delta)
        
        del delta
        gc.collect()
        
        voxel_M2 += batch_M2 + (voxel_count[..., None, None] * B / new_count[..., None, None]) * delta_outer
        
        voxel_count = new_count
        
        del new_count, delta_outer, batch_M2
        gc.collect()
        
        # batch_covariance = np.einsum('dhwbc,dhwbx->dhwcx', voxel_samples, voxel_samples) / B # (D', H', W', C_total, C_total)
        # voxel_M2 += batch_covariance
        # del batch_covariance, voxel_samples
        # gc.collect()
        
        # print("Calculating mean correction...", flush=True)
        
        # mean_correction = B * np.einsum('dhwc,dhwx->dhwcx', batch_mean, batch_mean)
        # voxel_M2 -= mean_correction
        # del mean_correction
        # gc.collect()
        
        # print("Calculating delta correction...", flush=True)
        
        # delta = batch_mean - voxel_mean
        # voxel_mean += delta * (B / new_count[...,None])
        # delta2 = batch_mean - voxel_mean
        # delta_correction = voxel_count[..., None, None] * np.einsum('dhwc,dhwx->dhwcx', delta, delta2)
        # voxel_M2 += delta_correction
        # del delta_correction, batch_mean, delta, delta2
        # gc.collect()
        
        # voxel_count = new_count
        
        # del new_count
        # gc.collect()
            
    # saving gaussian paramters
    clean_dir(output_dir=output_dir)    
    print("Saving Gaussian parameters...", flush=True)   
    np.savez_compressed(output_dir / "gaussians.npz", count=voxel_count, mean=voxel_mean, M2=voxel_M2)
        
    print(f"Gaussian fitting completed. Parameters saved in: {output_dir}")
    
def fit_gaussian3d_pca(conf, k: int = 64):
    """
    Two-pass Gaussian fitting with global PCA projection to k dims.
    Saves:
      - gaussians.npz: count (D',H',W'), mean (D',H',W',k), M2 (D',H',W',k,k)
      - pca.npz: mean (C_total,), components (C_total,k), explained_var (k,)
    Assumptions:
      - features_dir/<run_name>/<output>/ contains batch_*_features.npz from your 3D extractor
      - layer_shapes.npz contains entries: layer_name -> (C, D, H, W)
      - Each batch file stores per-layer flats either under f"{layer}_feat" (N, C_layer)
        or under {layer} (N, C_layer). We handle both.
    """
    print("Starting Gaussian fitting with PCA...", flush=True)

    features_dir = Path(f"{my_path}/checkpoints/{conf.training.run_name}/{conf.feature_extraction.output}")
    output_dir   = Path(f"{my_path}/checkpoints/{conf.training.run_name}/gaussians")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_layers = conf.feature_extraction.source_layers
    target_layer  = conf.feature_extraction.target_layer

    layer_shapes_path = features_dir / "layer_shapes.npz"
    if not layer_shapes_path.exists():
        raise FileNotFoundError(f"Missing layer_shapes.npz in {features_dir}")

    # Load shapes and target spatial shape (D', H', W') from the target layer
    layer_shapes_np = np.load(layer_shapes_path, allow_pickle=True)
    # {layer_name: (C, D, H, W)}
    layer_shapes = {k: tuple(map(int, layer_shapes_np[k])) for k in layer_shapes_np.files}
    Dp, Hp, Wp = map(int, layer_shapes[target_layer][1:])  # target (D', H', W')

    all_files = sorted(features_dir.glob("batch_*_features.npz"))
    if not all_files:
        raise FileNotFoundError(f"No feature files found in {features_dir}")

    # ---------------- PASS 1: accumulate global covariance for PCA ----------------
    print("Pass 1/2: accumulating global covariance for PCA...", flush=True)

    sum_x   = None      # (C_total,)
    sum_xxT = None      # (C_total, C_total)
    n_total = 0
    C_total = None

    for fpath in tqdm(all_files, desc="PCA pass"):
        # init locals so we can null them in finally
        data = None
        per_layer_list = None
        feat = None
        t = None
        X = None

        try:
            data = np.load(fpath, allow_pickle=True)

            per_layer_list = []
            # Build each layer to (B, C, D', H', W'), then flatten → (B*D'*H'*W', C_layer)
            for layer_name in source_layers:
                # read flats
                if f"{layer_name}_feat" in data.files:
                    flat = data[f"{layer_name}_feat"]
                elif layer_name in data.files:
                    flat = data[layer_name]
                else:
                    raise KeyError(f"{layer_name} not found in {fpath.name}")

                C, D, H, Wl = layer_shapes[layer_name]
                vox_per_sample = D * H * Wl
                if flat.shape[0] % vox_per_sample != 0:
                    raise ValueError(
                        f"{fpath.name}:{layer_name}: N={flat.shape[0]} not divisible by D*H*W={vox_per_sample}"
                    )
                B = flat.shape[0] // vox_per_sample

                # (N, C) -> (B, D, H, W, C) -> (B, C, D, H, W)
                feat = flat.reshape(B, D, H, Wl, C).transpose(0, 4, 1, 2, 3)

                # resize to (D', H', W') if needed
                if (D, H, Wl) != (Dp, Hp, Wp):
                    t = torch.from_numpy(feat)  # may be float64; that's fine
                    t = F.interpolate(t, size=(Dp, Hp, Wp), mode="trilinear", align_corners=False)
                    feat = t.numpy()
                    t = None

                # -> (B, D', H', W', C_layer) -> (N', C_layer)
                feat = feat.transpose(0, 2, 3, 4, 1)  # (B, D', H', W', C_layer)
                feat = feat.reshape(-1, feat.shape[-1]).astype(np.float64)
                per_layer_list.append(feat)

            # same number of rows across layers => concat channels
            X = np.concatenate(per_layer_list, axis=1)  # (N', C_total)

            if C_total is None:
                C_total = X.shape[1]
                sum_x   = np.zeros((C_total,), dtype=np.float64)
                sum_xxT = np.zeros((C_total, C_total), dtype=np.float64)

            # accumulate in float64 for stability
            sum_x   += X.sum(axis=0, dtype=np.float64)
            sum_xxT += X.T @ X
            n_total += X.shape[0]

        finally:
            # cleanup: drop references and GC
            X = None
            per_layer_list = None
            feat = None
            t = None
            data = None
            gc.collect()

    if n_total == 0:
        raise RuntimeError("No samples encountered for PCA.")

    pca_mean = (sum_x / n_total).astype(np.float32)         # (C_total,)
    cov  = (sum_xxT / n_total) - np.outer(pca_mean, pca_mean).astype(np.float64)  # (C_total, C_total)

    # eigen-decomposition (symmetric PSD)
    evals, evecs = np.linalg.eigh(cov)                      # ascending
    idx = np.argsort(evals)[::-1]                           # descending
    evals = evals[idx]
    evecs = evecs[:, idx]

    if k > C_total:
        k = C_total
        print(f"[warn] requested k > C_total; clipping k={k}")

    pca_W  = evecs[:, :k].astype(np.float32)   # (C_total, k)
    pca_ev = evals[:k].astype(np.float32)      # (k,)

    # ---------------- PASS 2: project features and accumulate voxel Gaussians ----
    print("Pass 2/2: projecting features and accumulating voxel Gaussians...", flush=True)

    initialized = False
    voxel_count = None
    voxel_mean  = None
    voxel_M2    = None

    for fpath in tqdm(all_files, desc="Gaussian pass"):
        data = None
        per_layer_list = None
        feat = None
        t = None
        X = None
        Xr = None
        batch_mean = None
        centered = None
        cp = None
        batch_M2 = None
        delta = None
        delta_outer = None

        try:
            data = np.load(fpath, allow_pickle=True)

            per_layer_list = []
            for layer_name in source_layers:
                if f"{layer_name}_feat" in data.files:
                    flat = data[f"{layer_name}_feat"]
                elif layer_name in data.files:
                    flat = data[layer_name]
                else:
                    raise KeyError(f"{layer_name} not found in {fpath.name}")

                C, D, H, Wl = layer_shapes[layer_name]
                vox_per_sample = D * H * Wl
                if flat.shape[0] % vox_per_sample != 0:
                    raise ValueError(
                        f"{fpath.name}:{layer_name}: N={flat.shape[0]} not divisible by D*H*W={vox_per_sample}"
                    )
                B = flat.shape[0] // vox_per_sample

                # (N, C) -> (B, C, D, H, W)
                feat = flat.reshape(B, D, H, Wl, C).transpose(0, 4, 1, 2, 3)

                if (D, H, Wl) != (Dp, Hp, Wp):
                    t = torch.from_numpy(feat)
                    t = F.interpolate(t, size=(Dp, Hp, Wp), mode="trilinear", align_corners=False)
                    feat = t.numpy()
                    t = None

                # -> (B, D', H', W', C_layer) -> (N', C_layer)
                feat = feat.transpose(0, 2, 3, 4, 1)
                feat = feat.reshape(-1, feat.shape[-1]).astype(np.float32)
                per_layer_list.append(feat)

            X = np.concatenate(per_layer_list, axis=1).astype(np.float32)  # (N', C_total)

            # Center & project to k dims
            X -= pca_mean[None, :]
            Xr = X @ pca_W  # (N', k)

            vox_per_vol = Dp * Hp * Wp
            if Xr.shape[0] % vox_per_vol != 0:
                raise ValueError(f"{fpath.name}: N'={Xr.shape[0]} not divisible by D'*H'*W'={vox_per_vol}")
            B = Xr.shape[0] // vox_per_vol

            Xr = Xr.reshape(B, Dp, Hp, Wp, k)  # (B, D', H', W', k)

            # batch stats over the batch dimension B
            batch_mean = Xr.mean(axis=0)                 # (D', H', W', k)
            centered   = Xr - batch_mean[None, ...]      # (B, D', H', W', k)
            cp = np.transpose(centered, (1, 2, 3, 0, 4)) # (D', H', W', B, k)
            batch_M2 = np.einsum('dhwbk,dhwbm->dhwkm', cp, cp)  # (D', H', W', k, k)

            if not initialized:
                voxel_count = np.zeros((Dp, Hp, Wp), dtype=np.int64)
                voxel_mean  = np.zeros((Dp, Hp, Wp, k), dtype=np.float64)
                voxel_M2    = np.zeros((Dp, Hp, Wp, k, k), dtype=np.float64)
                initialized = True

            old_n = voxel_count
            new_n = old_n + B
            delta = batch_mean - voxel_mean

            voxel_mean += delta * (B / new_n[..., None])
            delta_outer = np.einsum('dhwk,dhwm->dhwkm', delta, delta)

            voxel_M2 += batch_M2 + (old_n[..., None, None] * B / new_n[..., None, None]) * delta_outer
            voxel_count = new_n

        finally:
            # cleanup
            X = None
            Xr = None
            per_layer_list = None
            feat = None
            t = None
            data = None
            batch_mean = None
            centered = None
            cp = None
            batch_M2 = None
            delta = None
            delta_outer = None
            gc.collect()

    # ---------------- save outputs ----------------
    # clean_dir(output_dir=output_dir)

    print("Saving PCA...", flush=True)
    np.savez_compressed(
        output_dir / "pca.npz",
        mean=pca_mean,             # (C_total,)
        components=pca_W,          # (C_total, k)
        explained_var=pca_ev       # (k,)
    )

    print("Saving Gaussian parameters...", flush=True)
    np.savez_compressed(
        output_dir / "gaussians_pca.npz",
        count=voxel_count,         # (D', H', W')
        mean=voxel_mean,           # (D', H', W', k)
        M2=voxel_M2                # (D', H', W', k, k)
    )
    print(f"Gaussian fitting with PCA completed. Saved to: {output_dir}")
    
def main():
    print("Starting Gaussian fitting pipeline...", flush=True)
    torch.backends.cudnn.benchmark = True  # Speed up if input size is consistent
    conf = config.get_config()
    fit_gaussian3d(conf)
    fit_gaussian3d_pca(conf, k=64)


if __name__ == "__main__":
    main()