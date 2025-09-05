#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import gc
import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from accelerate import Accelerator
from tqdm import tqdm
from safetensors.torch import load_file as load_safetensors

# ---------------- Project paths ----------------
my_path = Path(os.path.dirname(__file__))
sys.path.append(str(my_path))
my_path = os.fspath(my_path.parent)
sys.path.append(str(my_path))

# ---------------- Project imports ----------------
from conf import val_config as config
from utils.dataloaders import MRI_Volume
from utils.utils import (
    mahalanobis_distance,
    project_to_target,
    normalize_sigmoid,
)

# =============================================================================
# Helpers
# =============================================================================

def subject_id_from_dataset(dataset, global_idx: int) -> str:
    """Extract a subject/file ID from dataset; fallback to index."""
    try:
        if hasattr(dataset, "df"):
            p = dataset.df.iloc[global_idx, 0]
            return Path(p).stem
    except Exception:
        pass
    try:
        if hasattr(dataset, "paths"):
            return Path(dataset.paths[global_idx]).stem
    except Exception:
        pass
    return f"{global_idx:04d}"

def _window_gray(x: np.ndarray, low_pct=2.0, high_pct=98.0) -> np.ndarray:
    """Percentile window + [0,1] normalize for nicer MRI background."""
    lo = np.percentile(x, low_pct)
    hi = np.percentile(x, high_pct)
    if hi <= lo:
        lo, hi = float(x.min()), float(max(x.max(), x.min() + 1.0))
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

def _draw_mri(ax, vol2d: np.ndarray):
    ax.imshow(_window_gray(vol2d), cmap="gray")

def _draw_gt(ax, mask2d: np.ndarray, alpha=0.45, cmap="Reds"):
    ax.imshow(mask2d.astype(float), cmap=cmap, alpha=alpha)

def _draw_heat(ax, heat2d: np.ndarray, cmap="turbo", alpha_max=0.75, alpha_gamma=1.0):
    """Overlay heatmap with opacity scaled by value."""
    heat = np.clip(heat2d, 0.0, 1.0)
    alpha = np.clip(heat, 0, 1) ** alpha_gamma * alpha_max
    im = ax.imshow(heat, cmap=cmap, vmin=0.0, vmax=1.0)
    im.set_alpha(alpha)
    return im

def plot_slice_pair(vol2d, mask2d, heat2d, out_path: Path,
                    title_left="MRI + Ground-truth", title_right="MRI + Heatmap",
                    heat_cmap="turbo", heat_alpha_max=0.75, heat_alpha_gamma=1.0,
                    gt_alpha=0.45):
    """Two-panel: Left MRI+GT, Right MRI+Heatmap (no GT)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: MRI + GT
    _draw_mri(axes[0], vol2d)
    _draw_gt(axes[0], mask2d, alpha=gt_alpha)
    axes[0].set_title(title_left)
    axes[0].axis("off")

    # Right: MRI + Heatmap
    _draw_mri(axes[1], vol2d)
    im = _draw_heat(axes[1], heat2d, cmap=heat_cmap,
                    alpha_max=heat_alpha_max, alpha_gamma=heat_alpha_gamma)
    axes[1].set_title(title_right)
    axes[1].axis("off")
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Normalized anomaly", rotation=270, labelpad=12)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_composite_heat_gt(
    vol2d,
    mask2d,
    heat2d,
    out_path: Path,
    title="MRI + Heatmap + Ground-truth",
    heat_cmap="turbo",
    heat_alpha_max=0.75,
    heat_alpha_gamma=1.0,
    gt_alpha=0.35,
    draw_gt_outline=True,
):
    """Single-panel composite: MRI base + semi-transparent GT + semi-transparent heatmap on top."""
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 6.0))

    # 1) MRI as background
    _draw_mri(ax, vol2d)

    # 2) Ground truth BELOW the heatmap so it doesn't hide it
    _draw_gt(ax, mask2d, alpha=gt_alpha, cmap="Purples")

    # Optional thin outline to make GT visible without blocking heatmap
    if draw_gt_outline and np.any(mask2d):
        try:
            ax.contour(mask2d.astype(float), levels=[0.5], colors="white", linewidths=0.8, alpha=0.9)
        except Exception:
            pass

    # 3) Heatmap ON TOP with intensity-based alpha (implemented inside _draw_heat)
    im = _draw_heat(
        ax,
        heat2d,
        cmap=heat_cmap,
        alpha_max=heat_alpha_max,
        alpha_gamma=heat_alpha_gamma,
    )

    ax.set_title(title)
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized anomaly", rotation=270, labelpad=12)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

# =============================================================================
# Main
# =============================================================================

def main():
    conf = config.get_config()
    out_root = Path(f"{my_path}/checkpoints/{conf.general.run_name}/plots")
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Plots will be saved in {out_root}", flush=True)

    torch.manual_seed(conf.seed)
    accelerator = Accelerator()
    device = accelerator.device

    # ---------------- Data ----------------
    dataloader = MRI_Volume(conf)  # iterable returning (volume, mask)
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

    # ---------------- Load Gaussians ----------------
    gdir = Path(f"{my_path}/checkpoints/{conf.general.run_name}/gaussians")
    gaussian_file = gdir / "gaussians.npz"
    if not gaussian_file.exists():
        print(f"[normal] {gaussian_file} not found — nothing to plot.", flush=True)
        return

    data = np.load(gaussian_file)
    voxel_count = torch.tensor(data["count"], dtype=torch.float32)
    voxel_mean  = torch.tensor(data["mean"],  dtype=torch.float32)  # (D',H',W',C)
    voxel_M2    = torch.tensor(data["M2"],    dtype=torch.float32)  # (D',H',W',C,C)
    voxel_count_safe = torch.clamp(voxel_count - 1, min=1)
    voxel_cov = voxel_M2 / voxel_count_safe[..., None, None]
    eps = 1e-8
    eye = torch.eye(voxel_cov.shape[-1])
    voxel_cov = voxel_cov + eps * eye.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    gaussians = {"mu": voxel_mean, "Sigma": voxel_cov}
    print(f"[normal] mean {voxel_mean.shape}, cov {voxel_cov.shape}", flush=True)

    source_layers = conf.feature_extraction.source_layers
    target_layer  = conf.feature_extraction.target_layer

    # ---------------- Collect all maps for global normalization ----------------
    all_names, all_vols, all_masks, all_anom = [], [], [], []

    with torch.no_grad():
        global_index = 0

        for _, (volume, mask) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Forward pass"):
            volume = volume.to(device, dtype=torch.float32)
            _ = model(volume)
            feats = model.features
            tgt = feats[target_layer].to("cpu")
            B, _, Dp, Hp, Wp = tgt.shape
            tgt_shape = (Dp, Hp, Wp)

            projected = []
            for lname in source_layers:
                f = feats[lname].to("cpu")
                if f.shape[2:] != tgt_shape:
                    f = project_to_target(feature_batch=f, target_shape=tgt_shape)
                projected.append(torch.tensor(f, dtype=torch.float32))
            feat_cat = torch.cat(projected, dim=1)  # (B,C_total,D',H',W')
            C_total = feat_cat.shape[1]
            feat_flat = feat_cat.view(B, C_total, -1).permute(0, 2, 1).to("cpu")

            distances = torch.zeros((B, Dp, Hp, Wp), device="cpu")
            for d in range(Dp):
                mu_d, Sigma_d = gaussians["mu"][d], gaussians["Sigma"][d]
                for h in range(Hp):
                    mu_dh, Sigma_dh = mu_d[h], Sigma_d[h]
                    for w in range(Wp):
                        idx = d * Hp * Wp + h * Wp + w
                        fv = feat_flat[:, idx, :]
                        distances[:, d, h, w] = mahalanobis_distance(fv, mu_dh[w], Sigma_dh[w])

            anom = F.interpolate(
                distances.unsqueeze(1), size=volume.shape[2:],
                mode="trilinear", align_corners=False
            ).squeeze(1)
            anom  = accelerator.gather_for_metrics(anom).to("cpu")
            vol_c = accelerator.gather_for_metrics(volume).to("cpu")
            msk_c = accelerator.gather_for_metrics(mask).to("cpu").type(torch.bool)

            for b in range(anom.shape[0]):
                all_anom.append(anom[b].clone())
                all_vols.append(vol_c[b, 0].clone())
                all_masks.append(msk_c[b].clone())
                all_names.append(subject_id_from_dataset(dataloader.dataset, global_index))
                global_index += 1

            model.features = {}

    model.remove_hooks()
    del model, gaussians
    gc.collect()

    # ---------------- Global normalization & Saving ----------------
    if accelerator.is_main_process:
        stack = torch.stack(all_anom, dim=0)
        norm_maps, *_ = normalize_sigmoid(
            stack, percentile_low=5, percentile_high=95,
            sample_fraction=0.01, use_log=True, weight=1.0
        )
        norm_maps = norm_maps.numpy()

        N = len(all_vols)
        for n in tqdm(range(N), desc="Saving ALL slices"):
            vol, mask, heat_all, name = all_vols[n].numpy(), all_masks[n].numpy().astype(bool), norm_maps[n], all_names[n]
            subj_dir = out_root / name
            subj_dir.mkdir(parents=True, exist_ok=True)

            D = vol.shape[0]
            for s in range(30, min(125, D)):  # only 30..124
                heat = heat_all[s]

                # 1) Two-panel: MRI+GT | MRI+Heatmap
                out_pair = subj_dir / f"slice_{s:03d}.png"
                plot_slice_pair(vol[s], mask[s], heat, out_pair)

                # 2) Composite: MRI+Heatmap+GT
                out_comp = subj_dir / f"slice_{s:03d}_composite.png"
                plot_composite_heat_gt(vol[s], mask[s], heat, out_comp)

        print(f"Saved qualitative figures under: {out_root}")

if __name__ == "__main__":
    main()
