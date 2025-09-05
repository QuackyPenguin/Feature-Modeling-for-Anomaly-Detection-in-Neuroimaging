import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.dataloaders import Train_MRI_Volume
from conf.train_config import get_config

output_dir = "/home/dkovacevic/MHM_project/plots"
suffix = "unhealthy1"


def plot_anomaly_distributions(anomaly_maps, labels, output_path, bin_size=0.01):
    """
    Plots and saves histograms for anomaly score distributions as percentages.

    Args:
        anomaly_maps (numpy.ndarray): Array of anomaly scores (shape: N, D, H, W).
        labels (numpy.ndarray): Boolean array of labels (same shape as anomaly_maps).
        output_path (str): Path to save the histogram plot.
        bin_size (float): Bin width for histograms.
    """
    # Flatten arrays for histogram plotting
    anomaly_values_all = anomaly_maps.ravel()
    anomalous_values = anomaly_maps[labels == 1].ravel()
    normal_values = anomaly_maps[labels == 0].ravel()

    # Define bins based on min/max values
    bins = np.arange(0, 1 + bin_size, bin_size)  # Always between 0 and 1

    # Create a 2x2 figure layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Anomaly Score Distributions (Percentage)", fontsize=16)

    # Helper function to format percentage axis
    def format_percentage(ax):
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # Plot individual histograms (normalized to percentages)
    axes[0, 0].hist(anomaly_values_all, bins=bins, color="blue", alpha=0.7, weights=np.ones_like(anomaly_values_all) * 100 / len(anomaly_values_all))
    axes[0, 0].set_title("All Anomaly Values")
    axes[0, 0].set_xlabel("Anomaly Score")
    axes[0, 0].set_ylabel("Percentage")
    format_percentage(axes[0, 0])

    axes[0, 1].hist(anomalous_values, bins=bins, color="red", alpha=0.7, weights=np.ones_like(anomalous_values) * 100 / len(anomalous_values))
    axes[0, 1].set_title("Anomalous Voxels (Label = 1)")
    axes[0, 1].set_xlabel("Anomaly Score")
    axes[0, 1].set_ylabel("Percentage")
    format_percentage(axes[0, 1])

    axes[1, 0].hist(normal_values, bins=bins, color="green", alpha=0.7, weights=np.ones_like(normal_values) * 100 / len(normal_values))
    axes[1, 0].set_title("Normal Voxels (Label = 0)")
    axes[1, 0].set_xlabel("Anomaly Score")
    axes[1, 0].set_ylabel("Percentage")
    format_percentage(axes[1, 0])

    # Fourth plot: All three histograms overlaid, normalized to percentages
    axes[1, 1].hist(anomaly_values_all, bins=bins, color="blue", alpha=0.5, label="All Voxels",
                    weights=np.ones_like(anomaly_values_all) * 100 / len(anomaly_values_all))
    axes[1, 1].hist(anomalous_values, bins=bins, color="red", alpha=0.5, label="Anomalous",
                    weights=np.ones_like(anomalous_values) * 100 / len(anomalous_values))
    axes[1, 1].hist(normal_values, bins=bins, color="green", alpha=0.5, label="Normal",
                    weights=np.ones_like(normal_values) * 100 / len(normal_values))
    # axes[1, 1].hist(anomaly_values_all, bins=bins, color="blue", alpha=0.5, label="All Voxels",
    #                 weights=np.ones_like(anomaly_values_all))
    # axes[1, 1].hist(anomalous_values, bins=bins, color="red", alpha=0.5, label="Anomalous",
    #                 weights=np.ones_like(anomalous_values))
    # axes[1, 1].hist(normal_values, bins=bins, color="green", alpha=0.5, label="Normal",
    #                 weights=np.ones_like(normal_values))
    
    axes[1, 1].set_title("Comparison of All Categories")
    axes[1, 1].set_xlabel("Anomaly Score")
    axes[1, 1].set_ylabel("Percentage")
    axes[1, 1].legend()
    format_percentage(axes[1, 1])

    # Adjust layout and save figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path)
    plt.close()


def visualize_grid(volume, n_slices = 8):
    """Visualize a 3D volume by plotting slices.
    
    Args:
        volume (torch.Tensor or np.ndarray): 3D volume to visualize.
            Should be shaped as (C, D, H, W) or (D, H, W).
        n_slices (int): Number of slices to plot.
    """
    
    if torch.is_tensor(volume):
        volume = volume.numpy() # Convert to numpy array for plotting.
    
    if volume.ndim == 4:
        volume = volume[0] # Remove channel dimension if present.
        
    depth = volume.shape[0]
    
    slice_indices = np.linspace(0, depth-1, n_slices).astype(int)
    
    subplot_rows = n_slices // 8
    subplot_cols = n_slices // subplot_rows
    
    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(20, 5 * subplot_rows))
    axs = axs.flatten()
    
    for i, idx in enumerate(slice_indices):
        if i>=len(axs) or idx>=depth:
            break
        axs[i].imshow(volume[idx, :, :], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f"Slice {idx}")
        
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/volume_grid_{suffix}.png")
    
    
def visualize_projection(volume):
    """
    Visualize maximum intensity projections (MIP) along each axis of a 3D volume.
    """
    
    if torch.is_tensor(volume):
        volume = volume.numpy() # Convert to numpy array for plotting.
    
    if volume.ndim == 4:
        volume = volume[0] # Remove channel dimension if present.
    
    # Compute maximum intensity projections along each axis
    mip_axial = np.max(volume, axis=0)  # MIP along depth axis (D)
    mip_coronal = np.max(volume, axis=1)  # MIP along height axis (H)
    mip_sagittal = np.max(volume, axis=2)  # MIP along width axis (W)

    # Plot the projections
    fig, axs = plt.subplots(1, 3, figsize=(15, 8))
    axs[0].imshow(mip_axial, cmap="gray")
    axs[0].axis("off")
    axs[0].set_title("Axial MIP (D-HW)")

    axs[1].imshow(mip_coronal, cmap="gray")
    axs[1].axis("off")
    axs[1].set_title("Coronal MIP (H-DW)")

    axs[2].imshow(mip_sagittal, cmap="gray")
    axs[2].axis("off")
    axs[2].set_title("Sagittal MIP (W-DH)")

    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/volume_projections_{suffix}.png")
    
    
def visualize_middle_slice(volume):
    """Visualize the middle slice of a 3D volume.
    
    Args:
        volume (torch.Tensor or np.ndarray): 3D volume to visualize.
            Should be shaped as (C, D, H, W) or (D, H, W).
    """
    
    if torch.is_tensor(volume):
        volume = volume.numpy() # Convert to numpy array for plotting.
    
    if volume.ndim == 4:
        volume = volume[0] # Remove channel dimension if present.
    
    depth = volume.shape[0]
    middle_slice = depth // 2
    
    plt.figure(figsize=(5, 5))
    plt.imshow(volume[middle_slice, :, :], cmap='gray')
    plt.axis('off')
    plt.title(f"Middle slice: {middle_slice}")
    
    plt.savefig(f"{output_dir}/middle_slice_{suffix}.png")
    

if __name__ == "__main__":
    conf = get_config()
    dataloader = Train_MRI_Volume(conf, mode="")
    print(f"Number of batches: {len(dataloader)}")
    batch = next(iter(dataloader))
    print("Loaded batch")
    print(f"Batch shape: {batch.shape}, expected (batch_size, channels, depth, height, width)")

    img = batch[0]

    print(f"Image shape: {batch[0].shape}, expected (channels, depth, height, width)")

    visualize_grid(img, n_slices=32)
    visualize_projection(img)
    visualize_middle_slice(img)
