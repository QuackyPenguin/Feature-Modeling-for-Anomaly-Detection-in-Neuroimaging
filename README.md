# Feature Modeling for Anomaly Detection in Neuroimaging

Voxel-wise anomaly detection in 3D brain MRI via **self-supervised feature learning** (SimCLR-style) and **per-voxel Gaussian modeling** with Mahalanobis scoring. The method yields **explainable heatmaps** (distances to healthy feature distributions) and a simple, reproducible pipeline. The entire report can be found in this repository: Research project - Denis Kovacevic.pdf

> **TL;DR**: Pretrain a 3D CNN on healthy MRIs (contrastive learning) → extract multi-scale features → fit per-voxel Gaussians on healthy data (Welford updates) → score anomalies with Mahalanobis distance + quantile–sigmoid normalization.
