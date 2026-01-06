# Unsupervised Music Clustering with VAE / Conv-VAE / CVAE using (CSE425)

This repository contains an unsupervised learning pipeline for clustering music tracks using **Variational Autoencoders (VAE)** and multi-modal extensions using **audio + lyrics**. The project follows the CSE425 course specification: Easy (VAE on MFCC), Medium (Hybrid Conv-VAE with audio+lyrics), and Hard (Conditional/Beta-style CVAE with baselines and richer metrics).  

## Project Overview

Goal: learn compact latent representations from music (audio and optionally lyrics) and cluster tracks in latent space.

Implemented tasks:
- **Easy:** MFCC → MLP-VAE → KMeans + visualization (t-SNE/UMAP) + baseline (PCA+KMeans).
- **Medium:** log-mel spectrogram + lyrics embeddings → Hybrid Conv-VAE → clustering with KMeans / Agglomerative / DBSCAN + baseline.
- **Hard:** multi-modal **CVAE(beta)** conditioned on genre + clustering + comparison vs PCA, AE, direct spectral baseline + detailed metrics and plots.

## Results (from saved CSVs)

### Easy task (MFCC)
From `clustering_metrics.csv`:
- VAE(mu)+KMeans: Silhouette **0.3248**, Calinski–Harabasz **372.26**
- PCA(8)+KMeans: Silhouette **0.1907**, Calinski–Harabasz **199.36**

### Medium task (audio+lyrics)
From `clustering_metrics_medium_final.csv`:
- Baseline PCA(16)+KMeans: Silhouette 0.1003, DB 2.6115, ARI 0.0795
- VAE(latent=16)+KMeans: Silhouette 0.3420, DB 1.0133, ARI 0.1582
- VAE(latent=16)+Agglomerative: Silhouette 0.3462, DB 0.9252, ARI 0.1426
- VAE(latent=16)+DBSCAN: Silhouette 0.0828, DB 2.0917, ARI 0.0904 (with noise points)

### Hard task (CVAE(beta))
From `hard_metrics_all_methods.csv`:
- PCA+KMeans: Sil 0.1003, NMI 0.1171, ARI 0.0795, Purity 0.3114
- AE+KMeans: Sil 0.1292, NMI 0.1991, ARI 0.1304, Purity 0.3814
- DirectSpectral(KMeans): Sil 0.1526, NMI 0.2194, ARI 0.1185, Purity 0.3943
- **CVAE(beta)+KMeans:** Sil **0.3434**, NMI **0.4197**, ARI **0.1697**, Purity **0.4471**


