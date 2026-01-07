# Unsupervised Music Clustering with VAE / Conv-VAE / CVAE using multimodal data (CSE425)

This repository contains an unsupervised learning pipeline for clustering music tracks using **Variational Autoencoders (VAE)** and multi-modal extensions using **audio + lyrics**. The project follows the CSE425 course specification: Easy (VAE on MFCC), Medium (Hybrid Conv-VAE with audio+lyrics), and Hard (Conditional/Beta-style CVAE with baselines and richer metrics).  

## Project Overview

Goal: learn compact latent representations from music (audio and optionally lyrics) and cluster tracks in latent space.

Implemented tasks:
- **Easy:** MFCC → MLP-VAE → KMeans + visualization (t-SNE/UMAP) + baseline (PCA+KMeans).
- **Medium:** log-mel spectrogram + lyrics embeddings → Hybrid Conv-VAE → clustering with KMeans / Agglomerative / DBSCAN + baseline.
- **Hard:** multi-modal **CVAE(beta)** conditioned on genre + clustering + comparison vs PCA, AE, direct spectral baseline + detailed metrics and plots.

## Repository Structure
```text
CSE425-Project/
      Dataset/
            Drive Link
            audio_to_lyrics_conversion.py
      Easy Task/
            Latent_Visualisation/
            clustering_metrics.csv
            easy_task.py
      Hard Task/
            Latent_Visualisation/
            hard_metrics_all_methods.csv
            hard_task.py
      Medium Task/
            Latent_Visualisation/
            clustering_metrics_medium_final.csv
            medium_task.py
      README.md
      requirements.txt
```

## Results (from saved CSVs)

### Easy task (MFCC)
From `clustering_metrics.csv`:
- VAE(mu)+KMeans: Silhouette **0.3248**, Calinski–Harabasz **372.26**
- PCA(8)+KMeans: Silhouette **0.1907**, Calinski–Harabasz **199.36**

### Medium task (audio+lyrics)
From `clustering_metrics_medium_final.csv`:
- Baseline PCA(16)+KMeans: Silhouette **0.1003**, Davies_Bouldin **2.6115**, ARI **0.0795**
- VAE(latent=16)+KMeans: Silhouette **0.3598**, Davies_Bouldin **0.9368**, ARI **0.1706**
- VAE(latent=16)+Agglomerative: Silhouette **0.3644**, Davies_Bouldin **0.8889**, ARI **0.1515**
- VAE(latent=16)+DBSCAN: Silhouette **0.1129**, DB **1.8234**, ARI **0.0410** (with noise points)

### Hard task (CVAE(beta))
From `hard_metrics_all_methods.csv`:
- PCA+KMeans: Silhouette **0.1003**, NMI **0.1171**, ARI **0.0795**, Purity **0.3114**
- AE+KMeans: Silhouette **0.1107**, NMI **0.1417**, ARI **0.0976**, Purity **0.3471**
- DirectSpectral(KMeans): Sil **0.1526**, NMI **0.2194**, ARI **0.1185**, Purity **0.3943**
- **CVAE(beta)+KMeans:** Sil **0.5240**, NMI **0.3741**, ARI **0.1589**, Purity **0.39**


