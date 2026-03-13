# CGN: Connectome Graph Networks

Applying graph neural networks and transformer architectures to brain connectivity data for autism spectrum disorder classification.

---

## Overview

CGN investigates whether graph-based deep learning can identify meaningful patterns in brain connectivity that distinguish individuals with autism spectrum disorder (ASD) from neurotypical controls. The project builds an end-to-end pipeline -- from raw fMRI preprocessing through connectome construction to classification -- using the ABIDE (Autism Brain Imaging Data Exchange) neuroimaging dataset. The core hypothesis is that representing brain connectivity as a graph, rather than a flat feature vector, allows models to capture topological structure that traditional classifiers miss.

## Research Questions

- Can graph neural networks outperform traditional ML baselines (SVM, Random Forest, Logistic Regression) on ASD vs. control classification from functional connectomes?
- Do graph attention mechanisms and transformer architectures learn to attend to neurobiologically relevant brain regions and connections?
- Which connectivity patterns and brain network properties (efficiency, modularity, clustering, small-worldness) are most predictive of ASD diagnosis?

## Approach

**Data.** The ABIDE dataset provides resting-state fMRI scans with phenotypic data across multiple acquisition sites. Preprocessing follows a standard pipeline: brain masking, temporal filtering (bandpass 0.01--0.1 Hz), z-score standardization, and ROI time series extraction using the AAL atlas (116 regions). Functional connectivity matrices are computed via correlation, partial correlation, and tangent space methods. Site-aware cross-validation accounts for multi-site acquisition variability.

**Models.** Three tiers of classifiers are compared:

- *Baselines* -- SVM (RBF and linear kernels), Random Forest, and Logistic Regression operating on vectorized upper-triangular connectivity features
- *Graph Neural Networks* -- BrainGCN (Graph Convolutional Network) and BrainGAT (Graph Attention Network) that operate directly on brain connectivity graphs, using global mean pooling for graph-level classification
- *Brain Graph Transformer* -- A transformer encoder with positional encoding applied to graph node embeddings, converting between sparse PyG graph representations and dense batched sequences for multi-head self-attention

**Pipeline.** The full pipeline is managed through YAML experiment configs and PyTorch Lightning, with Optuna integration for hyperparameter optimization across all model families. Evaluation uses stratified k-fold and leave-one-site-out cross-validation, reporting accuracy, AUC-ROC, F1, precision, recall, MCC, and Cohen's kappa. An interpretability module supports attention weight extraction, saliency maps, feature importance analysis, and neurobiological mapping back to atlas-labeled brain regions.

## Project Structure

- `src/connectome_analysis/` -- Core package: data loading and preprocessing, model architectures (baseline, GNN, transformer), training infrastructure, evaluation metrics, optimization, and interpretability
- `configs/` -- YAML experiment definitions for baseline, GNN, transformer, and hyperparameter optimization studies
- `notebooks/` -- Validation notebooks for data, baselines, GNNs, and full pipeline integration
- `scripts/` -- Data download/validation, experiment runners, and benchmarking utilities

## Tech Stack

Python, PyTorch, PyTorch Geometric, PyTorch Lightning, Transformers, Nilearn, Nibabel, NetworkX, scikit-learn, Optuna, Docker (with CUDA support)

## Status

Active development. Data loading, preprocessing, connectome construction, and all model architectures (baselines, GCN, GAT, Brain Graph Transformer) are implemented and functionally tested. The training infrastructure, cross-validation framework, and hyperparameter optimization pipeline are built out. Current work focuses on running full-scale experiments on the complete ABIDE dataset and refining the transformer's graph-to-sequence batching for proper per-graph attention. No benchmark results to report yet.

## License

MIT
