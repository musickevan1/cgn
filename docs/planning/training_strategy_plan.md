# Training Strategy Plan

## Overview
This document outlines the two-tier training approach for the CGN project, leveraging both local and cloud resources.

## Hybrid Training Architecture

### Tier 1: Local Development (MacBook CPU)
- **Purpose:** Baseline models, initial GNN prototypes, code development, debugging, data preprocessing, small-scale experiments, results analysis.
- **Characteristics:**
    - Baseline models (SVM, Random Forest, Logistic Regression)
    - Initial GNN prototypes (GCN, GAT with small datasets)
    - Code development and debugging
    - Data preprocessing and connectome construction
    - Small-scale experiments (<100 subjects, <30min training)
    - Results analysis and visualization

### Tier 2: Cloud Training (RunPod GPU)
- **Purpose:** Large-scale GNN training, Brain Graph Transformers, hyperparameter optimization, multi-modal models, extended training runs, parallel experiment execution.
- **Characteristics:**
    - Large-scale GNN training (full ABIDE/ADHD-200 datasets)
    - Brain Graph Transformers (Phase 5+)
    - Hyperparameter optimization (50+ combinations)
    - Multi-modal models (structural + functional connectivity)
    - Extended training runs (>1 hour)
    - Parallel experiment execution

## Resource Allocation
- **Local:** Ideal for rapid iteration, debugging, and smaller experiments.
- **Cloud:** Essential for computationally intensive tasks and large-scale studies.

## Considerations
- Data transfer between local and cloud environments.
- Cost efficiency of cloud resources.
- Scalability of the training pipeline.
