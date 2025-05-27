# CGN Phase 6: Training Pipeline & Hyperparameter Optimization

## Context
You are working on the **CGN (Connectome Graph Networks)** project - an AI-augmented connectome analysis system for neuroscience research. We have successfully completed Phase 4 (baseline models) and Phase 5 (brain graph transformer), and all models are implemented and functionally tested. Now we need to implement comprehensive training pipelines and conduct rigorous performance evaluation.

## Current Project State
- **Version:** 0.1.0 → 0.2.0 (Phase 6)
- **Phase:** Training Pipeline and Hyperparameter Optimization
- **Status:** Models implemented and tested, ready for full training
- **Target:** Achieve 70-75% accuracy on ABIDE autism classification task

## Technical Stack
- **Framework:** PyTorch Lightning ≥2.0.0 for training orchestration
- **Optimization:** Optuna for hyperparameter optimization
- **Data:** ABIDE dataset (autism spectrum disorder classification)
- **Models:** Baseline classifiers, GCNs, GATs, Brain Graph Transformer
- **Evaluation:** Cross-validation, statistical testing, interpretability analysis

## Phase 6 Objectives

### 1. Training Infrastructure Setup
- Implement PyTorch Lightning training modules for all model types
- Create comprehensive data loading and preprocessing pipelines
- Set up proper train/validation/test splits with site-aware cross-validation
- Implement early stopping, checkpointing, and logging

### 2. Hyperparameter Optimization Framework
- Integrate Optuna for systematic hyperparameter search
- Define search spaces for each model architecture
- Implement multi-objective optimization (accuracy + interpretability)
- Set up automated experiment tracking

### 3. Cross-Validation Strategy
- Implement leave-one-site-out cross-validation for ABIDE dataset
- Stratified k-fold validation accounting for demographic variables
- Statistical significance testing across folds
- Robust evaluation metrics beyond accuracy

### 4. Performance Evaluation System
- Comprehensive metrics: accuracy, AUC, precision, recall, F1-score
- Confusion matrices and ROC curves
- Statistical significance testing (permutation tests, bootstrapping)
- Model comparison framework with effect sizes

### 5. Interpretability Analysis
- Attention weight visualization for GAT and Transformer models
- Graph saliency maps for important brain connections
- Feature importance analysis for baseline models
- Neurobiological interpretation of learned patterns

## Implementation Tasks

### Task 1: PyTorch Lightning Training Framework
```python
# Create training modules in src/connectome_analysis/training/
- lightning_modules.py     # Base LightningModule classes
- baseline_trainer.py      # Classical ML training wrapper
- gnn_trainer.py          # GNN training with PyTorch Lightning
- transformer_trainer.py   # Brain Transformer training
- experiment_manager.py    # Experiment orchestration
```

### Task 2: Advanced Data Pipeline
```python
# Enhance data handling in src/connectome_analysis/data/
- dataset_splits.py       # Site-aware cross-validation splits
- data_modules.py         # PyTorch Lightning DataModules
- augmentation.py         # Data augmentation for connectomes
- quality_control.py      # Enhanced QC with outlier detection
```

### Task 3: Hyperparameter Optimization
```python
# Create optimization framework in src/connectome_analysis/optimization/
- optuna_integration.py   # Optuna study management
- search_spaces.py        # Hyperparameter search space definitions
- objective_functions.py  # Multi-objective optimization
- pruning_callbacks.py    # Early stopping for hyperparameter trials
```

### Task 4: Evaluation and Metrics
```python
# Comprehensive evaluation in src/connectome_analysis/evaluation/
- metrics.py              # Extended metrics beyond accuracy
- statistical_tests.py    # Significance testing framework
- cross_validation.py     # Robust CV with demographic stratification
- model_comparison.py     # Statistical model comparison
```

### Task 5: Interpretability Framework
```python
# Interpretability tools in src/connectome_analysis/interpretability/
- attention_analysis.py   # Attention weight extraction and visualization
- saliency_maps.py        # Graph-level saliency for brain connections
- feature_importance.py   # Classical ML feature importance
- neurobio_mapping.py     # Map learned features to brain networks
```

### Task 6: Experiment Configuration
```yaml
# Enhanced configs for training experiments
configs/training/
  ├── baseline_experiments.yaml    # Classical ML configurations
  ├── gnn_experiments.yaml         # GNN hyperparameter ranges
  ├── transformer_experiments.yaml # Transformer configurations
  └── optimization_studies.yaml    # Optuna study definitions
```

## Success Criteria

### Performance Targets
- **Primary Metric:** Classification accuracy ≥70% on ABIDE test set
- **Secondary Metrics:** AUC ≥0.75, balanced accuracy ≥65%
- **Statistical Significance:** p < 0.05 across cross-validation folds
- **Generalization:** Consistent performance across different imaging sites

### Technical Requirements
- **Reproducibility:** All experiments fully reproducible with random seeds
- **Scalability:** Training pipeline handles full ABIDE dataset (~2000 subjects)
- **Efficiency:** Hyperparameter optimization completes within reasonable time
- **Robustness:** Results stable across multiple random initializations

### Scientific Rigor
- **Cross-Validation:** Leave-one-site-out and stratified k-fold validation
- **Statistical Testing:** Proper multiple comparison corrections
- **Interpretability:** Clear neurobiological interpretation of model decisions
- **Comparison:** Rigorous comparison between model architectures

## Expected Outputs

### 1. Training Infrastructure
- Complete PyTorch Lightning training framework
- Automated hyperparameter optimization system
- Robust cross-validation implementation
- Comprehensive logging and experiment tracking

### 2. Performance Results
- Model performance comparison table
- Statistical significance analysis
- ROC curves and confusion matrices
- Cross-validation stability analysis

### 3. Interpretability Analysis
- Brain connectivity importance maps
- Attention weight visualizations
- Feature importance rankings
- Neurobiological interpretation report

### 4. Documentation
- Training pipeline documentation
- Hyperparameter optimization guide
- Results interpretation manual
- Reproducibility instructions

## Implementation Strategy

### Phase 6A: Core Training Infrastructure (Week 1-2)
1. Implement PyTorch Lightning modules for all model types
2. Create robust data loading and splitting strategies
3. Set up basic training loops with proper validation
4. Implement checkpointing and early stopping

### Phase 6B: Hyperparameter Optimization (Week 3-4)
1. Integrate Optuna optimization framework
2. Define comprehensive search spaces
3. Implement multi-objective optimization
4. Run systematic hyperparameter studies

### Phase 6C: Evaluation and Analysis (Week 5-6)
1. Conduct full cross-validation experiments
2. Perform statistical significance testing
3. Generate comprehensive performance reports
4. Implement interpretability analysis

### Phase 6D: Results and Documentation (Week 7-8)
1. Analyze and interpret results
2. Create publication-ready figures
3. Write comprehensive documentation
4. Prepare for Phase 7 (advanced evaluation)

## Key Considerations

### Neuroimaging-Specific Challenges
- **Site Effects:** Account for multi-site data heterogeneity
- **Sample Size:** Handle relatively small datasets (N~2000) appropriately
- **Data Quality:** Implement robust quality control measures
- **Demographics:** Control for age, sex, and other confounding variables

### Model-Specific Requirements
- **Baseline Models:** Proper feature scaling and selection
- **GNNs:** Graph construction and node feature engineering
- **Transformer:** Positional encoding and attention pattern analysis
- **All Models:** Prevent overfitting with appropriate regularization

### Statistical Rigor
- **Multiple Comparisons:** Proper correction for multiple model comparisons
- **Cross-Validation:** Ensure no data leakage between folds
- **Effect Sizes:** Report effect sizes alongside p-values
- **Confidence Intervals:** Provide uncertainty estimates for all metrics

## File Structure Updates
```
CGN/
├── src/connectome_analysis/
│   ├── training/                    # NEW: Training infrastructure
│   ├── optimization/                # NEW: Hyperparameter optimization
│   ├── evaluation/                  # ENHANCED: Comprehensive evaluation
│   └── interpretability/            # NEW: Model interpretation tools
├── configs/training/                # NEW: Training configurations
├── experiments/                     # NEW: Experiment results
├── notebooks/training_analysis/     # NEW: Training analysis notebooks
└── scripts/training/                # NEW: Training execution scripts
```

## Next Steps After Phase 6
- **Phase 7:** Advanced evaluation with external validation datasets
- **Phase 8:** Results analysis and publication preparation
- **Phase 9:** Clinical validation and real-world deployment considerations

This phase will establish CGN as a robust, scientifically rigorous platform for connectome-based AI research with state-of-the-art performance on autism classification tasks.