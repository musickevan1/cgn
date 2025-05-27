# CGN Phase 4 Development: Baseline Models and Initial Graph Neural Networks

## Project Context
You are working on the **CGN (Connectome Graph Networks)** project - an AI-augmented connectome analysis system for neuroscience research. The project uses graph neural networks and transformers for neuroimaging analysis, specifically targeting autism spectrum disorder (ASD) research with the ABIDE dataset.

**Current Status:**
- Version: 0.1.0 (Development Phase)
- Core infrastructure complete: data loading, preprocessing, connectome construction
- Ready for Phase 4: Baseline Models and Initial GNNs
- Focus: ABIDE dataset only (ASD vs Control classification)

## Technical Stack
- **Python 3.8-3.10**
- **PyTorch ≥2.0.0, PyTorch Geometric ≥2.3.0, PyTorch Lightning ≥2.0.0**
- **Neuroimaging:** Nilearn ≥0.10.0, Nibabel ≥5.0.0
- **Graph Analysis:** NetworkX ≥3.0
- **Scientific Computing:** NumPy, SciPy, Pandas, Scikit-learn

## Project Structure
```
CGN/
├── src/connectome_analysis/
│   ├── data/                    # COMPLETE (loaders.py, preprocessing.py, connectome.py)
│   ├── models/                  # TO IMPLEMENT (Phase 4)
│   ├── training/                # TO IMPLEMENT (Phase 4)
│   ├── evaluation/              # TO IMPLEMENT (Phase 4)
│   ├── utils/                   # Utilities
│   └── visualization/           # Plotting functions
├── configs/                     # Configuration files
├── scripts/                     # Utility scripts
├── notebooks/                   # Jupyter notebooks
├── tests/                       # Test suite
└── results/                     # Experimental results
```

## Phase 4 Objectives

### 1. Baseline Classifiers Implementation
Create `src/connectome_analysis/models/baseline.py`:

**Requirements:**
- Implement classical ML models: SVM, Random Forest, Logistic Regression
- Use connectivity matrices as flattened feature vectors
- Include proper feature scaling and selection
- Support both correlation and partial correlation connectomes
- Implement cross-validation with site stratification for ABIDE

**Key Classes:**
```python
class BaselineClassifier:
    """Base class for classical ML classifiers on connectome data"""
    
class SVMClassifier(BaselineClassifier):
    """Support Vector Machine for connectome classification"""
    
class RandomForestClassifier(BaselineClassifier):
    """Random Forest for connectome classification"""
    
class LogisticRegressionClassifier(BaselineClassifier):
    """Logistic Regression for connectome classification"""
```

**Features:**
- Handle upper triangular connectivity matrices (avoid redundancy)
- Feature selection methods (variance threshold, univariate selection)
- Hyperparameter optimization with grid search
- Cross-validation respecting ABIDE site structure
- Performance metrics: accuracy, AUC, precision, recall, F1

### 2. Graph Neural Networks Implementation
Create `src/connectome_analysis/models/gnn.py`:

**Requirements:**
- Implement GCN (Graph Convolutional Network) for graph classification
- Implement GAT (Graph Attention Network) with interpretable attention
- Support PyTorch Geometric data format
- Graph-level classification for subject prediction (ASD vs Control)
- Include graph pooling mechanisms (global mean, attention pooling)

**Key Classes:**
```python
class BrainGCN(torch.nn.Module):
    """Graph Convolutional Network for brain connectome classification"""
    
class BrainGAT(torch.nn.Module):
    """Graph Attention Network for brain connectome classification"""
    
class GraphPooling(torch.nn.Module):
    """Graph pooling mechanisms for graph-level representations"""
```

**Architecture Specifications:**
- 2-3 graph convolutional layers
- ReLU activation, dropout for regularization
- Global pooling (mean/max/attention) for graph-level features
- Final classification layer (sigmoid for binary classification)
- Support for both node features and edge weights

### 3. Training Pipeline Implementation
Create `src/connectome_analysis/training/trainer.py`:

**Requirements:**
- PyTorch Lightning-based training framework
- Support for both baseline and GNN models
- Proper data splitting with site stratification
- Early stopping and model checkpointing
- Comprehensive logging and metrics tracking

**Key Classes:**
```python
class ConnectomeTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for connectome models"""
    
class CrossValidator:
    """Cross-validation with site stratification for ABIDE"""
    
class ModelOptimizer:
    """Hyperparameter optimization using Optuna"""
```

**Training Features:**
- Leave-one-site-out cross-validation
- Stratified k-fold with site considerations
- Learning rate scheduling
- Gradient clipping for stability
- Multi-GPU support (if available)

### 4. Evaluation Framework Implementation
Create `src/connectome_analysis/evaluation/metrics.py`:

**Requirements:**
- Comprehensive evaluation metrics for neuroimaging classification
- Statistical significance testing
- Model interpretability analysis (especially for attention mechanisms)
- Comparison frameworks for baseline vs GNN performance

**Key Functions:**
```python
def compute_classification_metrics(y_true, y_pred, y_prob):
    """Compute accuracy, AUC, precision, recall, F1"""
    
def statistical_significance_test(results_dict):
    """Test for significant differences between models"""
    
def interpret_gat_attention(model, data_loader):
    """Extract and analyze GAT attention weights"""
    
def plot_roc_curves(results_dict):
    """Plot ROC curves for model comparison"""
```

### 5. Configuration Management
Create `configs/model_configs.yaml`:

**Requirements:**
- Centralized configuration for all models and training
- Hyperparameter specifications
- Data loading and preprocessing settings
- Evaluation and cross-validation parameters

**Configuration Structure:**
```yaml
data:
  dataset: "abide"
  atlas: "aal"  # 116 regions
  connectivity_method: "correlation"
  
baseline_models:
  svm:
    C: [0.1, 1.0, 10.0]
    kernel: ["rbf", "linear"]
  random_forest:
    n_estimators: [100, 200, 500]
    max_depth: [10, 20, None]
    
gnn_models:
  gcn:
    hidden_dim: [64, 128, 256]
    num_layers: [2, 3]
    dropout: [0.2, 0.5]
  gat:
    hidden_dim: [64, 128]
    num_heads: [4, 8]
    dropout: [0.2, 0.5]
    
training:
  batch_size: 32
  learning_rate: 0.001
  max_epochs: 200
  patience: 20
  
evaluation:
  cv_folds: 5
  test_size: 0.2
  stratify_by_site: true
```

## Implementation Guidelines

### Code Quality Standards
- **Type hints** throughout all new code
- **Comprehensive docstrings** with neuroimaging context
- **Error handling** with informative messages
- **Modular design** following existing project structure
- **Unit tests** for critical functions

### Neuroimaging Considerations
- **ABIDE multi-site nature**: Ensure models don't overfit to site-specific patterns
- **Balanced datasets**: Handle class imbalance if present
- **Connectivity thresholding**: Consider sparse vs dense connectivity matrices
- **Atlas choice**: Default to AAL-116, but support multiple atlases
- **Quality control**: Integrate with existing QC metrics from preprocessing

### Performance Targets
- **Baseline Models**: 60-65% accuracy (literature baseline)
- **GNN Models**: 70-75% accuracy (state-of-the-art target)
- **Statistical Significance**: Proper statistical testing between models
- **Reproducibility**: Seed management for consistent results

### Scientific Rigor
- **Cross-validation**: Never let the same subject appear in train and test
- **Site effects**: Account for scanner/site differences in evaluation
- **Multiple comparisons**: Appropriate statistical corrections
- **Interpretability**: Especially for attention mechanisms in GAT

## Deliverables for Phase 4

1. **Working baseline classifiers** with ABIDE evaluation
2. **Functional GCN and GAT models** for brain graph classification  
3. **Training pipeline** with proper cross-validation
4. **Evaluation framework** with statistical testing
5. **Configuration system** for reproducible experiments
6. **Documentation** and example usage scripts
7. **Unit tests** for critical components
8. **Results analysis** comparing baseline vs GNN performance

## Expected Outcomes
- Establish performance benchmarks on ABIDE dataset
- Demonstrate GNN superiority over classical methods
- Create foundation for Phase 5 (Brain Graph Transformers)
- Generate interpretable results showing which brain connections matter for ASD classification

## Next Steps After Phase 4
- Phase 5: Custom Brain Graph Transformer architecture
- Phase 6: Advanced training with hyperparameter optimization  
- Phase 7: Comprehensive evaluation and cross-validation
- Phase 8: Results analysis and publication materials

Remember: This is rapid research prototyping with 1-2 month development cycles. Balance scientific rigor with development speed, and ensure all implementations are ready for neuroscience research use.

## Getting Started
1. Implement baseline classifiers first to establish benchmarks
2. Create the training pipeline and evaluation framework
3. Implement GCN as the simpler GNN architecture
4. Add GAT with attention mechanisms
5. Run comprehensive evaluations and statistical comparisons

Focus on creating a solid foundation that other researchers can build upon, with clear documentation and reproducible results.